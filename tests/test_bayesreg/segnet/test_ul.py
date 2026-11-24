"""The profile v3_large of the segmenter and the tools of work package UL (runs/20261104-ul/plan.md; docs/bayesreg/04_分割器.md §15). CPU, synthetic small volumes (ids OASIS_OAS1_09xx_MR1), one thread.

Real repository files are read for subject ids and the seg35 label table only; no image, no label of a real subject. No test starts a unit, uses a GPU or runs a real training: the two real steps are called in a subprocess with a blocking systemd-run first on PATH and must stop at the snapshot check.
"""
from dataclasses import asdict, fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jd_curl.validation import InputValidationError

from bayesreg.eval.truth import load_sealed_lists
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.io.pair import load_pair, load_segmenter_registry, validate_segmentation_provenance
from bayesreg.seg.unet import cli, large, pins
from bayesreg.seg.unet import train as train_module
from bayesreg.seg.unet.calibrate import calibrate_member
from bayesreg.seg.unet.data import LabelAccessRefused, LabelGuard, OasisFiles, Subject, load_folds, load_full_folds, parse_folds, read_labels, random_patch
from bayesreg.seg.unet.infer import load_member, read_probabilities, segment_subject
from bayesreg.seg.unet.large import (GIB, MODEL_PREFIX_V3, PATCH_U1, SEGMENTER_ID_V3, TrainConfigLarge, UNet3DDeepSupervision, cap_gpu_memory, check_flip_axis, class_balanced_patch, deep_supervision_loss, ds_weights, flip_left_right, left_right_label_map, load_label_names,
                                     require_patch, train_large_member, window_starts, windows_per_volume)
from bayesreg.seg.unet.model import UNet3D, build_model
from bayesreg.seg.unet.registry import register, verify_members
from bayesreg.seg.unet.snapshot import CODE_ROOT, SNAPSHOT_PATHS, SNAPSHOT_PATHS_V2, SNAPSHOT_PATHS_V3, SNAPSHOT_PROFILES
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, write_json, write_pair
from bayesreg.seg.unet.train import TrainConfig, dice_ce_loss, gaussian_window, predict_logits, train_member

from . import _ul_old_fingerprint as old
from ._helpers import A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV, T0, TINY, TRAIN_SOURCE, TRAINING, tiny_subject

UL = CODE_ROOT/"experiments"/"ul"
sys.path[:0] = [str(CODE_ROOT/"experiments"/"unv2"), str(UL)]
import dev_lock  # noqa: E402


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load_script("ul_run", UL/"run_ul.py")
EVAL = load_script("ul_eval_u1", UL/"eval_u1.py")
DECIDE = load_script("ul_decide", UL/"decide_ul.py")
EST = load_script("ul_estimate", UL/"estimate.py")
TREE = load_script("ul_tree", CODE_ROOT/"experiments"/"unv2"/"synthetic_tree.py")
SEG = EVAL.eval_seg
NAMES = load_label_names()
TINY_LARGE = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[8, 8, 8], max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False, flip_axis_check=False)


def quiet(_):
    pass


def tiny_sets():
    return [tiny_subject(s, shape=(16, 16, 16), seed=i) for i, s in enumerate(A_TRAIN)], [tiny_subject(A_CALIB[0], shape=(16, 16, 16), seed=9)]


def lateral_volume(shape=(32, 36, 40), *, swapped=False, axis=2):
    """Labels of a head: the low half of `axis` holds the left cerebral structures (labels by their names), the high half the right ones, a midline column of the brain stem. swapped: the labels are on the wrong side."""
    ids = {v: k for k, v in NAMES.items()}
    labels = np.zeros(shape, dtype=np.uint8)
    n = shape[axis]
    index = np.arange(n)
    side = [1, 1, 1]
    side[axis] = n
    low = np.broadcast_to((index < n//2).reshape(side), shape)
    grid = np.indices(shape)
    wm = (grid[(axis+1) % 3] < shape[(axis+1) % 3]//2)
    left_wm, left_ctx, right_wm, right_ctx = ids["Left-Cerebral-White-Matter"], ids["Left-Cerebral-Cortex"], ids["Right-Cerebral-White-Matter"], ids["Right-Cerebral-Cortex"]
    if swapped:
        left_wm, left_ctx, right_wm, right_ctx = right_wm, right_ctx, left_wm, left_ctx
    labels[low & wm] = left_wm
    labels[low & ~wm] = left_ctx
    labels[~low & wm] = right_wm
    labels[~low & ~wm] = right_ctx
    # thin the lateral structures towards the midline so the left and right centroids are far apart along `axis`
    away = np.abs(grid[axis]-(n-1)/2) < n/8
    labels[away] = ids["Brain-Stem"]
    return labels


# ---- the left-right pairing, from the label table ------------------------------------------------------------------------------------------------------------

def test_map_read_from_the_table():
    lut, pairs, unpaired = left_right_label_map()
    assert len(lut) == 36 and len(pairs) == 16
    assert sorted(NAMES[u] for u in unpaired) == sorted(["Unknown", "3rd-Ventricle", "4th-Ventricle", "Brain-Stem"])
    for left, right in pairs:
        assert NAMES[left].startswith("Left-") and NAMES[right] == "Right-"+NAMES[left][5:] and lut[left] == right and lut[right] == left
    assert np.array_equal(lut[lut], np.arange(36))
    assert all(lut[u] == u for u in unpaired)
    assert sorted(l for p in pairs for l in p)+sorted(unpaired) != [] and sorted([l for p in pairs for l in p]+unpaired) == list(range(36))


def test_table_of_the_config_is_the_label_file():
    table = json.loads(large.GROUPS_FILE.read_text(encoding="utf-8"))["label_table"]
    path = Path("/mnt/d/data/seg35_labels.txt")
    if not path.is_file():
        pytest.skip("the label table file is not mounted")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == table["sha256"]
    read = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            read[int(parts[0])] = parts[1]
    assert read == NAMES


@pytest.mark.parametrize("names,match", [
    ({0: "Unknown", 1: "Left-A", 2: "Right-B"}, "no partner"),
    ({0: "Unknown", 1: "Right-A"}, "no partner on the left"),
    ({0: "Unknown", 1: "lh-A", 2: "rh-A"}, "form this rule does not read"),
    ({0: "Unknown", 1: "Cortex_left"}, "form this rule does not read"),
    ({0: "Unknown", 1: "Same", 2: "Same"}, "listed twice"),
])
def test_map_refuses_what_it_cannot_read(names, match):
    with pytest.raises(InputValidationError, match=match):
        left_right_label_map(names)


def test_map_accepts_a_table_it_can_read():
    lut, pairs, unpaired = left_right_label_map({0: "Unknown", 1: "Left-A", 2: "Midline", 3: "Right-A"})
    assert lut.tolist() == [0, 3, 2, 1] and pairs == [(1, 3)] and unpaired == [0, 2]


# ---- the flip -------------------------------------------------------------------------------------------------------------------------------------------------

def lut_tensor():
    return torch.as_tensor(left_right_label_map()[0].astype(np.int64))


def test_two_flips_restore_image_and_labels():
    rng = np.random.default_rng(0)
    image = torch.as_tensor(rng.random((1, 1, 6, 7, 8)).astype(np.float32))
    labels = torch.as_tensor(rng.integers(0, 36, (1, 6, 7, 8)))
    a, b = flip_left_right(image, labels, lut_tensor())
    assert not torch.equal(a, image) and not torch.equal(b, labels)
    c, d = flip_left_right(a, b, lut_tensor())
    assert torch.equal(c, image) and torch.equal(d, labels)


def test_flip_swaps_the_labels_and_keeps_the_anatomy_on_its_side():
    labels = torch.as_tensor(lateral_volume().astype(np.int64))[None]
    image = (labels.float()/36)[None]
    flipped_image, flipped = flip_left_right(image, labels, lut_tensor())
    ids = {v: k for k, v in NAMES.items()}
    n = labels.shape[-1]
    # a voxel at x carries, after the flip, the mirror of the label that was at n-1-x
    for z, y, x in ((0, 0, 0), (3, 30, 2), (20, 4, 37)):
        before = int(labels[0, z, y, n-1-x])
        assert int(flipped[0, z, y, x]) == int(lut_tensor()[before])
        assert float(flipped_image[0, 0, z, y, x]) == float(image[0, 0, z, y, n-1-x])
    # the left structures are still on the left (low x), which is what makes the flipped sample anatomically right
    low, high = flipped[0, :, :, :n//2], flipped[0, :, :, n//2:]
    for left, right in (("Left-Cerebral-Cortex", "Right-Cerebral-Cortex"), ("Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter")):
        assert (low == ids[left]).sum() > 0 and (high == ids[left]).sum() == 0
        assert (high == ids[right]).sum() > 0 and (low == ids[right]).sum() == 0
    # a midline label is not swapped
    assert (flipped == ids["Brain-Stem"]).sum() == (labels == ids["Brain-Stem"]).sum()


def test_flip_without_the_swap_would_put_the_labels_on_the_wrong_side():
    """The counterexample the swap exists for: the plain mirror moves Left labels to high x."""
    labels = torch.as_tensor(lateral_volume().astype(np.int64))[None]
    ids = {v: k for k, v in NAMES.items()}
    mirrored = torch.flip(labels, (-1,))
    n = labels.shape[-1]
    assert (mirrored[0, :, :, n//2:] == ids["Left-Cerebral-Cortex"]).sum() > 0 and (mirrored[0, :, :, :n//2] == ids["Left-Cerebral-Cortex"]).sum() == 0


def test_flip_axis_check_passes_on_lateral_labels():
    record = check_flip_axis([Subject("s", None, lateral_volume())], NAMES)
    assert record["n_subjects"] == 1 and record["smallest_x_separation_vox"] > 10 and sorted(NAMES[a] for a, _ in record["pairs"]) == ["Left-Cerebral-Cortex", "Left-Cerebral-White-Matter"]


def test_flip_axis_check_refuses_labels_on_the_wrong_side():
    with pytest.raises(InputValidationError, match="not trained"):
        check_flip_axis([Subject("s", None, lateral_volume()), Subject("t", None, lateral_volume(swapped=True))], NAMES)


def test_flip_axis_check_refuses_another_axis():
    with pytest.raises(InputValidationError, match="not trained"):
        check_flip_axis([Subject("s", None, lateral_volume(axis=1))], NAMES)


def test_flip_axis_check_refuses_a_missing_structure():
    labels = lateral_volume()
    labels[labels == {v: k for k, v in NAMES.items()}["Left-Cerebral-Cortex"]] = 0
    with pytest.raises(InputValidationError, match="no voxel"):
        check_flip_axis([Subject("s", None, labels)], NAMES)


# ---- deep supervision ----------------------------------------------------------------------------------------------------------------------------------------

def ds_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    return UNet3DDeepSupervision(**{**dict(base_channels=4, depth=3, ds_levels=2), **kwargs})


def test_outputs_in_training_and_in_eval():
    x = torch.zeros(2, 1, 16, 24, 32)
    model = ds_model().train()
    outputs = model(x)
    assert [tuple(o.shape) for o in outputs] == [(2, 36, 16, 24, 32), (2, 36, 8, 12, 16), (2, 36, 4, 6, 8)]
    model.eval()
    assert isinstance(model(x), torch.Tensor) and model(x).shape == (2, 36, 16, 24, 32)


def test_parent_weights_are_those_of_a_plain_unet_and_the_checkpoint_is_plain():
    torch.manual_seed(0)
    plain = UNet3D(base_channels=4, depth=3)
    model = ds_model()
    state, plain_state = model.plain_state_dict(), plain.state_dict()
    assert set(state) == set(plain_state) and all(torch.equal(state[k], plain_state[k]) for k in state)
    assert any(k.startswith("aux_heads.") for k in model.state_dict())
    rebuilt = build_model(dict(base_channels=4, depth=3))
    rebuilt.load_state_dict(state, strict=True)
    x = torch.as_tensor(np.random.default_rng(0).random((1, 1, 8, 8, 16)).astype(np.float32))
    assert torch.equal(rebuilt.eval()(x), model.eval()(x))
    assert torch.equal(model.train()(x)[0], plain.train()(x))           # instance norm has no running statistics


def test_ds_levels_must_fit_the_depth():
    with pytest.raises(ValueError):
        UNet3DDeepSupervision(base_channels=4, depth=2, ds_levels=2)
    assert len(ds_model(ds_levels=0).train()(torch.zeros(1, 1, 16, 16, 16))) == 1


def test_ds_weights():
    assert ds_weights(3) == pytest.approx([4/7, 2/7, 1/7])
    assert sum(ds_weights(5)) == pytest.approx(1) and ds_weights(1) == [1.]


def test_loss_is_the_weighted_sum_of_the_levels():
    g = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 6, (2, 8, 8, 8), generator=g)
    outputs = [torch.randn(2, 6, 8, 8, 8, generator=g, requires_grad=True), torch.randn(2, 6, 4, 4, 4, generator=g, requires_grad=True), torch.randn(2, 6, 2, 2, 2, generator=g, requires_grad=True)]
    loss, parts = deep_supervision_loss(outputs, labels, n_classes=6)
    manual = 0.
    for k, (output, weight) in enumerate(zip(outputs, (4/7, 2/7, 1/7))):
        manual = manual+weight*dice_ce_loss(output, labels[:, ::2**k, ::2**k, ::2**k], n_classes=6)[0]
    assert float(loss) == pytest.approx(float(manual))
    assert parts["ce"] == pytest.approx(dice_ce_loss(outputs[0], labels, n_classes=6)[1]["ce"]) and len(parts["level_losses"]) == 3
    grads = torch.autograd.grad(loss, outputs)
    assert all(float(g.abs().sum()) > 0 for g in grads)


def test_labels_of_a_level_are_the_nearest_labels():
    labels = torch.randint(0, 6, (1, 8, 8, 8))
    sampled = F.interpolate(labels[None].float(), scale_factor=.5, mode="nearest")[0].long()
    assert torch.equal(sampled, labels[:, ::2, ::2, ::2])


def test_loss_refuses_a_level_that_does_not_fit():
    with pytest.raises(ValueError, match="do not fit"):
        deep_supervision_loss([torch.zeros(1, 6, 8, 8, 8), torch.zeros(1, 6, 3, 4, 4)], torch.zeros(1, 8, 8, 8, dtype=torch.long), n_classes=6)


def test_auxiliary_heads_train():
    model = ds_model(depth=2, ds_levels=1).train()
    outputs = model(torch.zeros(1, 1, 8, 8, 8)+.5)
    loss, _ = deep_supervision_loss(outputs, torch.ones(1, 8, 8, 8, dtype=torch.long), n_classes=36)
    loss.backward()
    assert float(model.aux_heads[0].weight.grad.abs().sum()) > 0 and float(model.head.weight.grad.abs().sum()) > 0


# ---- patches -----------------------------------------------------------------------------------------------------------------------------------------------------

def rare_volume(shape=(32, 32, 32), rare=(30, 1, 2), rare_label=19, big_label=1):
    labels = torch.full((1,)+shape, big_label, dtype=torch.long)
    labels[0, :4, :4, :4] = 0
    labels[(0,)+tuple(rare)] = rare_label
    return torch.zeros((1, 1)+shape), labels


def test_patch_shape_and_bounds():
    image, labels = rare_volume()
    rng = np.random.default_rng(0)
    for _ in range(100):
        x, y, record = class_balanced_patch(image, labels, (8, 16, 12), rng)
        assert x.shape == (1, 1, 8, 16, 12) and y.shape == (1, 8, 16, 12)
        assert all(0 <= s <= n-p for s, n, p in zip(record["start"], (32, 32, 32), (8, 16, 12)))


def test_foreground_draws_a_class_not_a_voxel():
    """A one-voxel class is drawn about as often as the big one. data.random_patch, the v1 sampler, draws a voxel over all labels > 0 and almost never lands on it: the counterexample."""
    image, labels = rare_volume()
    rng = np.random.default_rng(1)
    records = [class_balanced_patch(image, labels, (8, 8, 8), rng, p_foreground=1.)[2] for _ in range(400)]
    assert all(r["foreground"] for r in records)
    share = np.mean([r["label"] == 19 for r in records])
    assert .4 < share < .6
    rng = np.random.default_rng(1)
    centred = 0
    for _ in range(100):
        x, y, record = class_balanced_patch(image, labels, (8, 8, 8), rng, p_foreground=1.)
        if record["label"] == 19:
            centred += 1
            assert (y == 19).sum() == 1                                  # shifted inside the volume, the voxel still in the patch
    assert centred > 0
    rng = np.random.default_rng(1)
    hits = sum(int((random_patch(image, labels, (8, 8, 8), rng, p_foreground=1.)[1] == 19).sum()) for _ in range(100))
    assert hits < 5


def test_uniform_patches_are_not_foreground_and_cover_the_range():
    image, labels = rare_volume()
    rng = np.random.default_rng(2)
    starts = np.array([class_balanced_patch(image, labels, (8, 8, 8), rng, p_foreground=0.)[2]["start"] for _ in range(300)])
    assert starts.min(axis=0).tolist() == [0, 0, 0] and starts.max(axis=0).tolist() == [24, 24, 24]


def test_foreground_share_follows_the_probability():
    image, labels = rare_volume()
    rng = np.random.default_rng(3)
    share = np.mean([class_balanced_patch(image, labels, (8, 8, 8), rng)[2]["foreground"] for _ in range(600)])
    assert abs(share-1/3) < .06


def test_background_only_sample_falls_back_to_a_uniform_patch():
    rng = np.random.default_rng(4)
    _, _, record = class_balanced_patch(torch.zeros(1, 1, 16, 16, 16), torch.zeros(1, 16, 16, 16, dtype=torch.long), (8, 8, 8), rng, p_foreground=1.)
    assert record["foreground"] is False


def test_patch_larger_than_the_volume_is_refused():
    image, labels = rare_volume()
    with pytest.raises(InputValidationError, match="larger than volume"):
        class_balanced_patch(image, labels, (64, 8, 8), np.random.default_rng(0))


@pytest.mark.parametrize("patch", [(128, 160, 100), (128, 160), None])
def test_patch_config_refused(patch):
    with pytest.raises(InputValidationError):
        require_patch(TrainConfigLarge(patch_size=patch))


def test_batch_size_refused():
    with pytest.raises(InputValidationError):
        require_patch(TrainConfigLarge(batch_size=0))


def test_u1_defaults():
    cfg = TrainConfigLarge()
    assert cfg.base_channels == 32 and cfg.depth == 4 and cfg.max_channels == 320 and tuple(cfg.patch_size) == PATCH_U1 == (128, 160, 112) and cfg.batch_size == 2 and cfg.ds_levels == 2 and cfg.flip_probability == .5
    assert cfg.lr == TrainConfig().lr and cfg.weight_decay == TrainConfig().weight_decay and cfg.poly_power == TrainConfig().poly_power and cfg.augment == TrainConfig().augment and cfg.grad_clip == TrainConfig().grad_clip
    assert cfg.model_config() == dict(base_channels=32, depth=4, n_classes=36, max_channels=320)
    assert require_patch(cfg) == PATCH_U1 and cfg.flip_axis_check is True and cfg.gpu_cap_gib == 20.


# ---- the sliding window ---------------------------------------------------------------------------------------------------------------------------------------------

class CountingModel(torch.nn.Module):
    """Logits = the call number (from 1) in every voxel and class; records the calls."""
    multiple, n_classes = 4, 3

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape[2:]))
        return torch.full((x.shape[0], self.n_classes)+tuple(x.shape[2:]), float(len(self.calls)))


def test_windows_of_the_real_volume():
    n, starts = windows_per_volume((192, 224, 160), PATCH_U1, .5)
    assert n == 8 and starts == [[0, 64], [0, 64], [0, 48]]
    assert windows_per_volume((192, 224, 160), PATCH_U1, .25)[0] == 8               # the volume is too small for .25 to change anything
    for size, window, axis in zip((192, 224, 160), PATCH_U1, starts):
        covered = set()
        for s in axis:
            covered |= set(range(s, s+window))
        assert covered == set(range(size))


@pytest.mark.parametrize("sigma", [None, .125])
def test_sliding_window_runs_every_window_and_covers_every_voxel(sigma):
    model = CountingModel()
    logits = predict_logits(model, np.zeros((18, 22, 20), dtype=np.float32), device="cpu", amp=False, window=(8, 8, 8), gaussian_sigma=sigma)
    assert logits.shape == (1, 3, 18, 22, 20)
    want, _ = windows_per_volume((20, 24, 20), (8, 8, 8), .5)                       # padded to multiples of 4
    assert len(model.calls) == want and set(model.calls) == {(8, 8, 8)}
    assert bool(torch.isfinite(logits).all()) and float(logits.min()) >= 1 and float(logits.max()) <= want


def test_gaussian_window():
    g = gaussian_window((8, 16, 12), .125, "cpu")
    assert g.shape == (8, 16, 12) and float(g.min()) > 0 and float(g.max()) <= 1
    assert torch.equal(g, g.flip(0)) and torch.equal(g, g.flip(1)) and torch.equal(g, g.flip(2))
    centre = g[3:5, 7:9, 5:7]
    assert float(centre.min()) == float(g.max()) == pytest.approx(float(np.exp(-.5*((.5/1.)**2+(.5/2.)**2+(.5/1.5)**2))), rel=1e-6)    # half a voxel from the centre on an even side
    assert float(gaussian_window((9, 9, 9), .125, "cpu").max()) == 1.


def test_weighted_and_plain_averages_on_two_windows():
    model_plain, model_gauss = CountingModel(), CountingModel()
    image = np.zeros((8, 8, 12), dtype=np.float32)
    plain = predict_logits(model_plain, image, device="cpu", amp=False, window=(8, 8, 8))
    gauss = predict_logits(model_gauss, image, device="cpu", amp=False, window=(8, 8, 8), gaussian_sigma=.125)
    assert len(model_plain.calls) == 2                                              # x starts 0 and 4
    assert plain[0, 0, 0, 0].tolist() == pytest.approx([1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2])
    g = gaussian_window((8, 8, 8), .125, "cpu")
    for x in range(12):
        w1 = float(g[0, 0, x]) if x < 8 else 0.
        w2 = float(g[0, 0, x-4]) if x >= 4 else 0.
        assert float(gauss[0, 0, 0, 0, x]) == pytest.approx((w1+2*w2)/(w1+w2), rel=1e-5)
    assert float(gauss[0, 0, 0, 0, 5]) != pytest.approx(float(plain[0, 0, 0, 0, 5]))


def test_default_path_is_the_old_formula():
    torch.manual_seed(0)
    model = build_model(dict(base_channels=4, depth=2)).eval()
    image = np.random.default_rng(0).random((12, 16, 20)).astype(np.float32)
    got = predict_logits(model, image, device="cpu", window=(8, 8, 8))
    out, hits = torch.zeros(1, 36, 12, 16, 20), torch.zeros(1, 1, 12, 16, 20)
    x = torch.as_tensor(image)[None, None]
    with torch.no_grad():
        for a in window_starts(12, 8):
            for b in window_starts(16, 8):
                for c in window_starts(20, 8):
                    sl = (slice(None), slice(None), slice(a, a+8), slice(b, b+8), slice(c, c+8))
                    out[sl] += model(x[sl]).float()
                    hits[sl] += 1
    assert torch.equal(got, out/hits)


# ---- the old profiles do not move ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(torch.__version__ != old.TORCH_VERSION, reason="a bitwise comparison of this torch version")
def test_old_profiles_bit_by_bit(tmp_path):
    """Weights, temperatures, logits (whole volume and sliding window), probabilities of train -> calibrate -> register -> infer of v1 and v2_full on synthetic subjects, against the file written from the code of 1933643, before UL."""
    saved = json.loads(old.GOLDEN.read_text(encoding="utf-8"))
    if saved["numpy"] != np.__version__:
        pytest.skip("a bitwise comparison of this numpy version")
    assert saved["commit"] == "1933643"
    assert json.loads(json.dumps(old.fingerprint(tmp_path))) == saved["fingerprint"]


def test_train_config_of_the_old_profiles_is_v2s_record():
    want = {"base_channels": 16, "depth": 4, "n_classes": 36, "max_channels": 320, "lr": 0.0005, "weight_decay": 0.0001, "betas": [0.9, 0.999], "warmup_iters": 200, "max_iters": 20000, "poly_power": 0.9, "eval_every": 500, "patience": 10, "min_iters": 4000, "patch_size": None, "amp": True,
            "grad_clip": 12.0, "ce_weight": 1.0, "dice_weight": 1.0, "dice_smooth": 1e-05, "percentile": 99.5, "log_every": 50, "cudnn_benchmark": True,
            "augment": {"p_spatial": 0.5, "max_rotation_deg": 10.0, "max_translation_vox": 8.0, "p_gamma": 0.3, "gamma_range": [0.7, 1.5], "p_scale_shift": 0.5, "scale_range": [0.9, 1.1], "shift_range": [-0.05, 0.05]}}     # train_record.json of foldA_s0 of unet35_v2_full
    assert json.loads(json.dumps(asdict(TrainConfig()))) == want
    assert TrainConfig().model_config() == dict(base_channels=16, depth=4, n_classes=36, max_channels=320)
    assert not {"batch_size", "ds_levels", "flip_probability", "window_gaussian_sigma"} & {f.name for f in fields(TrainConfig)}


def test_old_member_carries_no_new_key(tmp_path):
    train, calib = tiny_sets()
    train_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY, patch_size=(8, 8, 8)), device="cpu", log=quiet, subjects=(train, calib))
    calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, device="cpu", log=quiet, subjects=calib)
    blob = torch.load(tmp_path/"foldA_s0"/"member.pt", map_location="cpu", weights_only=True)
    assert "window_gaussian_sigma" not in blob and blob["window"] == [8, 8, 8]


def test_cli_of_the_old_profiles():
    assert cli.PROFILES["v1"] == cli.Profile(pins.DEFAULT_FOLDS, "45,8", Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1"), cli.DEFAULT_REGISTRY, "unet35_v1_ens", "unet35_v1")
    assert cli.PROFILES["v2_full"] == cli.Profile(pins.DEFAULT_FULL_FOLDS, "98,8", Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full"), CODE_ROOT/"configs/bayesreg/segmenters_v2.json", "unet35_v2_full", "unet35_v2_full")
    for profile, full, on_full in (("v1", False, False), ("v2_full", True, True), ("v3_large", False, True)):
        args = cli.build_parser().parse_args(["check-inputs", "--profile", profile])
        assert cli._full(args) is full and cli._on_full_folds(args) is on_full


def test_cli_profile_v3_defaults():
    args = cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large"]))
    assert args.folds == pins.DEFAULT_FULL_FOLDS and args.expect_counts == "98,8" and args.out == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v3_large")
    assert cli._pinned_sha(types.SimpleNamespace(synthetic=False, profile="v3_large")) == pins.PINNED_FULL_FOLDS_SHA256
    assert cli.PROFILES["v3_large"].segmenter_id == SEGMENTER_ID_V3 and cli.PROFILES["v3_large"].prefix == MODEL_PREFIX_V3


@pytest.mark.parametrize("profile,want", [("v1", "train_member"), ("v2_full", "train_member"), ("v3_large", "train_large_member")])
def test_cmd_train_dispatch(monkeypatch, profile, want):
    called = {}
    for module, name in ((train_module, "train_member"), (large, "train_large_member")):
        monkeypatch.setattr(module, name, lambda name=name, **kwargs: (called.update(name=name, config=kwargs["config"]), dict(best_calib_mean_dice35=0., stopped="x"))[1])
    monkeypatch.setattr(cli, "_require_snapshot", lambda args: None)
    monkeypatch.setattr(cli, "_guard", lambda args: (object(), {}))
    args = cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", profile, "--set", "max_iters=3"]))
    cli.cmd_train(args)
    assert called["name"] == want and called["config"] == dict(max_iters=3)


def test_real_training_cannot_switch_the_flip_check_off(monkeypatch):
    monkeypatch.setattr(cli, "_require_snapshot", lambda args: None)
    monkeypatch.setattr(cli, "_guard", lambda args: (object(), {}))
    args = cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large", "--set", "flip_axis_check=false"]))
    with pytest.raises(SystemExit, match="synthetic subjects only"):
        cli.cmd_train(args)


def test_synthetic_training_skips_the_flip_check_and_says_so(monkeypatch):
    called = {}
    monkeypatch.setattr(large, "train_large_member", lambda **kwargs: (called.update(kwargs), dict(best_calib_mean_dice35=0., stopped="x"))[1])
    monkeypatch.setattr(cli, "_require_snapshot", lambda args: None)
    monkeypatch.setattr(cli, "_guard", lambda args: (object(), {}))
    cli.cmd_train(cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large", "--synthetic"])))
    assert called["config"] == dict(flip_axis_check=False)


@pytest.mark.parametrize("argv,match", [
    (["register", "--profile", "v3_large"], "not for --profile v3_large"),
    (["infer", "--profile", "v3_large", "--subjects", DEV[0]], "not for --profile v3_large"),
    (["evaluate-calibration", "--profile", "v3_large"], "not for --profile v3_large"),
    (["check-tissue4", "--profile", "v3_large", "--report", "/nonexistent/r.json"], "not for --profile v3_large"),
    (["--dry-run", "--profile", "v3_large"], "run_ul.py --dry-run"),
])
def test_commands_refused_for_v3(argv, match):
    with pytest.raises(SystemExit, match=match):
        cli.main(argv)


def test_snapshot_paths():
    assert SNAPSHOT_PROFILES["v1"] == SNAPSHOT_PATHS and SNAPSHOT_PROFILES["v2_full"] == SNAPSHOT_PATHS_V2 and SNAPSHOT_PROFILES["v3_large"] == SNAPSHOT_PATHS_V3
    assert set(SNAPSHOT_PATHS_V3)-set(SNAPSHOT_PATHS_V2) == {"experiments/ul", "runs/20261103-unv2/v2_member_subsets.json"}
    assert not [p for p in SNAPSHOT_PATHS_V3 if "holdout" in p]
    assert all((CODE_ROOT/p).exists() for p in SNAPSHOT_PATHS_V3)


# ---- training on the recipe -----------------------------------------------------------------------------------------------------------------------------------------

def test_training_writes_a_plain_member(tmp_path):
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE), device="cpu", log=quiet, command=["test"], subjects=(train, calib))
    out = tmp_path/"foldA_s0"
    assert record["recipe"] == "v3_large" and record["iterations_run"] == 4 and record["stopped"] == "max_iters" and record["parameters"] > record["parameters_plain"]
    blob = torch.load(out/"model_best.pt", map_location="cpu", weights_only=True)
    assert not [k for k in blob["state_dict"] if k.startswith("aux_heads")] and blob["model_config"] == dict(base_channels=4, depth=2, n_classes=36, max_channels=320)
    build_model(blob["model_config"]).load_state_dict(blob["state_dict"], strict=True)
    rows = [json.loads(line) for line in (out/"train_log.jsonl").read_text().splitlines()]
    steps = [r for r in rows if r["event"] == "train"]
    assert len(steps) == 4 and all(r["step_seconds"] > 0 and len(r["subjects"]) == 2 and len(r["level_losses"]) == 2 for r in steps)
    assert rows[0]["ds_weights"] == pytest.approx([2/3, 1/3]) and rows[0]["recipe"] == "v3_large" and len(rows[0]["left_right_pairs"]) == 16
    assert record["flip_check"]["skipped"] is True
    with pytest.raises(FileExistsError):
        train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE), device="cpu", log=quiet, subjects=(train, calib))


def test_training_is_deterministic_on_the_cpu(tmp_path):
    train, calib = tiny_sets()
    states = []
    for name in ("a", "b"):
        train_large_member(files=None, guard=None, fold="A", seed=3, models_root=tmp_path/name, config=dict(TINY_LARGE), device="cpu", log=quiet, subjects=(train, calib))
        states.append(old.sha_state(torch.load(tmp_path/name/"foldA_s3"/"model_best.pt", map_location="cpu", weights_only=True)["state_dict"]))
    assert states[0] == states[1]


def test_training_steps_lower_the_loss():
    item = tiny_subject("s", shape=(8, 8, 8), seed=3, n_classes=36)
    torch.manual_seed(0)
    model = UNet3DDeepSupervision(base_channels=4, depth=2, ds_levels=1).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    x, y = torch.as_tensor(item.image)[None, None], torch.as_tensor(item.labels.astype(np.int64))[None]
    losses = []
    for _ in range(25):
        loss, _ = deep_supervision_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]-.3


@pytest.mark.parametrize("probability,flips", [(0., 0), (1., 8)])
def test_flip_probability(tmp_path, probability, flips):
    train, calib = tiny_sets()
    train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE, flip_probability=probability), device="cpu", log=quiet, subjects=(train, calib))
    rows = [json.loads(line) for line in (tmp_path/"foldA_s0"/"train_log.jsonl").read_text().splitlines()]
    assert rows[-1]["flips"] == flips


def test_wall_budget_stops_the_loop_and_keeps_the_best(tmp_path):
    train, calib = tiny_sets()
    record = train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE, max_iters=50, min_iters=50, eval_every=25, max_wall_seconds=0.), device="cpu", log=quiet, subjects=(train, calib))
    assert record["stopped"] == "wall_budget" and record["iterations_run"] == 1 and record["best_iteration"] == 1 and (tmp_path/"foldA_s0"/"model_best.pt").is_file()


def test_flip_check_runs_by_default_and_stops_the_training(tmp_path):
    train, calib = tiny_sets()                                           # random labels: no side
    with pytest.raises(InputValidationError, match="check_flip_axis"):
        train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE, flip_axis_check=True), device="cpu", log=quiet, subjects=(train, calib))
    assert not (tmp_path/"foldA_s0"/"model_best.pt").exists()


def test_fold_and_patch_refused(tmp_path):
    train, calib = tiny_sets()
    with pytest.raises(InputValidationError, match="fold"):
        train_large_member(files=None, guard=None, fold="C", seed=0, models_root=tmp_path, config=dict(TINY_LARGE), device="cpu", subjects=(train, calib))
    with pytest.raises(InputValidationError, match="multiples"):
        train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=dict(TINY_LARGE, patch_size=[8, 8, 9]), device="cpu", subjects=(train, calib))


def test_gpu_cap_passes_an_index():
    """torch refuses a device without an index here (read from its source; the _get_device_index below does not touch a GPU): "cuda" must be resolved first."""
    from torch._utils import _get_device_index
    with pytest.raises(ValueError, match="specified index"):
        _get_device_index(torch.device("cuda"))
    calls = []
    fake = types.SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda index: types.SimpleNamespace(total_memory=24*GIB), set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)))
    assert cap_gpu_memory(torch.device("cuda"), 20., cuda=fake) == 0 and cap_gpu_memory(torch.device("cuda:1"), 30, cuda=fake) == 1
    assert calls == [(20/24, 0), (1., 1)]
    assert all(isinstance(f, float) and isinstance(i, int) and _get_device_index(i) == i for f, i in calls)


# ---- a synthetic tree: the guard, the member checks, calibration, the registered path -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    work = tmp_path_factory.mktemp("ul_tree")
    torch.set_num_threads(1)
    TREE.build(work)
    allowlist = load_dev_allowlist(work/"dev_subjects.json")
    split = load_full_folds(work/"folds_full_v2.json")
    files = OasisFiles(work/"data")
    models = work/"models"
    config = dict(TINY_LARGE, patch_size=[16, 16, 16])
    records = {}
    for fold in ("A", "B"):
        guard = LabelGuard(split, allowlist, training_source=TREE.SOURCE)
        records[fold] = train_large_member(files=files, guard=guard, fold=fold, seed=0, models_root=models, config=config, device="cpu", log=quiet, command=["test"])
        guard = LabelGuard(split, allowlist, training_source=TREE.SOURCE)
        calibrate_member(files=files, guard=guard, fold=fold, seed=0, models_root=models, device="cpu", log=quiet, command=["test"])
    control = work/"v2"
    guard = LabelGuard(split, allowlist, training_source=TREE.SOURCE)
    train_member(files=files, guard=guard, fold="A", seed=0, models_root=control, config=dict(TINY), device="cpu", log=quiet, command=["test"])
    guard = LabelGuard(split, allowlist, training_source=TREE.SOURCE)
    calibrate_member(files=files, guard=guard, fold="A", seed=0, models_root=control, device="cpu", log=quiet, command=["test"])
    reference = RUN.synthetic_reference(work/"reference.json", control=control/"foldA_s0", subjects=TREE.DEV, data_root=work/"data")
    return types.SimpleNamespace(work=work, allowlist=allowlist, split=split, files=files, models=models, records=records, control=control/"foldA_s0", reference=reference)


def eval_kwargs(tree, out_dir):
    return dict(member_root=tree.models, out_dir=out_dir, reference_path=tree.reference, control_member=tree.control, data_root=tree.work/"data", allowlist_path=tree.work/"dev_subjects.json", folds_path=tree.work/"folds.json", dev_source="synthetic_dev", synthetic=True, device="cpu", log=quiet)


def test_member_passes_the_old_checks(tree):
    members, check = verify_members(tree.models, split=tree.split, allowlist=tree.allowlist, training_source=TREE.SOURCE)
    assert check["passed"] and sorted((m["fold"], m["seed"]) for m in check["members"]) == [("A", 0), ("B", 0)]
    assert all(m["training_reads"] == 6 and m["calibration_reads"] == 1 for m in check["members"])
    record = tree.records["A"]
    assert sorted(r["subject"] for r in record["label_guard"]["label_reads"]) == sorted(TRAINING)


def test_calibration_and_inference_use_the_gaussian_window(tree, tmp_path, monkeypatch):
    blob = torch.load(tree.models/"foldA_s0"/"member.pt", map_location="cpu", weights_only=True)
    assert blob["window_gaussian_sigma"] == .125 and blob["window"] == [16, 16, 16]
    registry_path = tmp_path/"segmenters.json"
    register(tree.models, registry_path, split=tree.split, allowlist=tree.allowlist, training_source=TREE.SOURCE, segmenter_id=SEGMENTER_ID_V3, prefix=MODEL_PREFIX_V3, full_training_set=True, log=quiet)
    (tree.models/"ensemble.json").unlink()
    registry = load_segmenter_registry(registry_path) if registry_path.exists() else None
    assert sorted(registry.models) == [f"{MODEL_PREFIX_V3}_foldA", f"{MODEL_PREFIX_V3}_foldB"]
    from bayesreg.seg.unet.groups import load_grouping
    grouping = load_grouping()
    prov = {}
    for subject in DEV:
        prov[subject] = segment_subject(subject, files=tree.files, registry=registry, allowlist=tree.allowlist, models_root=tree.models, grouping=grouping, pred_root=tmp_path/"pred", device="cpu", segmenter_id=SEGMENTER_ID_V3, command=["test"], log=quiet)
    assert prov[DEV[0]]["policy"] == "all_folds_mean" and len(prov[DEV[0]]["members"]) == 2
    probs, _ = read_probabilities(tmp_path/"pred"/DEV[0], "probs36")
    # the same probabilities by hand, with the Gaussian window of the members
    from bayesreg.seg.unet.data import normalize_image, read_image
    values, _ = read_image(tree.files, DEV[0])
    image, _ = normalize_image(values)
    total = 0
    for fold in ("A", "B"):
        entry = registry.models[f"{MODEL_PREFIX_V3}_fold{fold}"]
        model, blob = load_member(tree.models/f"fold{fold}_s0"/"member.pt", entry["member_model_sha256"][0], "cpu", entry=entry)
        z = predict_logits(model, image, device="cpu", amp=False, window=(16, 16, 16), gaussian_sigma=.125)
        total = total+torch.softmax(z/blob["temperature"], dim=1)[0]
    assert np.abs(probs-(total/2).numpy()).max() < 2e-3                 # the file is float16
    uniform = 0
    for fold in ("A", "B"):
        entry = registry.models[f"{MODEL_PREFIX_V3}_fold{fold}"]
        model, blob = load_member(tree.models/f"fold{fold}_s0"/"member.pt", entry["member_model_sha256"][0], "cpu", entry=entry)
        uniform = uniform+torch.softmax(predict_logits(model, image, device="cpu", amp=False, window=(16, 16, 16))/blob["temperature"], dim=1)[0]
    assert float(np.abs((uniform/2).numpy()-(total/2).numpy()).max()) > 1e-3     # the window does change the output, so the comparison above has teeth
    pair = load_pair(write_pair(tmp_path/"pair", tree.work/"data", DEV[0], DEV[1]), allowlist=tree.allowlist)
    for side, subject in (("fixed", DEV[0]), ("moving", DEV[1])):
        validate_segmentation_provenance(json.loads((tmp_path/"pred"/subject/"provenance.json").read_text()), pair, side=side, layer="structure", registry=registry, allowlist=tree.allowlist)


def test_label_guard_on_the_v3_path(tree, tmp_path):
    """T0, development and hold-out subjects are refused by the guard the training reads through; the training subject next to them is admitted."""
    guard = LabelGuard(tree.split, tree.allowlist, training_source=TREE.SOURCE)
    assert read_labels(tree.files, TRAINING[0], guard)[0].shape == (32, 32, 32)
    for subject in (TREE.DEV[0], TREE.T0[0], oasis(910), oasis(999), "OASIS_OAS1_0012_MR1"):
        with pytest.raises(LabelAccessRefused):
            read_labels(tree.files, subject, guard)


def test_folds_with_a_t0_subject_are_refused_before_any_label(tmp_path):
    folds = parse_folds(folds_document(A_TRAIN+T0, A_CALIB, B_TRAIN, B_CALIB))
    allowlist = load_dev_allowlist(write_json(tmp_path/"a.json", allowlist_document([("synthetic_t0", T0, True), (TRAIN_SOURCE, TRAINING+T0, False)])))
    with pytest.raises(LabelAccessRefused, match="never_for_training"):
        LabelGuard(folds, allowlist, training_source=TRAIN_SOURCE)


def test_folds_with_a_hold_out_subject_are_refused():
    with pytest.raises(InputValidationError, match="also listed"):
        parse_folds(folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB, holdout=[A_TRAIN[0]]))


# ---- the evaluation --------------------------------------------------------------------------------------------------------------------------------------------------------

def test_evaluation_on_the_tree(tree, tmp_path):
    document = EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e"))
    assert document["subjects"] == TREE.DEV and document["arms"] == ["U1", "U1_uniform", "v2_A0_control"]
    assert document["means"]["v2_A0_control"]["seg35"] == pytest.approx(document["reference"]["means"]["seg35"], abs=1e-12)     # the control reproduces the reference
    assert document["means"]["v2_A0_control"]["seg4"] == pytest.approx(document["reference"]["means"]["seg4"], abs=1e-12)
    for arm in document["arms"]:
        assert set(document["means"][arm]) == {"seg35", "seg4", "small_six", "big_six", "other_23"}
        assert document["seconds"][arm]["n"] == 2 and document["seconds"][arm]["median"] > 0
    assert document["window"]["window"] == [16, 16, 16] and document["window"]["overlap"] == .5 and document["window"]["gaussian_sigma"] == .125 and document["window"]["n_windows"] == 27
    assert (tmp_path/"e"/"per_subject.csv").is_file() and (tmp_path/"e"/"per_label.csv").is_file()
    assert len(json.loads((tmp_path/"e"/"eval_u1.json").read_text())["rows"]["U1"]) == 2
    with pytest.raises(FileExistsError):
        EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e"))
    # the same function of the v2 reference as the reference file wrote, at the rows
    assert document["paired"]["v2_A0_control_minus_reference"]["seg35"]["mean"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("source,match", [("synthetic_t0", "not a development subject"), (TREE.SOURCE, "not a development subject"), ("no_such_source", "no source")])
def test_evaluation_refuses_t0_and_training_subjects_before_opening_a_file(tree, tmp_path, monkeypatch, source, match):
    opened = []
    monkeypatch.setattr(EVAL, "read_image", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    with pytest.raises(SubjectSealed, match=match):
        EVAL.evaluate(**dict(eval_kwargs(tree, tmp_path/"e"), dev_source=source))
    assert not opened and not (tmp_path/"e").exists()


def test_evaluation_refuses_a_subject_in_the_label_contact(tree, tmp_path, monkeypatch):
    """The counterexample of the contact check: a member whose contact holds the development subject."""
    monkeypatch.setattr(EVAL, "read_image", lambda *a, **k: pytest.fail("read an image"))
    original = EVAL.member_files

    def with_contact(directory):
        path, sha, record = original(directory)
        return path, sha, dict(record, label_contact_subjects=list(record["label_contact_subjects"])+[TREE.DEV[0]])
    monkeypatch.setattr(EVAL, "member_files", with_contact)
    with pytest.raises(SubjectSealed, match="label contact"):
        EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e"))


def test_real_lists_admit_the_20_and_refuse_t0_and_the_hold_outs(tmp_path):
    """Ids only."""
    allowlist = load_dev_allowlist(SEG.ALLOWLIST)
    sealed = load_sealed_lists(CODE_ROOT)
    training = load_folds(SEG.FOLDS).training_subjects
    admitted = EVAL.admit(allowlist, SEG.DEV_SOURCE, sealed, training, dict(U1=sorted(training)))
    assert len(admitted) == 20 and admitted == [f"OASIS_OAS1_{n:04d}_MR1" for n in range(438, 458)]
    with pytest.raises(SubjectSealed, match="not a development subject"):
        EVAL.admit(allowlist, "stage_T0_tune_v1", sealed, training, {})
    for name in ("holdout_B_v1", "H1", "H4"):
        subject = sorted(sealed.subjects[name])[0]
        doc = allowlist_document([("dev", [admitted[0], subject], False)])
        bad = load_dev_allowlist(write_json(tmp_path/f"{name}.json", doc))
        with pytest.raises(SubjectSealed, match=f"sealed list {name}"):
            EVAL.admit(bad, "dev", sealed, training, {})
    bad = load_dev_allowlist(write_json(tmp_path/"train.json", allowlist_document([("dev", [admitted[0], sorted(training)[0]], False)])))
    with pytest.raises(SubjectSealed, match="not a development subject"):
        EVAL.admit(bad, "dev", sealed, training, {})


def test_real_training_folds_hold_no_t0_or_hold_out_subject():
    """Ids only: what the guard will admit for U1."""
    split = load_full_folds(pins.DEFAULT_FULL_FOLDS, expect_counts=pins.PINNED_FULL_COUNTS)
    allowlist = load_dev_allowlist(SEG.ALLOWLIST)
    sealed = load_sealed_lists(CODE_ROOT)
    contact = split.label_contact("A")
    assert len(contact) == 106 and not set(contact) & allowlist.never_for_training and all(sealed.which(s) is None for s in contact)
    assert all(not 438 <= int(s.split("_")[2]) <= 457 for s in contact)


def test_real_evaluation_refuses_other_inputs(tmp_path, lock):
    with pytest.raises(ValueError, match="the plan's"):
        EVAL.evaluate(member_root=tmp_path, out_dir=tmp_path/"e", dev_source="stage_T0_tune_v1", device="cpu")
    with pytest.raises(ValueError, match="the plan's"):
        EVAL.evaluate(member_root=tmp_path, out_dir=tmp_path/"e", reference_path=tmp_path/"other.json", device="cpu")


def test_synthetic_evaluation_refuses_real_ids_and_real_data_root(tree, tmp_path):
    with pytest.raises(pins.PinnedInputRefused):
        EVAL.evaluate(**dict(eval_kwargs(tree, tmp_path/"e"), data_root=Path("/mnt/d/data")))
    with pytest.raises(pins.PinnedInputRefused):
        EVAL.evaluate(**dict(eval_kwargs(tree, tmp_path/"e2"), allowlist_path=SEG.ALLOWLIST))


def test_reference_of_the_repository():
    """What the plan cites: A0 is fold A, seed 0, and is not the .9230 / .9566 of the brief, which is the mean over the four single members."""
    reference = EVAL.read_reference(EVAL.REFERENCE, pinned_sha256=EVAL.REFERENCE_SHA256)
    assert reference["member"] == "A0" and reference["members"] == ["A0", "A1", "B0", "B1"] and len(reference["rows"]) == 20
    assert round(reference["means"]["seg35"], 4) == .9227 and round(reference["means"]["seg4"], 4) == .9567
    assert round(reference["single_member_mean"]["seg35"], 4) == .9230 and round(reference["single_member_mean"]["seg4"], 4) == .9566
    assert round(reference["ensemble"]["seg35"], 4) == .9275 and round(reference["ensemble"]["seg4"], 4) == .9591
    assert reference["subjects"] == [f"OASIS_OAS1_{n:04d}_MR1" for n in range(438, 458)]
    with pytest.raises(ValueError, match="pinned"):
        EVAL.read_reference(EVAL.REFERENCE, pinned_sha256="0"*64)
    registered = json.loads((CODE_ROOT/"configs/bayesreg/segmenters_v2.json").read_text())["models"]["unet35_v2_full_foldA"]
    assert EVAL.CONTROL_SHA256 == registered["member_model_sha256"][0] and registered["fold"] == 0


def test_reference_that_does_not_add_up_is_refused(tmp_path):
    document = json.loads(EVAL.REFERENCE.read_text())
    document["summary"]["A0"]["seg35"] += .001
    path = tmp_path/"r.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not follow"):
        EVAL.read_reference(path)


# ---- the reading -----------------------------------------------------------------------------------------------------------------------------------------------------------

def made_evaluation(delta, *, control=0., n=20, fold="A", recipe_member="A0", seg4_delta=.001):
    ref = dict(seg35=.92266, seg4=.95667, small_six=.807, big_six=.954, other_23=.944)
    means = dict(U1={k: v+(delta if k == "seg35" else seg4_delta) for k, v in ref.items()}, U1_uniform=dict(ref), v2_A0_control=dict(ref, seg35=ref["seg35"]+control, seg4=ref["seg4"]+control))
    pair = dict(mean=delta, median=delta, n_positive=15, n_negative=5, n_zero=0, n=20, n_undefined=0)
    return dict(subjects=[f"s{i}" for i in range(n)], means=means, paired=dict(U1_minus_reference={k: pair for k in ref}), reference=dict(member=recipe_member, means=ref, single_member_mean=dict(seg35=.92304, seg4=.95663), ensemble=dict(seg35=.92754, seg4=.95908)),
                members=dict(U1=dict(fold=fold, seed=0), v2_A0_control=dict(fold=fold, seed=0)), seconds={a: dict(mean=1., median=1., minimum=.5, maximum=2., n=n) for a in means}, window=dict(window=list(PATCH_U1), overlap=.5, n_windows=8), gpu_peak_allocated_gib=5.)


MADE_RECORD = dict(recipe="v3_large", fold="A", seed=0, stopped="max_iters", iterations_run=30000, best_iteration=30000, best_calib_mean_dice35=.93, wall_seconds=4*3600., memory=dict(gpu_peak_allocated_gib=14., host_peak_rss_gib=7.))


@pytest.mark.parametrize("delta,reading", [(.0051, DECIDE.WORTH), (.005, DECIDE.WORTH), (.00499, DECIDE.MIDDLE), (.0035, DECIDE.MIDDLE), (.00201, DECIDE.MIDDLE), (.002, DECIDE.NOT_LIMIT), (.0019, DECIDE.NOT_LIMIT), (0., DECIDE.NOT_LIMIT), (-.01, DECIDE.NOT_LIMIT),
                                           (float("nan"), DECIDE.NOT_DECIDABLE), (None, DECIDE.NOT_DECIDABLE), (float("inf"), DECIDE.NOT_DECIDABLE)])
def test_reading_of(delta, reading):
    assert DECIDE.reading_of(delta) == reading


@pytest.mark.parametrize("delta,reading,text", [(.006, DECIDE.WORTH, "值得训完集成"), (.0035, DECIDE.MIDDLE, "中间带"), (.001, DECIDE.NOT_LIMIT, "配方不是主要限制，交 BZ 定 nnU-Net")])
def test_decide_each_branch(delta, reading, text):
    document = DECIDE.decide(made_evaluation(delta), MADE_RECORD)
    assert document["reading"] == reading and document["reading_text"] == text and document["delta_seg35"] == pytest.approx(delta, abs=1e-12) and document["not_decidable_because"] == []
    summary = DECIDE.summary_markdown(document)
    assert text in summary and "seg35" in summary and "seg4" in summary and "小六个" in summary and "大六个" in summary and "其余 23 个" in summary and "秒" in summary
    assert document["budgets"]["gpu_within_limit"] is True and document["budgets"]["host_within_limit"] is True and document["budgets"]["training_within_limit"] is True


@pytest.mark.parametrize("evaluation,record,reasons,match", [
    (None, MADE_RECORD, (), "没有评测结果"),
    (made_evaluation(.006), None, (), "train_record"),
    (made_evaluation(.006), dict(MADE_RECORD, recipe=None), (), "recipe"),
    (made_evaluation(.006), dict(MADE_RECORD, fold="B"), (), "不是 A0"),
    (made_evaluation(.006, fold="B"), MADE_RECORD, (), "不是 fold A"),
    (made_evaluation(.006, recipe_member="B0"), MADE_RECORD, (), "参照不是成员 A0"),
    (made_evaluation(.006, n=19), MADE_RECORD, (), "19 名受试者"),
    (made_evaluation(.006, control=.0003), MADE_RECORD, (), "控制不符"),
    (made_evaluation(.006), MADE_RECORD, ("训练失败",), "训练失败"),
])
def test_decide_not_decidable(evaluation, record, reasons, match):
    document = DECIDE.decide(evaluation, record, reasons=reasons)
    assert document["reading"] == DECIDE.NOT_DECIDABLE and any(match in r for r in document["not_decidable_because"]) and document["delta_seg35"] is None
    assert "不可判" in DECIDE.summary_markdown(document)


def test_control_inside_the_tolerance_decides():
    assert DECIDE.decide(made_evaluation(.006, control=.00019), MADE_RECORD)["reading"] == DECIDE.WORTH


def test_nan_mean_is_not_decidable():
    evaluation = made_evaluation(.006)
    evaluation["means"]["U1"]["seg35"] = float("nan")
    assert DECIDE.decide(evaluation, MADE_RECORD)["reading"] == DECIDE.NOT_DECIDABLE
    evaluation = made_evaluation(.006)
    evaluation["means"]["U1"]["seg4"] = None
    assert DECIDE.decide(evaluation, MADE_RECORD)["reading"] == DECIDE.NOT_DECIDABLE


def test_undefined_group_is_described_and_still_decides(tmp_path):
    evaluation = made_evaluation(.006)
    evaluation["means"]["U1"]["small_six"] = None
    document = DECIDE.decide(evaluation, MADE_RECORD)
    assert document["reading"] == DECIDE.WORTH and document["comparisons"]["same_member_A0"]["small_six"] is None
    DECIDE.write(tmp_path, dict(document, extra=float("nan")))
    assert json.loads((tmp_path/"decision_ul.json").read_text())["extra"] is None and "n/a" in (tmp_path/"summary_ul.md").read_text(encoding="utf-8")


def test_the_reference_of_the_reading_is_the_same_member_not_the_mean():
    """A delta the two references read differently: the same member decides, the other is written down."""
    document = DECIDE.decide(made_evaluation(.0052), MADE_RECORD)
    four = document["against_four_single_members"]
    assert document["reading"] == DECIDE.WORTH and four["reading_it_would_give"] == DECIDE.MIDDLE and four["reading_differs"] is True
    assert "与上面的读数不同" in DECIDE.summary_markdown(document)
    assert DECIDE.decide(made_evaluation(.007), MADE_RECORD)["against_four_single_members"]["reading_differs"] is False


def test_budgets_are_reported_and_do_not_change_the_reading():
    record = dict(MADE_RECORD, wall_seconds=6*3600., memory=dict(gpu_peak_allocated_gib=21., host_peak_rss_gib=9.))
    document = DECIDE.decide(made_evaluation(.006), record)
    assert document["reading"] == DECIDE.WORTH and document["budgets"]["gpu_within_limit"] is False and document["budgets"]["host_within_limit"] is False and document["budgets"]["training_within_limit"] is False


def test_decision_files_never_overwritten_and_missing_inputs_read_as_not_decidable(tmp_path):
    out = tmp_path/"analysis"
    assert DECIDE.main(["--eval", str(tmp_path/"none.json"), "--member", str(tmp_path/"none"), "--out-dir", str(out), "--reason", "没有跑完"]) == 0
    document = json.loads((out/"decision_ul.json").read_text())
    assert document["reading"] == DECIDE.NOT_DECIDABLE and "没有跑完" in document["not_decidable_because"] and (out/"summary_ul.md").is_file()
    with pytest.raises(FileExistsError):
        DECIDE.main(["--eval", str(tmp_path/"none.json"), "--member", str(tmp_path/"none"), "--out-dir", str(out)])


# ---- the start strings, the gate ----------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("step", ["smoke", "train"])
def test_launch_strings(step):
    argv = RUN.launch_argv(step)
    text = " ".join(argv)
    assert "pytest" not in text and f"--unit=tag-ul-{step}" in argv and "seq-vp" not in text and "--collect" in argv
    assert f"--working-directory={RUN.OUT}/snapshot" in argv and "--property=MemoryMax=8G" in argv and str(RUN.RESULTS) in text and argv[-5:-3] == [step, "--out"]
    assert f"--property=RuntimeMaxSec={dict(smoke=3600, train=20700)[step]}" in argv
    assert argv[:2] == ["systemd-run", "--user"] and f"{RUN.OUT}/snapshot/experiments/ul/run_ul.py" in argv


def test_launch_patch_only_for_the_smoke():
    assert RUN.launch_argv("smoke", patch=(112, 144, 96))[-2:] == ["--patch", "112,144,96"]
    assert "--patch" not in RUN.launch_argv("train", patch=(112, 144, 96))


@pytest.mark.parametrize("change,match", [
    (lambda a: a+["pytest"], "test runner"),
    (lambda a: [x.replace("--unit=tag-ul-smoke", "--unit=seq-vp-x") for x in a], "tag-ul"),
    (lambda a: [x for x in a if not x.startswith("--unit=")], "tag-ul"),
    (lambda a: a+["seq-vp-other"], "seq-vp"),
    (lambda a: [x.replace(f"--working-directory={RUN.OUT}/snapshot", "--working-directory=/tmp") for x in a], "working directory"),
    (lambda a: [x for x in a if x != "--property=MemoryMax=8G"], "MemoryMax"),
    (lambda a: [x for x in a if not x.startswith("--property=RuntimeMaxSec=")], "RuntimeMaxSec"),
])
def test_check_launch_refuses(change, match):
    with pytest.raises(ValueError, match=match):
        RUN.check_launch(change(RUN.launch_argv("smoke")))


def test_launch_step_unknown():
    with pytest.raises(ValueError):
        RUN.launch_argv("other")


def fake_run(units="", free=23000, util=0, pids="", gpu_rc=0, units_rc=0):
    def run(argv, **kwargs):
        if argv[0] == "systemctl":
            return types.SimpleNamespace(returncode=units_rc, stdout=units, stderr="failed")
        if "--query-compute-apps=pid" in argv:
            return types.SimpleNamespace(returncode=0, stdout=pids, stderr="")
        return types.SimpleNamespace(returncode=gpu_rc, stdout=f"{free}, {util}\n", stderr="")
    return run


def gate(run=None, load=.2, home=500e9, c_drive=40*2**30, available=12.):
    def disk(path):
        return types.SimpleNamespace(free=home if path == "/home" else c_drive)
    return RUN.gate_problems("tag-ul-smoke", run=run or fake_run(), load=lambda: (load, 0, 0), disk=disk, available=lambda: available)


def test_gate_open():
    assert gate() == []
    assert gate(fake_run(units="tag-ul-smoke.service loaded active running x\n")) == []


@pytest.mark.parametrize("kwargs,match", [
    (dict(run=fake_run(units="tag-tc-queue.service loaded active running x\n")), "units active"),
    (dict(run=fake_run(units="seq-vp-run.service loaded active running x\n")), "units active"),
    (dict(run=fake_run(units="tag-ul-train.service loaded active running x\n")), "units active"),
    (dict(run=fake_run(units_rc=1)), "systemctl failed"),
    (dict(load=1.5), "load"),
    (dict(run=fake_run(free=21000)), "GPU free"),
    (dict(run=fake_run(util=21)), "utilization"),
    (dict(run=fake_run(pids="123\n")), "compute processes"),
    (dict(run=fake_run(gpu_rc=1)), "no reading"),
    (dict(home=99e9), "/home"),
    (dict(c_drive=14*2**30), "/mnt/c"),
    (dict(available=8.9), "MemAvailable"),
])
def test_gate_closed(kwargs, match):
    problems = gate(**kwargs)
    assert len(problems) >= 1 and any(match in p for p in problems)


def test_gate_failed_queries_count_as_closed():
    def broken(argv, **kwargs):
        raise OSError("no such file")
    problems = gate(broken)
    assert any("systemctl failed" in p for p in problems) and any("nvidia-smi failed" in p for p in problems)


def test_wait_for_gate():
    answers = [["x"], ["y"], []]
    naps = []
    RUN.wait_for_gate("u", 5, problems=lambda unit: answers.pop(0), sleep=naps.append, poll=1)
    assert len(naps) == 2 and not answers
    with pytest.raises(RUN.Stop, match="stayed closed"):
        RUN.wait_for_gate("u", 0, problems=lambda unit: ["busy"], sleep=naps.append, poll=1)


# ---- the steps and the smoke's rule ----------------------------------------------------------------------------------------------------------------------------------------------

def test_step_runs_once_and_a_started_step_is_not_rerun(tmp_path):
    calls = []

    def ok(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(returncode=0)

    RUN.run_step("a", ["x"], state=tmp_path/"s", log_path=tmp_path/"l", cwd=tmp_path, run=ok)
    RUN.run_step("a", ["x"], state=tmp_path/"s", log_path=tmp_path/"l", cwd=tmp_path, run=ok)
    assert len(calls) == 1
    with pytest.raises(RUN.Stop, match=r"failed \(exit 3\)"):
        RUN.run_step("b", ["y"], state=tmp_path/"s", log_path=tmp_path/"l", cwd=tmp_path, run=lambda argv, **k: types.SimpleNamespace(returncode=3))
    with pytest.raises(RUN.Stop, match="did not finish"):
        RUN.run_step("b", ["y"], state=tmp_path/"s", log_path=tmp_path/"l", cwd=tmp_path, run=ok)
    assert len(calls) == 1 and (tmp_path/"s"/"b.started").is_file() and not (tmp_path/"s"/"b.done").exists()


def test_chain_order_and_contents():
    spec = RUN.Spec(out=RUN.OUT, cwd=RUN.OUT/"snapshot")
    steps = RUN.chain_steps(spec, RUN.train_config(dict(patch_size=list(PATCH_U1), max_iters=30000, eval_every=500, max_wall_seconds=16200.)))
    assert [name for name, _ in steps] == ["train_A0", "calibrate_A0", "check_members", "eval_u1", "decide_ul"]
    train = " ".join(steps[0][1])
    assert "--profile v3_large --fold A --seed 0" in train and "--set max_iters=30000" in train and "max_wall_seconds=16200.0" in train and "--device cuda" in train and "--synthetic" not in train
    assert "pytest" not in " ".join(" ".join(a) for _, a in steps)
    assert all("--profile v3_large" in " ".join(a) for _, a in steps[:3])


MEASURED = dict(step_seconds_median=.5, eval_seconds_median=5., data_ready_seconds=100., gpu_peak_allocated_gib=14., gpu_peak_reserved_gib=15., host_peak_rss_gib=7., patch_size=list(PATCH_U1))


def test_steps_rule():
    document = RUN.steps_rule(MEASURED)
    want = int(.9*(5*3600-1800-100)/(.5+5/500))
    assert document["ok"] and document["max_iters"] == want-want % 500 and document["max_iters"] % 500 == 0 and document["max_wall_seconds"] == 16200 and document["reasons"] == []
    assert RUN.train_config(document) == dict(patch_size=list(PATCH_U1), max_iters=document["max_iters"], eval_every=500, max_wall_seconds=16200., min_iters=10000)


@pytest.mark.parametrize("change,match", [(dict(gpu_peak_allocated_gib=20.5), "GPU peak"), (dict(gpu_peak_allocated_gib=None), "GPU peak"), (dict(gpu_peak_reserved_gib=19.1), "reserved peak"), (dict(gpu_peak_reserved_gib=None), "reserved peak"),
                                          (dict(host_peak_rss_gib=8.5), "host peak"), (dict(step_seconds_median=2.), "fewer than")])
def test_steps_rule_refuses(change, match):
    document = RUN.steps_rule(dict(MEASURED, **change))
    assert not document["ok"] and any(match in r for r in document["reasons"])


def test_steps_rule_bars_are_included():
    assert RUN.steps_rule(dict(MEASURED, gpu_peak_allocated_gib=19., gpu_peak_reserved_gib=19., host_peak_rss_gib=8.))["ok"]


def write_smoke(out, patch=PATCH_U1):
    directory = RUN.smoke_root(out, patch)/"foldA_s0"
    directory.mkdir(parents=True)
    rows = [dict(event="start", data_ready_seconds=70.)]+[dict(event="train", it=i, step_seconds=(9. if i < 10 else .5)) for i in range(60)]+[dict(event="eval", it=30, eval_seconds=5.), dict(event="eval", it=60, eval_seconds=6.), dict(event="end")]
    (directory/"train_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (directory/"train_record.json").write_text(json.dumps(dict(memory=dict(gpu_peak_allocated_gib=15., gpu_peak_reserved_gib=16., host_peak_rss_gib=7.), config=dict(patch_size=list(patch), batch_size=2), parameters=1, parameters_plain=1, environment={})))
    return directory


def test_read_smoke_skips_the_first_steps(tmp_path):
    measured = RUN.read_smoke(write_smoke(tmp_path))
    assert measured["step_seconds_median"] == .5 and measured["n_steps_timed"] == 50 and measured["eval_seconds_median"] == 5.5 and measured["data_ready_seconds"] == 70. and measured["gpu_peak_allocated_gib"] == 15.


def test_read_smoke_refuses_an_unfinished_run(tmp_path):
    with pytest.raises(RUN.Stop, match="no such file"):
        RUN.read_smoke(tmp_path/"none")
    directory = write_smoke(tmp_path)
    rows = (directory/"train_log.jsonl").read_text().splitlines()
    (directory/"train_log.jsonl").write_text("\n".join(rows[:-1])+"\n")
    with pytest.raises(RUN.Stop, match="did not finish"):
        RUN.read_smoke(directory)


def test_decide_steps_writes_once(tmp_path):
    write_smoke(tmp_path)
    document = RUN.decide_steps(tmp_path, PATCH_U1)
    assert document["ok"] and json.loads((tmp_path/"steps_decision.json").read_text())["max_iters"] == document["max_iters"]
    assert RUN.read_decision(tmp_path)["ok"]
    with pytest.raises(RUN.Stop, match="never overwritten"):
        RUN.decide_steps(tmp_path, PATCH_U1)


def test_decide_steps_failed_rule_leaves_no_decision_for_the_training(tmp_path):
    directory = write_smoke(tmp_path)
    record = json.loads((directory/"train_record.json").read_text())
    record["memory"]["gpu_peak_allocated_gib"] = 21.
    (directory/"train_record.json").write_text(json.dumps(record))
    assert not RUN.decide_steps(tmp_path, PATCH_U1)["ok"]
    assert (tmp_path/f"steps_decision_{RUN.patch_name(PATCH_U1)}.json").is_file() and not (tmp_path/"steps_decision.json").exists()
    with pytest.raises(RUN.Stop, match="comes? first"):
        RUN.read_decision(tmp_path)


def test_decide_steps_other_patch_refused(tmp_path):
    directory = RUN.smoke_root(tmp_path, (112, 144, 96))/"foldA_s0"
    directory.parent.mkdir(parents=True)
    write_smoke(tmp_path, patch=PATCH_U1).rename(directory)
    with pytest.raises(RUN.Stop, match="used patch"):
        RUN.decide_steps(tmp_path, (112, 144, 96))


def test_read_decision_refuses_a_failed_rule(tmp_path):
    (tmp_path/"steps_decision.json").write_text(json.dumps(dict(ok=False, reasons=["x"])))
    with pytest.raises(RUN.Stop, match="did not pass"):
        RUN.read_decision(tmp_path)


def test_estimate_counts_v2s_parameters():
    """The counting model of estimate.py agrees with what v2's record says (5648452 parameters)."""
    row = EST.count(EST.network(16, 0), (1, 1)+EST.VOLUME)
    assert row["parameters"] == 5648452 and row["head_elements"] == 36*192*224*160


# ---- the sandbox: real steps stop here; the dry run; the stop of the real entry -------------------------------------------------------------------------------------------------------

@pytest.fixture
def blocked_env(tmp_path):
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    shim = bin_dir/"systemd-run"
    marker = tmp_path/"systemd_run_called"
    shim.write_text(f"#!/bin/sh\necho called >> {marker}\nexit 1\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"), marker


@pytest.mark.parametrize("step", ["smoke", "train"])
def test_real_steps_stop_outside_a_snapshot(tmp_path, blocked_env, step):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(UL/"run_ul.py"), step, "--out", str(tmp_path/"out")], env=env, capture_output=True, text=True, timeout=120, cwd=str(CODE_ROOT))
    assert result.returncode == 8 and "STOP" in result.stdout and "snapshot" in result.stdout
    assert not marker.exists() and not (tmp_path/"out").exists()


def test_print_launch_starts_nothing(tmp_path, blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(UL/"run_ul.py"), "print-launch", "smoke"], env=env, capture_output=True, text=True, timeout=120, cwd=str(CODE_ROOT))
    assert result.returncode == 0 and result.stdout.startswith("systemd-run --user --unit=tag-ul-smoke") and "pytest" not in result.stdout
    assert not marker.exists()


def test_cli_train_real_refused_outside_a_snapshot(tmp_path, blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", "-m", "bayesreg.seg.unet.cli", "train", "--profile", "v3_large", "--fold", "A", "--seed", "0", "--out", str(tmp_path/"out"), "--device", "cpu", "--data-root", str(tmp_path/"no_data")], env=env, capture_output=True, text=True, timeout=120, cwd=str(CODE_ROOT))
    assert result.returncode != 0 and "snapshot" in (result.stdout+result.stderr)
    assert not (tmp_path/"out").exists() and not marker.exists()


def test_dry_run_end_to_end(tmp_path, blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(UL/"run_ul.py"), "--dry-run", "--scratch", str(tmp_path/"work")], env=env, capture_output=True, text=True, timeout=300, cwd=str(CODE_ROOT))
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    assert "dry run ok" in result.stdout and "pytest" not in result.stdout.replace(str(tmp_path), "")
    assert "--unit=tag-ul-smoke" in result.stdout and "--unit=tag-ul-train" in result.stdout
    assert not marker.exists()
    decision = json.loads((tmp_path/"work"/"out"/"analysis"/"decision_ul.json").read_text())
    assert decision["reading"] == "not_decidable" and decision["not_decidable_because"] == ["2 名受试者，不是 20"]
    assert json.loads((tmp_path/"work"/"out"/"control_check"/"eval_control.json").read_text())["passed"] is True
    assert "start train_A0:" in (tmp_path/"work"/"out"/"chain.log").read_text() and "start train_A0:" in (tmp_path/"work"/"out"/"run_ul.log").read_text()
    assert not dev_lock.DEV_READ_LOCK.exists() or "EXISTS" in result.stdout


# ---- the fix round: flip_axis_check cannot be set on real data (review, minor 1) ------------------------------------------------------------------------------------------------

def stub_train_entry(monkeypatch):
    called = []
    monkeypatch.setattr(large, "train_large_member", lambda **kwargs: (called.append(kwargs), dict(best_calib_mean_dice35=0., stopped="x"))[1])
    monkeypatch.setattr(cli, "_require_snapshot", lambda args: None)
    monkeypatch.setattr(cli, "_guard", lambda args: (object(), {}))
    return called


@pytest.mark.parametrize("value", ["false", "0", "null", "true", "1", '""'])
def test_real_training_cannot_set_the_flip_check_whatever_the_value(monkeypatch, value):
    called = stub_train_entry(monkeypatch)
    args = cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large", "--set", f"flip_axis_check={value}"]))
    with pytest.raises(SystemExit, match="cannot be set on real data"):
        cli.cmd_train(args)
    assert not called


def test_real_training_cannot_set_the_flip_check_in_a_config_file(monkeypatch, tmp_path):
    called = stub_train_entry(monkeypatch)
    path = tmp_path/"c.json"
    path.write_text(json.dumps(dict(flip_axis_check=None)))
    args = cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large", "--config", str(path)]))
    with pytest.raises(SystemExit, match="cannot be set on real data"):
        cli.cmd_train(args)
    assert not called


def test_real_training_without_the_key_runs_with_the_check_on(monkeypatch):
    called = stub_train_entry(monkeypatch)
    cli.cmd_train(cli._apply_profile(cli.build_parser().parse_args(["train", "--fold", "A", "--seed", "0", "--profile", "v3_large"])))
    assert "flip_axis_check" not in called[0]["config"] and TrainConfigLarge(**called[0]["config"]).flip_axis_check is True


@pytest.mark.parametrize("value,refusal", [(False, pins.PinnedInputRefused), (None, InputValidationError), (0, InputValidationError), ("", InputValidationError), ("false", InputValidationError)])
def test_the_library_also_refuses_it_on_a_real_data_root(tmp_path, value, refusal):
    """Past the command line: refused before the member directory or a label (guard None would fail at its first use)."""
    with pytest.raises(refusal):
        train_large_member(files=OasisFiles(Path("/mnt/d/data")), guard=None, fold="A", seed=0, models_root=tmp_path/"m", config=dict(TINY_LARGE, flip_axis_check=value), device="cpu", log=quiet)
    assert not (tmp_path/"m").exists()


# ---- the fix round: the training loop keeps image and labels together (review, minor 11) -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("probability", [0., 1.])
def test_training_loop_flips_and_crops_image_and_labels_together(tmp_path, monkeypatch, probability):
    """The image encodes the label of each voxel, spatial and intensity augmentation off; what reaches the network must still decode to its labels (through the swap when flipped)."""
    lut = lut_tensor()
    rng = np.random.default_rng(5)

    def subject(name):
        labels = rng.integers(0, 36, (16, 16, 16)).astype(np.uint8)
        return Subject(name, (labels/36).astype(np.float32), labels, image_sha256="0"*64, label_sha256="1"*64, normalization=dict(percentile=99.5, scale=1.))

    seen = []
    original = large.class_balanced_patch

    def spy(x, y, patch, sampler, **kwargs):
        out = original(x, y, patch, sampler, **kwargs)
        seen.append(((out[0][0, 0]*36).round().long().clone(), out[1][0].clone()))
        return out

    monkeypatch.setattr(large, "class_balanced_patch", spy)
    config = dict(TINY_LARGE, augment=dict(p_spatial=0., p_gamma=0., p_scale_shift=0.), flip_probability=probability, max_iters=6, min_iters=6, eval_every=6)
    train_large_member(files=None, guard=None, fold="A", seed=0, models_root=tmp_path, config=config, device="cpu", log=quiet, subjects=([subject(A_TRAIN[0]), subject(A_TRAIN[1])], [subject(A_CALIB[0])]))
    assert len(seen) == 12
    for decoded, labels in seen:
        assert decoded.shape == (8, 8, 8) and torch.equal(labels, lut[decoded] if probability else decoded)
    if probability:
        assert any(not torch.equal(lut[decoded], decoded) for decoded, _ in seen)        # the swap did something, so the check above has teeth


# ---- the fix round: the read lock of the development truth (review, important 3) ------------------------------------------------------------------------------------------------------

@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path/"locks"/"ul_dev_read.lock"
    monkeypatch.setattr(dev_lock, "DEV_READ_LOCK", path)
    return path


@pytest.fixture
def real_like(tree, lock, monkeypatch):
    """The real path of evaluate on the synthetic tree: only the pins point at it. No real subject, no file of a real subject."""
    monkeypatch.setattr(EVAL, "REFERENCE", Path(tree.reference))
    monkeypatch.setattr(EVAL, "REFERENCE_SHA256", hashlib.sha256(Path(tree.reference).read_bytes()).hexdigest())
    monkeypatch.setattr(EVAL, "CONTROL_SHA256", EVAL.member_files(tree.control)[1])
    monkeypatch.setattr(SEG, "ALLOWLIST", tree.work/"dev_subjects.json")
    monkeypatch.setattr(SEG, "DEV_SOURCE", "synthetic_dev")
    monkeypatch.setattr(EVAL, "load_pinned_allowlist", lambda path: (load_dev_allowlist(path), dict(allowlist_path=str(path))))
    return lock


def real_kwargs(tree, out_dir, **changes):
    return dict(eval_kwargs(tree, out_dir), synthetic=False, **changes)


def no_subject_file(monkeypatch):
    opened = []
    monkeypatch.setattr(EVAL, "read_image", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    return opened


def test_lock_path_is_fixed_and_outside_every_results_root():
    path = dev_lock.DEV_READ_LOCK
    assert path == Path("/home/rmcfz/seq_vp_tag_results/seg/ul_dev_read.lock")
    assert RUN.OUT not in path.parents and EVAL.CONTROL_MEMBER not in path.parents and path.parent == RUN.OUT.parent
    assert "--lock" not in (UL/"eval_u1.py").read_text(encoding="utf-8") and "--lock" not in (UL/"run_ul.py").read_text(encoding="utf-8")


def test_lock_functions(lock):
    dev_lock.check()                                                        # none yet
    with pytest.raises(ValueError, match="nothing to unlock"):
        dev_lock.check("BZ, some day")
    first = dev_lock.take(dict(out_dir="a"))
    assert first["event"] == "first_read" and lock.is_file()
    with pytest.raises(dev_lock.DevReadLocked, match="read once"):
        dev_lock.check()
    with pytest.raises(dev_lock.DevReadLocked, match="not read"):           # O_EXCL: two runs that both passed the check
        dev_lock.take(dict(out_dir="b"))
    assert [r["out_dir"] for r in dev_lock.describe()] == ["a"]
    for blank in ("", "  "):
        with pytest.raises(ValueError, match="words"):
            dev_lock.check(blank)
    dev_lock.check("BZ, 11-05 9 am: read it again")
    second = dev_lock.take(dict(out_dir="c"), bz_go="BZ, 11-05 9 am: read it again")
    assert second["event"] == "bz_approved_reread" and [r["out_dir"] for r in dev_lock.describe()] == ["a", "c"] and dev_lock.describe()[1]["bz_go"].startswith("BZ")


def test_real_path_locks_before_the_first_truth_and_refuses_any_new_directory(tree, real_like, tmp_path, monkeypatch):
    original = SEG.read_truth
    states = []
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: (states.append(real_like.exists()), original(*a, **k))[1])
    document = EVAL.evaluate(**real_kwargs(tree, tmp_path/"e1"))
    assert states and all(states)
    lines = dev_lock.describe()
    assert len(lines) == 1 and lines[0]["event"] == "first_read" and lines[0]["out_dir"] == str(tmp_path/"e1") and lines[0]["member_sha256"] == document["members"]["U1"]["member_sha256"] and document["dev_read"] == lines[0]
    before = real_like.read_bytes()
    opened = no_subject_file(monkeypatch)
    with pytest.raises(dev_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**real_kwargs(tree, tmp_path/"e2"))                   # another --out-dir
    with pytest.raises(dev_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**dict(real_kwargs(tree, tmp_path/"e3"), member_root=tmp_path/"another_root"))     # another --out: the check comes before anything of it is looked at
    assert not opened and not (tmp_path/"e2").exists() and not (tmp_path/"e3").exists() and real_like.read_bytes() == before


def test_failure_before_the_first_truth_leaves_no_lock(tree, real_like, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the forward pass failed")

    with monkeypatch.context() as patch:
        patch.setattr(EVAL, "timed_forward", broken)
        with pytest.raises(RuntimeError, match="forward pass"):
            EVAL.evaluate(**real_kwargs(tree, tmp_path/"e1"))
    assert not real_like.exists()
    assert EVAL.evaluate(**real_kwargs(tree, tmp_path/"e2"))["dev_read"]["event"] == "first_read"


def test_failure_after_the_first_truth_keeps_the_lock(tree, real_like, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("the truth file failed")

    with monkeypatch.context() as patch:
        patch.setattr(SEG, "read_truth", broken)
        with pytest.raises(OSError, match="truth file"):
            EVAL.evaluate(**real_kwargs(tree, tmp_path/"e1"))
    assert real_like.is_file()
    with pytest.raises(dev_lock.DevReadLocked):
        EVAL.evaluate(**real_kwargs(tree, tmp_path/"e2"))


def test_bz_go_reads_again_and_is_recorded(tree, real_like, tmp_path):
    with pytest.raises(ValueError, match="nothing to unlock"):
        EVAL.evaluate(**real_kwargs(tree, tmp_path/"e0", bz_go="BZ, some day"))
    EVAL.evaluate(**real_kwargs(tree, tmp_path/"e1"))
    with pytest.raises(ValueError, match="words"):
        EVAL.evaluate(**real_kwargs(tree, tmp_path/"e2", bz_go="  "))
    words = "BZ, 11-05 9 am: read it again"
    document = EVAL.evaluate(**real_kwargs(tree, tmp_path/"e3", bz_go=words))
    assert document["dev_read"]["event"] == "bz_approved_reread" and document["dev_read"]["bz_go"] == words
    assert [r["event"] for r in dev_lock.describe()] == ["first_read", "bz_approved_reread"]
    summary = DECIDE.summary_markdown(DECIDE.decide(dict(made_evaluation(.006), dev_read=document["dev_read"]), MADE_RECORD))
    assert words in summary and "再次读取" in summary


def test_synthetic_path_does_not_touch_the_lock(tree, lock, tmp_path):
    EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e1"))
    assert not lock.exists()
    lock.parent.mkdir(parents=True)
    lock.write_text("a lock of a real read\n")
    assert EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e2"))["dev_read"] is None
    assert lock.read_text() == "a lock of a real read\n"
    with pytest.raises(ValueError, match="synthetic runs do not touch it"):
        EVAL.evaluate(**eval_kwargs(tree, tmp_path/"e3"), bz_go="BZ, some day")


def test_command_line_stops_with_exit_8_when_locked(lock, tmp_path, capsys):
    lock.parent.mkdir(parents=True)
    dev_lock.take(dict(out_dir="a"))
    assert EVAL.main(["--member-root", str(tmp_path/"root"), "--out-dir", str(tmp_path/"e")]) == 8
    assert "STOP" in capsys.readouterr().out and not (tmp_path/"e").exists()


# ---- the fix round: the control of the evaluator alone, before the training (review, minor 7) ----------------------------------------------------------------------------------------

def control_kwargs(tree, out_dir, **changes):
    kwargs = dict(eval_kwargs(tree, out_dir), **changes)
    kwargs.pop("member_root")
    return kwargs


def control_argv(tree, out_dir):
    return ["--control-only", "--synthetic", "--control-member", str(tree.control), "--reference", str(tree.reference), "--allowlist", str(tree.work/"dev_subjects.json"), "--folds", str(tree.work/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(tree.work/"data"), "--out-dir", str(out_dir), "--device", "cpu"]


def test_control_alone_passes_needs_no_u1_and_takes_no_lock(tree, lock, tmp_path):
    document = EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c"))
    assert document["passed"] is True and abs(document["differences"]["seg35"]) < 1e-12 and abs(document["differences"]["seg4"]) < 1e-12 and document["tolerance"] == DECIDE.CONTROL_TOL
    assert json.loads((tmp_path/"c"/"eval_control.json").read_text())["passed"] is True and not lock.exists()
    with pytest.raises(FileExistsError):
        EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c"))
    assert EVAL.main(control_argv(tree, tmp_path/"c2")) == 0


def test_control_alone_on_the_real_path_goes_past_an_existing_lock(tree, real_like, tmp_path):
    real_like.parent.mkdir(parents=True)
    real_like.write_text("a lock of a real read\n")
    document = EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c", synthetic=False))
    assert document["passed"] is True and real_like.read_text() == "a lock of a real read\n"


def test_control_alone_is_pinned_on_the_real_path(tree, real_like, tmp_path, monkeypatch):
    monkeypatch.setattr(EVAL, "CONTROL_SHA256", "0"*64)
    with pytest.raises(ValueError, match="not the pinned control"):
        EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c", synthetic=False))
    with pytest.raises(ValueError, match="the plan's"):
        EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c", synthetic=False, dev_source="synthetic_t0"))


def test_control_that_does_not_reproduce_fails_with_exit_8(tree, tmp_path):
    document = json.loads(Path(tree.reference).read_text())
    for row in document["per_subject"]["A0"]:
        row["seg35"] += .001
    document["summary"]["A0"]["seg35"] += .001
    tampered = tmp_path/"reference.json"
    tampered.write_text(json.dumps(document))
    result = EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c", reference_path=tampered))
    assert result["passed"] is False and result["differences"]["seg35"] == pytest.approx(-.001)
    argv = control_argv(tree, tmp_path/"c2")
    argv[argv.index(str(tree.reference))] = str(tampered)
    assert EVAL.main(argv) == 8


@pytest.mark.parametrize("source,match", [("synthetic_t0", "not a development subject"), (TREE.SOURCE, "not a development subject")])
def test_control_alone_refuses_t0_and_training_subjects_before_opening_a_file(tree, tmp_path, monkeypatch, source, match):
    opened = no_subject_file(monkeypatch)
    with pytest.raises(SubjectSealed, match=match):
        EVAL.evaluate_control(**control_kwargs(tree, tmp_path/"c", dev_source=source))
    assert not opened


def test_bz_go_does_not_go_with_the_control(tree, tmp_path):
    with pytest.raises(SystemExit):
        EVAL.main(control_argv(tree, tmp_path/"c")+["--bz-go", "BZ, some day"])


# ---- the fix round: the real steps on stubs, prepare, status, the logs (review, minors 3 and 11) -------------------------------------------------------------------------------------

def stub_real_step(monkeypatch):
    """No snapshot check, no gate, no step runs; records what would."""
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    monkeypatch.setattr(RUN, "require_snapshot", lambda: None)
    gates, steps = [], []
    monkeypatch.setattr(RUN, "wait_for_gate", lambda unit, minutes: gates.append(unit))
    monkeypatch.setattr(RUN, "run_step", lambda name, argv, **kwargs: steps.append((name, argv, kwargs)))
    return gates, steps


def write_decision(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/"steps_decision.json").write_text(json.dumps(dict(RUN.steps_rule(MEASURED))))


def write_control(out, passed=True):
    (Path(out)/RUN.CONTROL_DIR).mkdir(parents=True, exist_ok=True)
    (Path(out)/RUN.CONTROL_DIR/"eval_control.json").write_text(json.dumps(dict(passed=passed, differences=dict(seg35=0.), tolerance=2e-4)))


def test_smoke_step_on_stubs(tmp_path, monkeypatch):
    gates, steps = stub_real_step(monkeypatch)
    monkeypatch.setattr(RUN, "read_smoke", lambda directory: dict(MEASURED))
    write_control(tmp_path)
    RUN.cmd_smoke(types.SimpleNamespace(out=tmp_path, patch=None, steps=60, wait_minutes=30))
    assert gates == ["tag-ul-smoke"] and [name for name, _, _ in steps] == ["smoke_128x160x112", "control_check"]
    train = " ".join(steps[0][1])
    assert "--profile v3_large --fold A --seed 0" in train and "--set max_iters=60" in train and "--device cuda" in train and "--synthetic" not in train and str(tmp_path/"smoke_128x160x112") in train
    control = " ".join(steps[1][1])
    assert "--control-only" in control and "--synthetic" not in control and "--member-root" not in control and "--device cuda" in control
    assert "pytest" not in (train+control).replace(str(tmp_path), "") and (tmp_path/"run_ul.log").is_file()


def test_train_step_on_stubs(tmp_path, monkeypatch, lock):
    gates, steps = stub_real_step(monkeypatch)
    write_decision(tmp_path)
    write_control(tmp_path)
    RUN.cmd_train(types.SimpleNamespace(out=tmp_path, wait_minutes=30))
    assert gates == ["tag-ul-train"] and [name for name, _, _ in steps] == ["train_A0", "calibrate_A0", "check_members", "eval_u1", "decide_ul"]
    assert f"--set max_iters={RUN.steps_rule(MEASURED)['max_iters']}" in " ".join(steps[0][1])
    assert steps[3][1][1:3] == ["-B", str(CODE_ROOT/"experiments/ul/eval_u1.py")] and "--bz-go" not in steps[3][1]       # the chain never carries BZ's words


@pytest.mark.parametrize("missing,message", [("decision", "decide-steps"), ("control", "control"), ("failed_control", "did not reproduce"), ("lock", "read once")])
def test_train_step_refuses_before_gate_and_steps(tmp_path, monkeypatch, lock, capsys, missing, message):
    gates, steps = stub_real_step(monkeypatch)
    if missing != "decision":
        write_decision(tmp_path)
    if missing not in ("control", "failed_control"):
        write_control(tmp_path)
    if missing == "failed_control":
        write_control(tmp_path, passed=False)
    if missing == "lock":
        lock.parent.mkdir(parents=True)
        dev_lock.take(dict(out_dir="a"))
    assert RUN.main(["train", "--out", str(tmp_path)]) == 8
    assert not gates and not steps
    assert message in capsys.readouterr().out and message in (tmp_path/"run_ul.log").read_text()


def test_prepare(tmp_path, monkeypatch):
    made, checks = [], []

    def make(repo, dest, paths):
        made.append((repo, dest, paths))
        Path(dest).mkdir(parents=True)
        return dict(commit="abc")

    monkeypatch.setattr(RUN, "LOG_FILE", None)
    monkeypatch.setattr(RUN, "make_snapshot", make)
    monkeypatch.setattr(RUN, "verify_snapshot", lambda root: dict(commit="abc", verified=str(root)))
    monkeypatch.setattr(RUN, "env_record", lambda path, snapshot: Path(path).write_text("env\n"))
    answers = [types.SimpleNamespace(returncode=0, stdout="", stderr=""), types.SimpleNamespace(returncode=0, stdout="", stderr=""), types.SimpleNamespace(returncode=1, stdout="no", stderr="pin")]

    def fake_run(argv, **kwargs):
        checks.append((argv, kwargs))
        return answers.pop(0)

    monkeypatch.setattr(RUN.subprocess, "run", fake_run)
    out = tmp_path/"out"
    assert RUN.prepare(out, tmp_path/"repo") == dict(commit="abc")
    assert made == [(tmp_path/"repo", out/"snapshot", SNAPSHOT_PROFILES["v3_large"])]
    argv, kwargs = checks[0]
    assert argv[2:] == ["-m", "bayesreg.seg.unet.cli", "check-inputs", "--profile", "v3_large", "--require-snapshot"] and kwargs["cwd"] == str(out/"snapshot") and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert list(out.glob("env_*.txt"))
    assert RUN.prepare(out, tmp_path/"repo")["verified"] == str(out/"snapshot") and len(made) == 1     # made once, verified after
    with pytest.raises(RUN.Stop, match="pinned-input check failed"):
        RUN.prepare(out, tmp_path/"repo")


def test_status(tmp_path, lock, capsys):
    RUN.cmd_status(types.SimpleNamespace(out=tmp_path))
    text = capsys.readouterr().out
    assert "state: (none)" in text and "read lock" in text and text.count("(none)") >= 4
    write_decision(tmp_path)
    write_control(tmp_path)
    (tmp_path/"chain_state").mkdir()
    (tmp_path/"chain_state"/"train_A0.started").write_text("x")
    (tmp_path/"chain.log").write_text("a line of the chain\n")
    lock.parent.mkdir(parents=True)
    dev_lock.take(dict(out_dir="a"))
    RUN.cmd_status(types.SimpleNamespace(out=tmp_path))
    text = capsys.readouterr().out
    assert "train_A0.started" in text and '"ok": true' in text and '"passed": true' in text and "a line of the chain" in text and "first_read" in text


def test_steps_and_stops_reach_the_files(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    RUN.set_run_log(tmp_path)
    RUN.run_step("a", ["x", "y z"], state=tmp_path/"s", log_path=tmp_path/"chain.log", cwd=tmp_path, run=lambda argv, **k: types.SimpleNamespace(returncode=0))
    assert "start a: x 'y z'" in (tmp_path/"chain.log").read_text() and "end a: exit 0" in (tmp_path/"chain.log").read_text()
    assert "start a:" in (tmp_path/"run_ul.log").read_text() and "done a" in (tmp_path/"run_ul.log").read_text()


# ---- the fix round: what the reading says ---------------------------------------------------------------------------------------------------------------------------------------------

def test_time_truncation_is_marked_and_does_not_change_the_reading():
    config = dict(max_iters=30000, warmup_iters=200, lr=5e-4, poly_power=.9)
    record = dict(MADE_RECORD, stopped="wall_budget", iterations_run=25000, config=config)
    document = DECIDE.decide(made_evaluation(.001), record)
    assert document["reading"] == DECIDE.NOT_LIMIT and document["training"]["truncated_by_time"] is True
    schedule = document["training"]["schedule"]
    assert schedule["share_of_max_iters"] == pytest.approx(25000/30000) and schedule["lr_at_stop"] == pytest.approx(5e-4*(1-(24999-200)/29800)**.9)
    assert "训练被时间截断" in DECIDE.summary_markdown(document)
    finished = DECIDE.decide(made_evaluation(.001), dict(MADE_RECORD, iterations_run=30000, config=config))
    assert finished["training"]["truncated_by_time"] is False and "训练被时间截断" not in DECIDE.summary_markdown(finished) and finished["training"]["schedule"]["share_of_max_iters"] == 1.
    assert "早停" in DECIDE.summary_markdown(DECIDE.decide(made_evaluation(.001), dict(MADE_RECORD, stopped="early_stop", iterations_run=12000, config=config)))


def test_schedule_formula_is_trains(tmp_path):
    cfg = TrainConfigLarge(max_iters=1000, warmup_iters=10)
    for done in (1, 5, 10, 11, 500, 1000):
        schedule = DECIDE.schedule_at_stop(dict(iterations_run=done, config=asdict(cfg)))
        assert schedule["lr_at_stop"] == pytest.approx(train_module._lr_at(done-1, cfg))
    assert DECIDE.schedule_at_stop(dict(iterations_run=5)) is None and DECIDE.schedule_at_stop(None) is None


def test_note_says_what_the_reading_does_not_say():
    for words in ("早停参数", "步数也不同", "这个配方、这个步数", "没有测 nnU-Net 本身", "不说明传到配准有多少"):
        assert words in DECIDE.NOTE
    assert DECIDE.NOTE in DECIDE.summary_markdown(DECIDE.decide(made_evaluation(.006), MADE_RECORD))


def test_early_stop_parameters_of_u1_are_the_ones_the_plan_lists():
    """plan.md revision 1 and 配方差异.md: the eighth difference; v2's values are 10 and 4000."""
    assert (TrainConfigLarge().patience, TrainConfigLarge().min_iters) == (20, 10000) and (TrainConfig().patience, TrainConfig().min_iters) == (10, 4000)
