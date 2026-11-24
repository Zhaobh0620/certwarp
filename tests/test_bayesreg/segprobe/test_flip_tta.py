"""SEGP P2 (runs/20261105-segprobe/任务书.md): the left-right flip tool experiments/segprobe/flip_tta.py and the read-back of experiments/segprobe/segprobe_common.py. CPU, synthetic volumes, tmp_path only; the real repository
files read are the label table, N0's record of the table and the allowlist with its sealed lists (ids only). No image, no label, nothing of a results directory.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"segprobe_{name}", REPO/"experiments/segprobe"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


flip_tta = _load("flip_tta")
common = sys.modules["segprobe_common"]
eval_seg = sys.modules["eval_seg"]


class SideOracle(nn.Module):
    """Logits from the image and the position, as a network trained on unflipped images behaves: a brain voxel at low x is class `left`, at high x class `right`; a voxel above .5 of the midline structure stays class 13
    wherever it is; background otherwise. Fully convolutional in the way the tool uses it: the padded input comes in, the logits of that shape go out."""
    n_classes, multiple = 36, 4

    def __init__(self, left=1, right=20, gain=20.):
        super().__init__()
        self.left, self.right, self.gain = left, right, gain

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.n_classes)+tuple(x.shape[2:]), dtype=x.dtype)
        brain = (x[:, 0] > 0).to(x.dtype)
        midline = (x[:, 0] > .5).to(x.dtype)
        half = x.shape[-1]//2
        out[:, 0] = self.gain*(1-brain)
        out[:, self.left, ..., :half] = self.gain*brain[..., :half]
        out[:, self.right, ..., half:] = self.gain*brain[..., half:]
        out[:, 13] = 2*self.gain*midline
        return out


def member_of(model, temperature=1.):
    return dict(model=model, temperature=temperature, amp=False, window=None, gaussian_sigma=None, percentile=99.5)


def blob(shape=(16, 16, 16), seed=0):
    """An asymmetric volume whose width is the padded width, so that the oracle's midline is the middle of the image."""
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=np.float32)
    image[3:13, 3:13, 2:14] = .3
    image[5:9, 5:9, 7:9] = .9                 # a midline structure
    image[4:6, 4:6, 3:5] = 0                  # a hole on the left only
    return image*(1+.01*rng.random(shape, dtype=np.float32))


# ---- the table -----------------------------------------------------------------------------------------------------------------------------------

def test_table_from_the_label_names_is_an_involution_with_the_midline_fixed():
    lut = flip_tta.swap_table()
    assert lut.dtype == torch.int64 and lut.shape == (36,)
    assert torch.equal(lut[lut], torch.arange(36))
    assert [l for l in range(36) if lut[l] == l] == [0, 11, 12, 13]
    names = json.loads((REPO/"configs/bayesreg/structure_groups_v1.json").read_text())["label_table"]["names"]
    for label in range(36):
        name, partner = names[str(label)], names[str(int(lut[label]))]
        if name.startswith("Left-"):
            assert partner == "Right-"+name[len("Left-"):]
        elif name.startswith("Right-"):
            assert partner == "Left-"+name[len("Right-"):]
        else:
            assert partner == name
    assert int(lut[1]) == 20 and int(lut[2]) == 21 and int(lut[19]) == 35 and int(lut[14]) == 30


def test_table_equals_the_one_n0_used():
    record = json.loads((REPO/"runs/20261103-n0/n0_gpu_log.json").read_text())
    assert flip_tta.swap_table().tolist() == record["permutation"]
    assert record["lr_axis_zyx"] == 2 and flip_tta.FLIP_DIM == -1            # x of (nz, ny, nx), the axis N0 checked on the labels


# ---- flip back -----------------------------------------------------------------------------------------------------------------------------------

def test_flip_back_mirrors_x_only_and_swaps_the_channels():
    q = torch.zeros(36, 4, 5, 6)
    q[1, 2, 3, 0] = 1.                        # the mirror image: "left" at x = 0, z = 2, y = 3
    out = flip_tta.flip_back(q, flip_tta.swap_table())
    assert out[20, 2, 3, 5] == 1. and out.sum() == 1.        # the original image: right, at the mirrored x; z and y unchanged


def test_flip_back_is_its_own_inverse():
    q = torch.rand(36, 4, 5, 6)
    lut = flip_tta.swap_table()
    assert torch.equal(flip_tta.flip_back(flip_tta.flip_back(q, lut), lut), q)


def test_a_network_that_reads_sides_by_position_gives_the_same_labels_in_both_directions():
    """The whole path on an oracle that is right about the sides: the flip back and the swap must bring the mirror's answer back onto the unflipped one."""
    image = blob()
    with torch.no_grad():
        unflipped, flip_only, tta, seconds = flip_tta.tta_probabilities([member_of(SideOracle())], image, device="cpu", lut=flip_tta.swap_table())
    maps = flip_tta.tops(unflipped, flip_only, tta)
    assert seconds >= 0
    assert set(np.unique(maps["top"])) == {0, 1, 13, 20}
    assert np.array_equal(maps["top"], maps["top_flip"])
    assert np.array_equal(maps["top"], maps["top_tta"])
    assert maps["top"].dtype == np.uint8
    assert torch.allclose(unflipped, flip_only, atol=1e-5)


def test_negative_control_without_the_swap_the_sides_are_wrong():
    """The test above would catch a missing swap: without it, the mirror's answer has left and right the wrong way round."""
    image = blob()
    mirrored = np.flip(image, axis=-1).copy()
    with torch.no_grad():
        q = torch.softmax(flip_tta.predict_logits(SideOracle(), mirrored, device="cpu", amp=False), dim=1)[0]
    no_swap = torch.flip(q, (-1,)).argmax(0).numpy()
    with torch.no_grad():
        p = torch.softmax(flip_tta.predict_logits(SideOracle(), image, device="cpu", amp=False), dim=1)[0]
    top = p.argmax(0).numpy()
    assert (no_swap[top == 1] == 20).all() and (no_swap[top == 20] == 1).all()


def test_negative_control_a_flip_along_another_axis_is_seen():
    image = blob()
    lut = flip_tta.swap_table()
    with torch.no_grad():
        q = torch.softmax(flip_tta.predict_logits(SideOracle(), np.flip(image, axis=-1).copy(), device="cpu", amp=False), dim=1)[0]
        p = torch.softmax(flip_tta.predict_logits(SideOracle(), image, device="cpu", amp=False), dim=1)[0]
    wrong = torch.flip(q, (-2,))[lut].argmax(0).numpy()
    assert not np.array_equal(wrong, p.argmax(0).numpy())


def test_tta_is_the_renormalized_mean_of_the_two_directions_over_the_members():
    torch.manual_seed(1)
    from bayesreg.seg.unet.model import build_model
    a, b = build_model(dict(base_channels=4, depth=2)).eval(), build_model(dict(base_channels=4, depth=2)).eval()
    image = blob()
    lut = flip_tta.swap_table()
    with torch.no_grad():
        unflipped, flip_only, tta, _ = flip_tta.tta_probabilities([member_of(a, 1.3), member_of(b, .9)], image, device="cpu", lut=lut)
        p = [torch.softmax(flip_tta.predict_logits(m, image, device="cpu", amp=False)/t, dim=1)[0] for m, t in ((a, 1.3), (b, .9))]
        q = [flip_tta.flip_back(torch.softmax(flip_tta.predict_logits(m, np.flip(image, -1).copy(), device="cpu", amp=False)/t, dim=1)[0], lut) for m, t in ((a, 1.3), (b, .9))]
    assert torch.allclose(unflipped, (p[0]+p[1])/2, atol=1e-5)
    assert torch.allclose(flip_only, (q[0]+q[1])/2, atol=1e-5)
    assert torch.allclose(tta, (p[0]+p[1]+q[0]+q[1])/4, atol=1e-5)
    for tensor in (unflipped, flip_only, tta):
        assert float((tensor.sum(0)-1).abs().max()) < 1e-5


def test_odd_sizes_are_padded_and_the_mirror_is_of_the_image_not_of_the_padding():
    """The padding is at the high end, so the mirror must be taken before it: the mirror's answer has the shape of the image and lies over the same voxels."""
    image = blob(shape=(16, 16, 14))
    with torch.no_grad():
        unflipped, flip_only, _, _ = flip_tta.tta_probabilities([member_of(SideOracle())], image, device="cpu", lut=flip_tta.swap_table())
    assert unflipped.shape == flip_only.shape == (36, 16, 16, 14)
    assert np.array_equal(unflipped.argmax(0).numpy() > 0, image > 0)
    assert np.array_equal(flip_only.argmax(0).numpy() > 0, image > 0)


# ---- the pipeline without a GPU -------------------------------------------------------------------------------------------------------------------

def _pipeline(tmp_path):
    subjects = ["OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1"]
    volumes = {s: blob(seed=i) for i, s in enumerate(subjects)}
    members_of = dict(v1=[member_of(SideOracle())], v2=[member_of(SideOracle(), 2.)])
    reads = []

    def read_volume(subject):
        reads.append(subject)
        return volumes[subject].astype(np.float64), dict(sha256="0"*64)

    seconds = flip_tta.infer_all(subjects, members_of, read_volume=read_volume, device="cpu", lut=flip_tta.swap_table(), out_dir=tmp_path, log=lambda *a, **k: None)
    return subjects, volumes, seconds, reads


def _truth_from(volumes, lookup_seg4):
    def read_truth(subject, kind):
        image = volumes[subject]
        half = image.shape[-1]//2
        labels = np.zeros(image.shape, dtype=np.uint8)
        labels[..., :half][image[..., :half] > 0] = 1
        labels[..., half:][image[..., half:] > 0] = 20
        labels[image > .5] = 13
        return (labels if kind == "seg35" else lookup_seg4[labels].astype(np.uint8)), "f"*64
    return read_truth


def test_infer_writes_the_three_maps_and_counts_the_forward_passes_without_any_truth(tmp_path):
    subjects, _, seconds, reads = _pipeline(tmp_path)
    assert reads == subjects
    for name in ("v1", "v2"):
        for subject in subjects:
            assert seconds[name][subject]["n_forward"] == 2 and seconds[name][subject]["forward_s"] >= 0
            with np.load(tmp_path/name/f"{subject}.npz") as npz:
                assert sorted(npz.files) == ["top", "top_flip", "top_tta"]


def test_evaluation_of_the_maps_with_the_functions_of_eval_seg(tmp_path):
    subjects, volumes, seconds, _ = _pipeline(tmp_path)
    lookup = eval_seg.load_seg4_lookup()
    rows, truth_sha = flip_tta.evaluate_tops(subjects, ["v1", "v2"], out_dir=tmp_path, read_truth=_truth_from(volumes, lookup), lookup=lookup)
    assert sorted(truth_sha) == sorted(subjects)
    tables = flip_tta.tabulate(rows, subjects, dict(v1=(1., 1.), v2=(1., 1.)))
    for name in ("v1", "v2"):
        table = tables[name]
        for arm in flip_tta.ARMS:
            assert table["means"][arm]["seg35"] == pytest.approx(1.)
            assert table["means"][arm]["seg4"] == pytest.approx(1.)
        assert table["tta_minus_unflipped"]["seg35"]["mean"] == pytest.approx(0.)
        assert table["worth_using"] is False                 # no gain
        assert table["control"] is None                      # two subjects: the committed means are over 20
    document = dict(tables=tables, seconds=seconds)
    assert "不值得用" in flip_tta.summary_markdown(document)


def test_the_rule_of_n0_g():
    subjects = [f"s{i}" for i in range(20)]

    def row(value):
        return dict(seg35=value, seg4=value, groups={g: value for g in eval_seg.GROUPS})

    def tables(gain, n_positive):
        rows = dict(v1={s: dict(unflipped=row(.9), flip_only=row(.9), tta=row(.9+(gain if i < n_positive else 0.))) for i, s in enumerate(subjects)})
        return flip_tta.tabulate(rows, subjects, dict(v1=(.9, .9)))["v1"]

    assert tables(.0030, 20)["worth_using"] is True          # mean .0030, 20/20
    assert tables(.0030, 14)["worth_using"] is False         # 14/20
    assert tables(.0001, 20)["worth_using"] is False         # the mean is too small
    assert tables(.0030, 20)["control"]["passed"] is True


# ---- refusals ----------------------------------------------------------------------------------------------------------------------------------

def test_a_subject_outside_the_admitted_ones_is_refused():
    admitted = ["OASIS_OAS1_0438_MR1", "OASIS_OAS1_0439_MR1"]
    assert flip_tta.select_subjects(admitted, None) == admitted
    assert flip_tta.select_subjects(admitted, ["OASIS_OAS1_0439_MR1"]) == ["OASIS_OAS1_0439_MR1"]
    with pytest.raises(flip_tta.SubjectSealed):
        flip_tta.select_subjects(admitted, ["OASIS_OAS1_0001_MR1"])


def test_the_admitted_subjects_are_the_20_development_subjects():
    subjects, record = common.admit_dev_subjects()
    assert subjects == [f"OASIS_OAS1_{k:04d}_MR1" for k in range(438, 458)]
    assert record["dev_source"] == "stage_L2R_val_v1"


class FakeCuda:
    def __init__(self, free_gib, available=True):
        self.free, self.available = free_gib, available

    def is_available(self):
        return self.available

    def mem_get_info(self, device):
        return int(self.free*flip_tta.GIB), int(24*flip_tta.GIB)


def test_a_busy_gpu_is_refused():
    assert flip_tta.require_free_gpu("cuda", 12., cuda=FakeCuda(20.))["free_gib"] == pytest.approx(20.)
    with pytest.raises(flip_tta.GpuBusy):
        flip_tta.require_free_gpu("cuda", 12., cuda=FakeCuda(5.))
    with pytest.raises(ValueError):
        flip_tta.require_free_gpu("cuda", 12., cuda=FakeCuda(20., available=False))


def test_results_are_never_overwritten(tmp_path):
    (tmp_path/"old").write_text("x")
    with pytest.raises(FileExistsError):
        flip_tta.run(names=["v1"], out_dir=tmp_path, device="cpu")
    assert (tmp_path/"old").read_text() == "x"


def test_dry_run_on_the_cpu_touches_no_gpu_no_image_and_no_truth(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the dry run must not read this")
    monkeypatch.setattr(eval_seg, "read_truth", refuse)
    monkeypatch.setattr(flip_tta, "read_image", refuse)
    monkeypatch.setattr(flip_tta, "require_free_gpu", refuse)
    monkeypatch.setattr(flip_tta, "infer_all", refuse)
    monkeypatch.setattr(torch.cuda, "synchronize", refuse)
    lines = []
    record = flip_tta.dry_run(["v1", "v2"], ["OASIS_OAS1_0438_MR1"], allowlist=None, real_members=False, log=lines.append)
    assert record["lut"] == flip_tta.swap_table().tolist()
    assert any("16 pairs" in line for line in lines)
    assert record["synthetic_seconds"] >= 0


# ---- the read-back, slab by slab -----------------------------------------------------------------------------------------------------------------

def _stored(tmp_path, seed=0, shape=(36, 20, 6, 5)):
    rng = np.random.default_rng(seed)
    p = rng.random(shape)
    p /= p.sum(axis=0)
    raw = p.astype(np.float16)
    path = tmp_path/f"p{seed}.npz"
    np.savez_compressed(path, p=raw)
    return path, raw


def test_slab_read_back_equals_read_back_probabilities_bit_by_bit(tmp_path):
    from bayesreg.seg.channels import read_back_probabilities
    path, raw = _stored(tmp_path)
    array, record = read_back_probabilities(path)
    assert common.check_raw(raw, "x", slab=7) == record["raw_max_abs_sum_deviation"]
    rebuilt = np.concatenate([common.normalized_slab(raw, a, min(a+7, raw.shape[1])) for a in range(0, raw.shape[1], 7)], axis=1)
    assert rebuilt.tobytes() == array.tobytes()
    assert common.argmax_of_mean([raw], slab=7).tobytes() == np.argmax(array, axis=0).astype(np.uint8).tobytes()


def test_argmax_of_the_mean_is_the_mean_of_the_read_backs(tmp_path):
    from bayesreg.seg.channels import read_back_probabilities
    path1, raw1 = _stored(tmp_path, 1)
    path2, raw2 = _stored(tmp_path, 2)
    a1, _ = read_back_probabilities(path1)
    a2, _ = read_back_probabilities(path2)
    assert common.argmax_of_mean([raw1, raw2], slab=7).tobytes() == np.argmax((a1+a2)/2, axis=0).astype(np.uint8).tobytes()


def test_ties_go_to_the_smallest_label():
    raw = np.zeros((36, 2, 1, 1), dtype=np.float16)
    raw[3], raw[7] = .5, .5
    assert common.argmax_of_mean([raw]).tolist() == [[[3]], [[3]]]


@pytest.mark.parametrize("mutate, message", [(lambda r: r.__setitem__((0, 0, 0, 0), -.1), "negative"), (lambda r: r.__setitem__((0, 0, 0, 0), np.nan), "non-finite"),
                                             (lambda r: r.__setitem__((slice(None), 0, 0, 0), 0.), "zero total"), (lambda r: r.__setitem__((0, 0, 0, 0), r[0, 0, 0, 0]+.1), "deviates")])
def test_check_raw_makes_the_checks_of_the_read_back(tmp_path, mutate, message):
    _, raw = _stored(tmp_path)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        common.check_raw(raw, "x", slab=7)


def test_check_raw_refuses_another_storage_type():
    with pytest.raises(ValueError, match="stored as"):
        common.check_raw(np.ones((36, 1, 1, 1), dtype=np.int16), "x")
