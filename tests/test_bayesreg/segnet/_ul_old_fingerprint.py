"""The fingerprint of the old profiles v1 and v2_full on synthetic subjects (CPU, one thread), for test_ul.py: "the old profiles do not move".

_ul_old_fingerprint.json was written by this file from the code of commit 1933643, before any line of work package UL existed (runs/20261104-ul/plan.md):

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -B -m tests.test_bayesreg.segnet._ul_old_fingerprint --write

test_ul.py computes the same numbers from the current code and compares them bit by bit (sha256 of the raw bytes of weights, logits and probabilities; float.hex of the temperatures). Bitwise on this machine and this torch only (the test
skips under another torch version, as the bitwise replicas of this repository do).
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from bayesreg.seg.unet.calibrate import calibrate_member
from bayesreg.seg.unet.cli import dry_run, dry_run_v2
from bayesreg.seg.unet.model import build_model
from bayesreg.seg.unet.train import TrainConfig, predict_logits, train_member

GOLDEN = Path(__file__).with_name("_ul_old_fingerprint.json")
TINY = dict(base_channels=4, depth=2, max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)
TORCH_VERSION = "2.5.1"


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def sha_state(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def sha_json(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def tiny_subject(name, shape, seed, n_classes=36):
    from bayesreg.seg.unet.data import Subject
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, shape).astype(np.uint8)
    image = (labels/n_classes+rng.normal(0, .01, shape)).astype(np.float32)
    image[labels == 0] = 0
    return Subject(name, image, labels, image_sha256="0"*64, label_sha256="1"*64, normalization=dict(percentile=99.5, scale=1.))


def member_fingerprint(root, config):
    """train_member -> calibrate_member with `subjects=`; weights, temperature and what member.pt carries."""
    train = [tiny_subject(f"OASIS_OAS1_09{n:02d}_MR1", (16, 16, 16), n) for n in (1, 2)]
    calib = [tiny_subject("OASIS_OAS1_0903_MR1", (16, 16, 16), 3)]
    record = train_member(files=None, guard=None, fold="A", seed=0, models_root=root, config=config, device="cpu", log=lambda _: None, command=["fingerprint"], subjects=(train, calib))
    member, _ = calibrate_member(files=None, guard=None, fold="A", seed=0, models_root=root, device="cpu", log=lambda _: None, command=["fingerprint"], subjects=calib)
    blob = torch.load(Path(root)/"foldA_s0"/"member.pt", map_location="cpu", weights_only=True)
    return dict(state=sha_state(blob["state_dict"]), temperature=float(member["temperature"]).hex(), member_keys=sorted(blob), config=sha_json(record["config"]), model_config=blob["model_config"], record_keys=sorted(record),
                best_calib=float(record["best_calib_mean_dice35"]).hex(), window=blob["window"], amp=blob["amp"])


def logits_fingerprint():
    torch.manual_seed(0)
    model = build_model(dict(base_channels=4, depth=2)).eval()
    image = np.random.default_rng(0).random((12, 16, 20)).astype(np.float32)
    return dict(whole=sha_tensor(predict_logits(model, image, device="cpu")), window=sha_tensor(predict_logits(model, image, device="cpu", window=(8, 8, 8))),
                window_25=sha_tensor(predict_logits(model, image, device="cpu", window=(8, 8, 12), overlap=.25)))


def probs_fingerprint(work, subjects):
    return {s: dict(probs36=sha_tensor(torch.from_numpy(np.load(Path(work)/"models"/"pred"/s/"probs36.npz")["p"])), groups=sha_tensor(torch.from_numpy(np.load(Path(work)/"models"/"pred"/s/"structure_groups.npz")["p"]))) for s in subjects}


def fingerprint(scratch):
    torch.set_num_threads(1)
    scratch = Path(scratch)
    out = dict(config_default=sha_json(asdict(TrainConfig())), logits=logits_fingerprint())
    torch.manual_seed(0)
    default_model = build_model(TrainConfig().model_config())
    out["model_default_init"] = sha_state(default_model.state_dict())
    del default_model
    out["model_config_default"] = TrainConfig().model_config()
    out["whole_volume_member"] = member_fingerprint(scratch/"whole", dict(TINY))
    out["patch_member"] = member_fingerprint(scratch/"patch", dict(TINY, patch_size=(8, 8, 8)))
    v1 = dry_run(scratch/"v1", config=dict(TINY), log=lambda _: None)
    assert v1["ok"], v1
    out["v1_probs"] = probs_fingerprint(scratch/"v1", sorted(v1["inferred"]))
    out["v1_temperatures"] = {k: float(v["temperature"]).hex() for k, v in v1["members"].items()}
    v2 = dry_run_v2(scratch/"v2", config=dict(TINY), log=lambda _: None)
    assert v2["ok"], v2
    out["v2_probs"] = probs_fingerprint(scratch/"v2", sorted(v2["inferred"]))
    out["v2_temperatures"] = {k: float(v["temperature"]).hex() for k, v in v2["members"].items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        document = fingerprint(scratch)
    if args.write:
        if GOLDEN.exists():
            raise SystemExit(f"{GOLDEN} exists; written once from the pristine code")
        GOLDEN.write_text(json.dumps(dict(commit="1933643", torch=torch.__version__, numpy=np.__version__, fingerprint=document), indent=1)+"\n", encoding="utf-8")
    if args.check:
        saved = json.loads(GOLDEN.read_text(encoding="utf-8"))["fingerprint"]
        print("identical" if saved == json.loads(json.dumps(document)) else "DIFFERENT")
    print(json.dumps(document, indent=1)[:400])


if __name__ == "__main__":
    main()
