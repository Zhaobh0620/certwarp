"""Estimates of the GPU memory and of the time per step of the profile v3_large (work package UL, runs/20261104-ul/plan.md).  NOT MEASURED: nothing here runs on the GPU; the smoke run (run_ul.py smoke) measures, and decide-steps uses only what it measures.

Each network runs once on the meta device (no memory, no data) with hooks that count, per layer, the elements of the output and the multiply-accumulates.  The only measurements are v2's, unet35_v2_full (RTX 4090, AMP, whole volume, batch 1):
  12.09 GB peak allocated (train_record eval_history max_memory_gb of foldA_s0; the four members 12.090-12.101) and .1836 s per step (foldA_s1, median over the spans of 50 steps of train_log.jsonl without an evaluation; augmentation, loss and optimizer included).

Memory, two bounds with one measured point:
  M = 16 B * parameters (weights, gradients, two Adam states) + loss_tensors * 36 * voxels * 4 B + k * activation elements,  k fitted to 12.09 GB at v2's shape.
  loss_tensors = 0 (everything scales as the activations) and 5 (float32 logits, log-softmax, softmax and two gradient buffers at full resolution, which do not grow with the base width).
Time: t = .1836 s * MACs / MACs of v2.  Augmentation on the whole volume runs once per sample, so batch 2 costs a second one this model does not count; a wider net usually uses the GPU better per MAC; both left out.

    python -B experiments/ul/estimate.py
"""
import json
from pathlib import Path
import sys

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bayesreg.seg.unet.large import PATCH_U1, UNet3DDeepSupervision, windows_per_volume  # noqa: E402
from bayesreg.seg.unet.model import UNet3D, count_parameters  # noqa: E402

VOLUME = (192, 224, 160)                     # (z, y, x), the provenance of the v2 predictions: grid.shape_zyx
V2_PEAK_GB = 12.09                           # unet35_v2_full foldA_s0 train_record: 12.0898, the four members 12.090-12.101
V2_STEP_S = .1836                            # unet35_v2_full foldA_s1 train_log.jsonl
GIB = 2**30


def count(model, shape):
    """(activation elements, MACs, elements of the outputs of the heads) of one forward pass of the model in training mode on (B, 1, D, H, W)."""
    totals = dict(activations=0, macs=0)
    handles = []

    def hook(module, inputs, output):
        elements = output.numel()
        totals["activations"] += elements
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            kernel = module.kernel_size[0]*module.kernel_size[1]*module.kernel_size[2]
            totals["macs"] += output.shape[0]*output.shape[2]*output.shape[3]*output.shape[4]*kernel*module.in_channels*module.out_channels

    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.InstanceNorm3d)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        out = model(torch.zeros(shape, device="meta"))
    for handle in handles:
        handle.remove()
    outs = out if isinstance(out, list) else [out]
    return dict(totals, head_elements=sum(o.numel() for o in outs), parameters=count_parameters(model))


def network(base, ds_levels):
    with torch.device("meta"):
        if ds_levels:
            return UNet3DDeepSupervision(base_channels=base, ds_levels=ds_levels).train()
        return UNet3D(base_channels=base).train()


def fit(reference, loss_tensors):
    fixed = 16*reference["parameters"]+loss_tensors*4*reference["head_elements"]
    return (V2_PEAK_GB*1e9-fixed)/reference["activations"]


def memory_gib(row, k, loss_tensors):
    return (16*row["parameters"]+loss_tensors*4*row["head_elements"]+k*row["activations"])/GIB


def main():
    reference = count(network(16, 0), (1, 1)+VOLUME)
    k0, k5 = fit(reference, 0), fit(reference, 5)
    print(f"v2 (base 16, whole volume {VOLUME}, batch 1): activations {reference['activations']/1e9:.2f} G, MACs {reference['macs']/1e9:.0f} G, parameters {reference['parameters']/1e6:.2f} M; measured {V2_PEAK_GB} GB, {V2_STEP_S} s per step")
    print(f"fit: k = {k0:.2f} B per activation element if loss_tensors = 0, {k5:.2f} if 5\n")
    print("NOT MEASURED. estimate, base 32, deep supervision on levels 1 and 2 (the model of the file header):")
    print(f"{'patch (z,y,x)':>16} {'batch':>5} {'vox/step M':>10} {'peak GiB, loss_tensors 0':>25} {'5':>6} {'s/step':>7} {'steps in 4.05 h':>16} {'windows .5':>10} {'.25':>5} {'fwd s, 8 subj eval':>18}")
    rows = []
    for patch, batch in ((PATCH_U1, 2), ((128, 128, 128), 2), ((128, 176, 128), 2), ((144, 176, 128), 2), ((112, 144, 96), 2), ((160, 192, 128), 1), ((192, 224, 160), 1)):
        row = count(network(32, 2), (batch, 1)+patch)
        seconds = V2_STEP_S*row["macs"]/reference["macs"]
        one = count(network(32, 0), (1, 1)+patch)
        window_seconds = V2_STEP_S/3.3*one["macs"]/reference["macs"]         # forward only: v2's evaluation of 8 whole volumes took about .44 s, a step .18 s, so a forward pass is about a third of a step or less
        doc = dict(patch=patch, batch=batch, voxels_per_step=batch*patch[0]*patch[1]*patch[2], peak_gib_0=memory_gib(row, k0, 0), peak_gib_5=memory_gib(row, k5, 5), seconds_per_step=seconds, steps_4h=4.05*3600/seconds,
                   windows_50=windows_per_volume(VOLUME, patch, .5)[0], windows_25=windows_per_volume(VOLUME, patch, .25)[0], forward_seconds_per_window=window_seconds, parameters=row["parameters"], parameters_plain=one["parameters"])
        rows.append(doc)
        print(f"{str(patch):>16} {batch:>5} {doc['voxels_per_step']/1e6:>10.2f} {doc['peak_gib_0']:>25.1f} {doc['peak_gib_5']:>6.1f} {seconds:>7.2f} {doc['steps_4h']:>16.0f} {doc['windows_50']:>10d} {doc['windows_25']:>5d} {8*doc['windows_50']*window_seconds:>18.1f}")
    print("\nJSON "+json.dumps(dict(reference=reference, rows=rows)))


if __name__ == "__main__":
    main()
