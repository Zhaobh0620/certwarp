"""3D U-Net of the structure-layer segmenter (docs/bayesreg/04_分割器.md §3). Pure torch.

Four down-sampling stages (strided 3x3x3 convolutions), base width 16 doubling per stage
(16, 32, 64, 128, 256), every convolution followed by InstanceNorm3d(affine) and LeakyReLU(0.01);
transposed-convolution up-sampling, skip concatenation, a 1x1x1 head with 36 logits (seg35 labels
0..35, class c = label c). Fully convolutional: any input whose spatial sizes are multiples of
2**depth works; `pad_to_multiple` / `crop_to` handle the rest.
"""
import torch
from torch import nn
import torch.nn.functional as F

N_CLASSES = 36


class ConvBlock(nn.Module):
    """Two (conv 3x3x3 -> InstanceNorm -> LeakyReLU) layers; the first may be strided."""

    def __init__(self, cin, cout, stride=1, negative_slope=.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(cout, affine=True, eps=1e-5),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.InstanceNorm3d(cout, affine=True, eps=1e-5),
            nn.LeakyReLU(negative_slope, inplace=True))

    def forward(self, x):
        return self.layers(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, n_classes=N_CLASSES, base_channels=16, depth=4, max_channels=320):
        super().__init__()
        if depth < 1 or base_channels < 1:
            raise ValueError("depth and base_channels must be positive")
        widths = [min(base_channels*2**k, max_channels) for k in range(depth+1)]
        self.depth, self.widths, self.n_classes = depth, tuple(widths), n_classes
        self.encoder = nn.ModuleList([ConvBlock(in_channels, widths[0])]
                                     + [ConvBlock(widths[k-1], widths[k], stride=2) for k in range(1, depth+1)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(widths[k+1], widths[k], 2, stride=2) for k in range(depth)])
        self.decoder = nn.ModuleList([ConvBlock(2*widths[k], widths[k]) for k in range(depth)])
        self.head = nn.Conv3d(widths[0], n_classes, 1)

    @property
    def multiple(self):
        return 2**self.depth

    def forward(self, x):
        if any(n % self.multiple for n in x.shape[2:]):
            raise ValueError(f"spatial sizes {tuple(x.shape[2:])} must be multiples of {self.multiple}")
        skips = []
        for k, block in enumerate(self.encoder):
            x = block(x)
            if k < self.depth:
                skips.append(x)
        for k in reversed(range(self.depth)):
            x = self.decoder[k](torch.cat([self.up[k](x), skips[k]], dim=1))
        return self.head(x)


def build_model(config):
    """UNet3D from the `model` part of a training config (dict)."""
    return UNet3D(in_channels=1, n_classes=config.get("n_classes", N_CLASSES),
                  base_channels=config.get("base_channels", 16), depth=config.get("depth", 4),
                  max_channels=config.get("max_channels", 320))


def pad_to_multiple(x, multiple):
    """Zero-pad the last three axes of `x` at the high end to multiples of `multiple`; returns (padded, shape)."""
    shape = tuple(x.shape[-3:])
    pads = [(-n) % multiple for n in shape]
    if not any(pads):
        return x, shape
    return F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0])), shape


def crop_to(x, shape):
    return x[..., :shape[0], :shape[1], :shape[2]]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


__all__ = ["N_CLASSES", "UNet3D", "build_model", "count_parameters", "crop_to", "pad_to_multiple"]
