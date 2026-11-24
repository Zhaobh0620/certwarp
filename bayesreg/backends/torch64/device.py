"""Device choice and host <-> device transfers of the torch FP64 backend (work package G1).

`Device` is adapted from the mainline class of the same name:
    source  archive/seq_vp_3d/jd_curl/three_d/backends/torch_ops.py lines 33-95 (Transfers, Device)
    commit  eb808be401fde7ed6fee73c2d86212c8c463a513, file sha256
            b314fdb0207d2707cfa5ace4f2a3cab27990666817680b6839794aba43472088
Kept: explicit cpu/cuda only, the allocation probe that fails instead of falling back, float64 only,
the transfer counters, `check`, `download`, `synchronize`.  Changed: `upload` always copies (the E76
lean_host switch is dropped), `upload_index` / `upload_bool` added for int64 / bool arrays;
`download_owned` and `scalar` are not copied.
"""
from dataclasses import dataclass

import numpy as np
import torch

from jd_curl.validation import InputValidationError


def default_device():
    """C17 / G1: cuda when available, otherwise cpu (the cpu device runs the same code, for tests)."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Transfers:
    upload_calls: int = 0
    upload_bytes: int = 0
    download_calls: int = 0
    download_bytes: int = 0


class Device:
    def __init__(self, name=None):
        requested = default_device() if name is None else torch.device(name)
        if requested.type not in ("cpu", "cuda"):
            raise InputValidationError("only explicit cpu/cuda devices supported")
        # Allocation must fail if CUDA is unavailable: never silently fall back.
        self.device = torch.empty(0, device=requested, dtype=torch.float64).device
        self.transfers = Transfers()

    @property
    def type(self):
        return self.device.type

    def upload(self, array):
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        self.transfers.upload_calls += 1
        self.transfers.upload_bytes += array.nbytes
        return torch.tensor(array, device=self.device, dtype=torch.float64)

    def upload_index(self, array):
        array = np.array(array, dtype=np.int64, copy=True, order="C")
        self.transfers.upload_calls += 1
        self.transfers.upload_bytes += array.nbytes
        return torch.tensor(array, device=self.device, dtype=torch.int64)

    def upload_bool(self, array):
        array = np.array(array, dtype=bool, copy=True, order="C")
        self.transfers.upload_calls += 1
        self.transfers.upload_bytes += array.nbytes
        return torch.tensor(array, device=self.device, dtype=torch.bool)

    def check(self, x, shape=None):
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or x.device != self.device
                or x.requires_grad or (shape is not None and tuple(x.shape) != tuple(shape))):
            raise InputValidationError("device/float64/shape/autograd mismatch")

    def download(self, tensor):
        """A fresh C-contiguous host copy (float64, int64 or bool tensors)."""
        if not isinstance(tensor, torch.Tensor) or tensor.device != self.device or tensor.requires_grad:
            raise InputValidationError("download: a tensor of this device without autograd required")
        self.transfers.download_calls += 1
        self.transfers.download_bytes += tensor.numel()*tensor.element_size()
        return np.ascontiguousarray(tensor.detach().cpu().numpy()).copy()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def describe(self):
        """gpu-port procedure, section 3: device model, driver-side CUDA, torch version."""
        out = dict(device=str(self.device), torch=torch.__version__, torch_cuda=torch.version.cuda,
                   dtype="float64")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            out.update(gpu=props.name, total_memory_bytes=int(props.total_memory),
                       capability=f"{props.major}.{props.minor}")
        return out
