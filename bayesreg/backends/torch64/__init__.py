"""torch FP64 backend of bayesreg.model (work package G1): F_B, gradient, GN HVP, GN diagonal, c_dat and the
C09.1 metric on cuda (or cpu), same names and semantics as the numpy reference; see model.py for the
device / precision split and the memory estimate.  Importing this package imports torch."""
from .device import Device, default_device
from .model import (GEOMETRY_CACHE_BYTES, TorchChannelPoint, TorchEvaluation, TorchMetric, TorchModelState,
                    build_metric, c_dat, domain_check, estimate_peak_bytes, evaluate, gn_diagonal, gradient, hvp,
                    jacobian_slots, node_field_cm, resolved_state_options, to_numpy)
from .q1 import CHUNK_CELLS, DeviceQ1, dst1, poisson_solve, stiffness_apply
from .spline import SPLINE_CHUNK, TorchSpline, cubic_basis, sample_channels

__all__ = ["CHUNK_CELLS", "GEOMETRY_CACHE_BYTES", "SPLINE_CHUNK", "Device", "DeviceQ1", "TorchChannelPoint",
           "TorchEvaluation", "TorchMetric", "TorchModelState", "TorchSpline", "build_metric", "c_dat",
           "cubic_basis", "default_device", "domain_check", "dst1", "estimate_peak_bytes", "evaluate",
           "gn_diagonal", "gradient", "hvp", "jacobian_slots", "node_field_cm", "poisson_solve",
           "resolved_state_options", "sample_channels", "stiffness_apply", "to_numpy"]
