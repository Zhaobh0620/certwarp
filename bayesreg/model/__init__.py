"""F_B residual blocks, energy, gradient, GN Hessian action, metric and stable comparison (contract C04-C09.1)."""
from .blocks import (EXTRA_LOGVOL_BLOCKS, GAUSS_BLOCKS, NONPOSITIVE_GAUSS_JD, ChannelPoint, Curl, Diffusion, LogVolume,
                     LogVolumeClosed27, LogVolumeCorners, channel_c_dat, channel_cdat_weight, channel_expand,
                     channel_gn_diagonal, channel_jvp, channel_point, channel_vjp, gauss_jvp, gauss_value, gauss_vjp, logvol_extra_block,
                     merge_cotangents, nonpositive_logvol_jd)
from .energy import (BLOCK_NAMES, Evaluation, Metric, Reduction, brain_mean_inverse_tau2, build_metric, c_dat,
                     evaluate, gn_diagonal, gradient, half_square_change, hvp, logvol_change, metric_bP,
                     stable_reduction)
from .support import (BRAIN, DEFAULT_SUPPORT_THRESHOLD, SIGMA_DOMAINS, SUPPORT, SUPPORT_RULE, SUPPORT_SHARED,
                      SupportLevel, check_theta, check_theta_coarse, cpu_channel_rho, domain_record,
                      resolve_sigma_domain, support_level, theta_lower_bound)
from .state import (CLOSED27_SIMPSON, CLOSED27_UNIFORM, CLOSED27_WEIGHTS, LOGVOL_DEFAULT, LOGVOL_GAUSS27,
                    LOGVOL_GAUSS27_CLOSED27, LOGVOL_GAUSS27_CORNERS, LOGVOL_POINTS, ModelState, build_samplers,
                    require_extended_longdouble)

__all__ = ["BRAIN", "DEFAULT_SUPPORT_THRESHOLD", "SIGMA_DOMAINS", "SUPPORT", "SUPPORT_RULE", "SUPPORT_SHARED",
           "SupportLevel", "check_theta", "check_theta_coarse", "cpu_channel_rho", "domain_record",
           "resolve_sigma_domain", "support_level", "theta_lower_bound", "BLOCK_NAMES", "CLOSED27_SIMPSON", "CLOSED27_UNIFORM", "CLOSED27_WEIGHTS", "EXTRA_LOGVOL_BLOCKS",
           "GAUSS_BLOCKS", "LOGVOL_DEFAULT", "LOGVOL_GAUSS27", "LOGVOL_GAUSS27_CLOSED27", "LOGVOL_GAUSS27_CORNERS",
           "LOGVOL_POINTS", "NONPOSITIVE_GAUSS_JD", "ChannelPoint", "Curl", "Diffusion", "Evaluation", "LogVolume",
           "LogVolumeClosed27", "LogVolumeCorners", "Metric", "ModelState", "Reduction", "brain_mean_inverse_tau2",
           "build_metric", "build_samplers", "c_dat", "channel_c_dat", "channel_cdat_weight", "channel_expand", "channel_gn_diagonal",
           "channel_jvp", "channel_point", "channel_vjp", "evaluate", "gauss_jvp", "gauss_value", "gauss_vjp",
           "gn_diagonal", "gradient", "half_square_change", "hvp", "logvol_change", "logvol_extra_block",
           "merge_cotangents", "metric_bP", "nonpositive_logvol_jd", "require_extended_longdouble", "stable_reduction"]
