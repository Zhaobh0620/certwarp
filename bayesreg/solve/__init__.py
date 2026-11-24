"""B-line solver (contract C07-C13, C16): backend + certificate, PG inner, outer options (a)/(b),
multilevel transfer with back-off, stopping, export; the C09.1-3a JD affine-scaling PG metric (jd_scaling).
Backends: numpy (CPU reference) and torch64 (device FP64 with the GPU-path stable comparison of C08-8; torch is
imported only when that backend is built)."""
from .backend import (BACKENDS, CertOutcome, CertificateContractError, NumpyBackend, Torch64Backend, certificate_config,
                      certificate_record, certify, make_backend, module_backend, rejection_reason)
from .export import (displacement_to_voxels, export_native, voxels_to_displacement, write_disp, write_final_state)
from .checkpoints import CheckpointConfig, ProfileRecorder
from .init_field import InitFieldRejected, InitSeed, load_init_field
from .multilevel import (CHANNEL_SOURCES, TRANSFER_ALPHAS, Eta, FixedSigma, GaussianMStep, LevelInput, RunResult,
                         SupportMStep, SupportSigma,
                         SigmaMonitor, SolverConfig, eta_temporary, first_certified, level_inputs, solve_multilevel,
                         transfer)
from .jd_scaling import JDScaledMetric, JDScalingConfig, node_jd_estimate, scaled_metric, scaling_factors
from .outer import Commit, CommitTracker, RoundResult, solve_direct, solve_proximal, solve_round
from .pg import DirectObjective, PGConfig, ProximalObjective, bb2_proposal, run_pg, sampling_step_bound
from .proximal import ProximalParams, default_lambda
from .stopping import (RUN_STATUSES, RUN_WORDING, Caps, continuation, final_status, g0_of, gf_b, gradient_norm_mass,
                       step_ratio, threshold)

__all__ = ["BACKENDS", "CHANNEL_SOURCES", "Caps", "CertOutcome", "CertificateContractError", "CheckpointConfig", "Commit",
           "CommitTracker", "DirectObjective", "Eta", "FixedSigma", "GaussianMStep", "InitFieldRejected", "InitSeed",
           "JDScaledMetric", "JDScalingConfig", "LevelInput", "NumpyBackend", "PGConfig", "ProfileRecorder",
           "ProximalObjective", "ProximalParams", "RUN_STATUSES", "RUN_WORDING", "RoundResult", "RunResult",
           "SigmaMonitor", "SolverConfig", "SupportMStep", "SupportSigma", "TRANSFER_ALPHAS", "Torch64Backend", "bb2_proposal", "certificate_config",
           "certificate_record", "certify", "continuation", "default_lambda", "displacement_to_voxels",
           "eta_temporary", "export_native", "final_status", "first_certified", "g0_of", "gf_b", "gradient_norm_mass",
           "level_inputs", "load_init_field", "make_backend", "module_backend", "node_jd_estimate", "rejection_reason", "run_pg", "scaled_metric",
           "scaling_factors", "sampling_step_bound", "solve_direct",
           "solve_multilevel", "solve_proximal", "solve_round", "step_ratio", "threshold", "transfer",
           "voxels_to_displacement", "write_disp", "write_final_state"]
