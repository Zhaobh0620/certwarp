"""OPT1 reference timing (work package OPT1; for reference only, not a speed claim): the device work of one PG
iteration on a synthetic 64^3 problem, before OPT1 (the verbatim fd93113 functions of
tests/test_bayesreg/torch64/_opt1_reference.py) and with OPT1 (C2a Jacobian cache, C3 speculative gradient).

One iteration = evaluate(trial) + the C08-8 device comparison gpu_fb_change(base, trial) + gradient(trial) (an accepted
step; a rejected candidate is the first two).  The base is evaluated once per arm before timing (with OPT1 it holds its
Jacobian cache, as the solver's base does); each repeated trial is dropped after its iteration.  Synthetic data only
(tests/test_bayesreg/solve/_synthetic.py: plane-wave image, smooth u*, C = 1), no data/ access.  Warm-up 1 iteration per
arm (not counted), then `--repeats` iterations alternating old / new; cuda synchronized around every timed call; medians
and ranges; the device peak per arm.  The arms' results are compared bitwise (energy, tile sums, Delta F_B, gradient).
The GPU state (nvidia-smi) and the load average are recorded: a shared GPU makes the numbers indicative only.
An exception (e.g. a CUDA error) stops the script without retry; the record written so far is saved with an "error"
entry (type, message, traceback, case and stage, UTC time, the GPU state at the error) and the exit code is 1 (OPT1b,
review docs/bayesreg/reviews/OPT1_复核_20261030.md item 4; the cuda attempt of OPT1 left no record).

    python -B experiments/opt1/time_iteration.py --device cuda --out runs/20261030-spd/opt1_timing_synth64.json
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bayesreg.model import ModelState, build_samplers                       # noqa: E402
import bayesreg.backends.torch64 as T                                       # noqa: E402
from bayesreg.backends.torch64 import stable as S                           # noqa: E402
from tests.test_bayesreg.solve import _synthetic as syn                     # noqa: E402
from tests.test_bayesreg.torch64 import _opt1_reference as R                # noqa: E402
from tests.test_bayesreg.torch64._cases import smooth_field                 # noqa: E402


def problem(n, points):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.true_displacement(grid, (2., -1.5, 1.5)))
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 2), sigma2=np.array([1e-2]), alpha=1e-2, nu=1e-2,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), logvol_points=points)
    u = smooth_field(grid, .3, 5)
    return state, u, u+smooth_field(grid, .05, 6)


def sample_gpu():
    try:
        query = "--query-gpu=name,memory.used,memory.total,utilization.gpu"
        out = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", query, "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        out = f"unavailable: {exc}"
    return dict(nvidia_smi=out, loadavg=os.getloadavg())


class Arm:
    def __init__(self, name, state, device, u, u_trial, chunk_cells):
        self.name = name
        self.old = name == "pre_opt1"
        options = dict(jacobian_cache_slots=0, speculative_gradient=False) if self.old else {}
        self.ds = T.TorchModelState(state, device, chunk_cells=chunk_cells, **options)
        self.evaluate = R.evaluate if self.old else T.evaluate
        self.change = R.gpu_fb_change if self.old else S.gpu_fb_change
        self.gradient = R.gradient if self.old else T.gradient
        self.u_trial = u_trial
        self.base = self.evaluate(self.ds, u)
        self.times = dict(evaluate=[], compare=[], gradient=[])
        self.result = None

    def iteration(self, sync, record=True):
        ticks = [time.perf_counter()]
        trial = self.evaluate(self.ds, self.u_trial)
        sync()
        ticks.append(time.perf_counter())
        delta, blocks = self.change(self.ds, self.base, trial)
        sync()
        ticks.append(time.perf_counter())
        g = self.gradient(self.ds, trial)
        sync()
        ticks.append(time.perf_counter())
        if record:
            for k, name in enumerate(("evaluate", "compare", "gradient")):
                self.times[name].append(ticks[k+1]-ticks[k])
        result = (trial.energy.hex(), trial.tile_sums.tobytes(), repr(delta), T.to_numpy(g).tobytes())
        if self.result is None:
            self.result = result
        assert result == self.result, "nondeterministic"
        del trial, g


def summary(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), n=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-cells", type=int, default=32768, help="TorchModelState chunk_cells (shared GPU)")
    parser.add_argument("--points", default="gauss27+corners,gauss27")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    sources = {path: hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest() for path in (
        "bayesreg/backends/torch64/model.py", "bayesreg/backends/torch64/stable.py",
        "bayesreg/backends/torch64/q1.py", "tests/test_bayesreg/torch64/_opt1_reference.py",
        "experiments/opt1/time_iteration.py")}
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                          "CUDA_VISIBLE_DEVICES")}
    record = dict(purpose="OPT1 reference timing only (shared machine; not a speed claim)", head=sha,
                  tracked_changes_uncommitted=bool(dirty), source_sha256=sources,
                  device=T.Device(args.device).describe(), n=args.n, repeats=args.repeats, chunk_cells=args.chunk_cells,
                  torch_threads=torch.get_num_threads(), env=env,
                  gpu_before=sample_gpu(), cases={})
    points = stage = None
    try:
        for points in args.points.split(","):
            stage = "setup"
            state, u, u_trial = problem(args.n, points)
            arms, peaks = {}, {}
            for name in ("pre_opt1", "opt1"):
                stage = f"warm-up {name}"
                if device.type == "cuda":
                    sync()
                    before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                arms[name] = Arm(name, state, args.device, u, u_trial, args.chunk_cells)
                arms[name].iteration(sync, record=False)                       # warm-up
                if device.type == "cuda":
                    sync()
                    peaks[name] = torch.cuda.max_memory_allocated(device)-before
            for repeat in range(args.repeats):
                for name in ("pre_opt1", "opt1"):
                    stage = f"repeat {repeat} {name}"
                    arms[name].iteration(sync)
            assert arms["pre_opt1"].result == arms["opt1"].result, "OPT1 not bitwise"
            case = dict(cells=state.grid.n_cells, chunks=len(arms["opt1"].ds.q1.chunks), bitwise_equal=True,
                        peak_bytes=peaks or None, jacobian_cache_bytes_base=arms["opt1"].base.jacobian_cache_nbytes)
            for name, arm in arms.items():
                times = {k: summary(v) for k, v in arm.times.items()}
                accepted = [sum(t) for t in zip(*arm.times.values())]
                rejected = [a+b for a, b in zip(arm.times["evaluate"], arm.times["compare"])]
                times.update(accepted_step=summary(accepted), rejected_candidate=summary(rejected))
                case[name] = times
            case["ratio_new_over_old_median"] = {k: case["opt1"][k]["median"]/case["pre_opt1"][k]["median"]
                                                 for k in case["opt1"]}
            record["cases"][points] = case
            print(points, json.dumps(case["ratio_new_over_old_median"]), flush=True)
    except Exception as exc:          # OPT1b (review OPT1_复核_20261030 item 4): keep the evidence, stop, no retry
        record["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                               points=points, stage=stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               gpu_at_error=sample_gpu(), retried=False)
        print(f"error in {points} / {stage}: {type(exc).__name__}: {exc} (recorded in {args.out}; not retried)",
              file=sys.stderr, flush=True)
    record["gpu_after"] = sample_gpu()
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=1)
    return 1 if "error" in record else 0


if __name__ == "__main__":
    sys.exit(main())
