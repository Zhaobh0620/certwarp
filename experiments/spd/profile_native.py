"""Native-level per-kernel profiler of the B-line solver (work package SPD; bench rules; tool only).

Question: where does one native-level PG step of the torch64 GPU path spend its wall clock?  Every item is timed on
ONE fixed native state (by default the G2 final state of the M2 GPU smoke) on up to three paths -- torch64 on cuda,
torch64 on cpu, the numpy CPU reference -- plus the path-independent host work of the solver (certificate, JD scaling,
BB2, domain bound, vector bookkeeping).  A fixed-state timing says how long that kernel takes on that state; it is NOT
a whole-registration speed and must not be converted into one (project rules, wording rules).

    python -B experiments/spd/profile_native.py --out DIR
        [--run-dir /home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G2]  (config.json, hyper.json, log.jsonl, timing.json,
                                                                    final_native.npz; read only)
        [--state PATH] [--pair STAGE/PAIR] [--data-root DIR] [--allowlist PATH] [--tissue-root DIR]
        [--paths cuda,torch-cpu,numpy] [--repeats 5] [--warmup 1] [--items PREFIX,...]
        [--iteration-steps 3] [--iteration-repeats R] [--iteration-warmup auto|N] [--cpu-threads N]
        [--numba-threads N] [--cert-thread-sweep 4,8,16] [--torch-profiler] [--with-fallback-items]
        [--no-sync-count] [--max-worker-seconds S]
        [--allow-shared-gpu] [--allow-busy-units] [--skip-gate] [--no-gpu-query] [--keep-trial] [--allow-dirty]

Order.  The driver checks the gate (below), writes cmd.txt, git.txt, env.txt, data.txt, config.json, then runs one
worker process per path, strictly in series: always `host` first (it builds the fixed trial state, see below), then the
requested backend paths in the given order.  Each worker loads the inputs itself through the label firewall
(bayesreg.io.load_pair: pair.npz members fixed / moving only, the development allowlist; Atropos tissue posteriors via
run_pair.prepare_channels), builds the native ModelState of the run (eta and the native round's sigma^2 from the run
record, logvol_points, tile size, cpu_threads), times its items and writes path_<name>.json; its stdout/stderr go to
worker_<name>.log.  A worker that fails is recorded and the next one still runs (nothing is retried).  The driver then
writes profile.json and profile.md.  The trial file (trial.npz, ~160 MB at native size) is deleted at the end unless
--keep-trial (its sha256 stays in profile.json).

Fixed states.  base u_b = the run's final_native.npz u (state_hash checked against the file).  The host worker
computes, with the numpy CPU reference, g = grad F_B(u_b), the PG direction p = -S~ M^-1 S~ g with M frozen at the
run's recorded native (c_dat, b_P) and S~ at u_b, and the trial u_t = inject_free(theta_b + tau p): tau starts at the
median accepted native tau of the run's log.jsonl (else min(1, .9 tau_dom)) and is halved until the CPU certificate
passes and F_B is defined; the first rejected tau (or a doubled tau, at most 12 doublings) gives the rejected-candidate
certificate timing.  r = c1 tau |g.p| is the required decrease of the PG comparison.  All paths time u_b and u_t.

Items (seconds per call; warmup calls timed separately; median, IQR, min, max of --repeats calls):
  host:   certificate.pass / .pass_base / .reject (CPU FP64, PrescreenCertificateConfig of C07-2; numba threads
          recorded; --cert-thread-sweep repeats .pass at each thread count via numba.set_num_threads and checks that
          the outcome is identical), jd.scaling_factors (C09.1-3a S~ estimate, host numpy), jd.apply_host (S~ g twice),
          pg.bb2_proposal, pg.domain_bound, pg.slope (g.p), pg.candidate (theta + tau p), pg.isfinite, pg.inject_free,
          pg.readonly_copy, pg.step_ratio, pg.gf_b, pg.state_hash, pg.node_field_copy (as Torch64Backend.evaluate).
          --numba-threads N: every worker calls numba.set_num_threads(N) first (N <= NUMBA_NUM_THREADS), so that a
          launch with NUMBA_NUM_THREADS=16 can sweep the certificate up to 16 threads while every other item runs with
          the run's N (G2: 4).
  per backend path (cuda / torch-cpu / numpy):
          energy.data (channel block: domain check, spline sampling, residual, host longdouble sum; torch also
          energy.data_device), energy.geometry_gauss, energy.diffusion, energy.logvol_gauss, energy.curl,
          energy.geometry_corners, energy.logvol_corners (each standalone: the geometry it needs + that block's
          residual and square sums; "incremental" = block - geometry is derived in the report), kq.apply (K_Q u,
          stiffness_apply: the diffusion term of the gradient), energy.evaluate (the backend's evaluate as the solver
          calls it; torch also energy.evaluate_device without the upload), gradient.channel / .diffusion / .logvol /
          .curl / .total (backend.gradient with blocks=...; torch also gradient.total_device without the download),
          metric.c_dat, metric.build, metric.solve, metric.jd_solve (S~ M^-1 S~ g, the PG direction), torch also
          metric.solve_device; compare.* (torch: compare.gpu = Torch64Backend.stable_compare with r, the C08-8 GPU
          path incl. the fallback decision -- the path taken is recorded per call --, compare.device_gauss_changes,
          compare.channel_host, compare.decision (gpu_guard + fallback_rule); cuda with --with-fallback-items:
          compare.cpu_fallback_cold (fresh comparator: both states on the CPU reference + C08-1..7) and
          compare.cpu_fallback_warm (base from the base slot) -- off by default: they are the CPU reference functions
          the numpy path times anyway (energy.evaluate + compare.cpu_reference), they cost minutes per call set at
          native size, and the G2 native level had no fallback (its log.jsonl); numpy: compare.cpu_reference =
          stable_reduction), transfer.upload_u / transfer.download_free
          (cuda only), and
          iteration: run_pg (C09.2, the solver's own code) from u_b for --iteration-steps accepted steps, --iteration-
          repeats times, through a timing proxy of the backend and of the JD-scaled metric (each proxied call is
          synchronized before its clock stops; values are untouched -- tested bitwise); per iteration the wall clock,
          the candidates and their reasons, and the seconds of certify / evaluate / stable_compare (or
          stable_reduction) / gradient / confirm_gradient_norm / metric solve / JD refresh; "host_other" = wall - sum.
          Iteration 1 starts from the fixed state without BB2 history (tau = initial step); iterations >= 2 use BB2.
Per item also: peak device memory (torch.cuda.max_memory_allocated / reserved, reset per item), host peak RSS of the
item (VmHWM after writing 5 to /proc/self/clear_refs; null when the kernel refuses), host<->device transfers counted by
bayesreg.backends.torch64.Device (upload/download calls and bytes per call; implicit scalar reads such as bool(tensor)
are not counted there), and on cuda the synchronizing operations of one extra call under
torch.cuda.set_sync_debug_mode("warn") (experimental in torch: not every synchronizing operation warns), and with
--torch-profiler one extra call under torch.profiler (best effort: kernel count, top operators by device time).
GPU timing: torch.cuda.synchronize(device) before the clock starts and before it stops, for every timed call.

Report.  profile.md: the item table (median [IQR] per path), the incremental block costs, resources, the iteration
breakdown, a model that weights the item medians by the run's native-level counts (candidates, certificate passes and
rejections, evaluations, comparisons, gradients, metric solves, JD computations, CPU fallbacks; read from hyper.json and
log.jsonl) next to the run's measured native-level seconds -- labelled as a model, not a measurement --, and
"主要瓶颈", which only restates measured shares (no inference).

Gate (skipped with --skip-gate, tests only): three nvidia-smi samples 1 s apart (when cuda is requested): GPU busy if
utilization >= 10 % or memory.used >= 1500 MiB (--gpu-busy-util, --gpu-busy-mib: the Windows desktop holds some GPU
memory under WSL; raise the floor only after looking at an idle sample) or a compute process is listed (WSL's nvidia-smi
usually lists none) -> STOP unless --allow-shared-gpu (then
every cuda number is marked "GPU shared: no timing conclusion"); active or activating seq-vp-* / tag-* user units other
than the unit running this script -> STOP unless --allow-busy-units; MemAvailable >= 8 GiB.  The gate samples are
written to profile.json and repeated after the last worker.  --max-worker-seconds: a worker that has used it up skips
its remaining items (recorded as skipped; an item in progress is never interrupted).

Rules kept: jd_curl/ and bayesreg/ are not modified (internal functions are imported); the certificate is the CPU FP64
call of bayesreg.solve.backend; nothing is written outside --out; the run directory is only read.
"""
import argparse
from collections import Counter
from contextlib import closing
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.boundary import inject_free, restrict_free  # noqa: E402
from jd_curl.three_d.cell_q1 import cell_tiles  # noqa: E402
from jd_curl.validation import readonly_copy  # noqa: E402

from bayesreg.geom import node_field, ordered_map, tile_geometry  # noqa: E402
from bayesreg.geom import stiffness_apply as numpy_stiffness_apply  # noqa: E402
from bayesreg.model import Curl, Diffusion, LogVolume, LogVolumeCorners  # noqa: E402
from bayesreg.model import channel_point as numpy_channel_point  # noqa: E402
from bayesreg.model.energy import _node_change_sum, _node_sum  # noqa: E402
from bayesreg.solve import (DirectObjective, Eta, JDScaledMetric, JDScalingConfig, PGConfig, bb2_proposal,  # noqa: E402
                            make_backend, run_pg, sampling_step_bound, scaling_factors)
from bayesreg.solve.multilevel import _template  # noqa: E402
from bayesreg.solve.stopping import gf_b, state_hash, step_ratio  # noqa: E402

LD = np.longdouble
SCHEMA = "spd.profile_native.v1"
PATH_SCHEMA = "spd.profile_native.path.v1"
DEFAULT_RUN = Path("/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G2")
BACKEND_PATHS = ("cuda", "torch-cpu", "numpy")
NATIVE_LEVEL = 3
GPU_BUSY_UTIL = 10                 # percent
GPU_BUSY_MIB = 1500
MEM_FLOOR_MIB = 8*1024
TAU_MAX_DOUBLINGS = 12
ITERATION_COMPONENTS = ("certify", "evaluate", "stable_compare", "stable_reduction", "gradient",
                        "confirm_gradient_norm", "metric_solve", "jd_refresh")
BLOCKS_GAUSS = ("geometry_gauss", "diffusion", "logvol_gauss", "curl")
BLOCKS_CORNERS = ("geometry_corners", "logvol_corners")


# ----------------------------------------------------------------------------------------------
# Statistics and resources
# ----------------------------------------------------------------------------------------------

def summarize(samples):
    """n, median, q25, q75, iqr, min, max, mean of a list of seconds (numpy linear percentiles); None when empty."""
    values = np.asarray([float(s) for s in samples], dtype=np.float64)
    if values.size == 0:
        return None
    q25, median, q75 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return dict(n=int(values.size), median=median, q25=q25, q75=q75, iqr=q75-q25, min=float(values.min()),
                max=float(values.max()), mean=float(values.mean()))


class HostPeak:
    """Peak RSS of a section: VmHWM after `echo 5 > /proc/self/clear_refs` (Linux >= 4.0); None when refused."""

    def __init__(self):
        self.supported = None

    def reset(self):
        try:
            with open("/proc/self/clear_refs", "w") as handle:
                handle.write("5")
            self.supported = True
        except OSError:
            self.supported = False

    @staticmethod
    def vm(key):
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith(key+":"):
                        return int(line.split()[1])/1024.
        except OSError:
            return None
        return None

    def peak_mb(self):
        return self.vm("VmHWM") if self.supported else None


def lifetime_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.


# ----------------------------------------------------------------------------------------------
# Gate and environment
# ----------------------------------------------------------------------------------------------

def nvidia_smi():
    for candidate in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def parse_gpu_query(text):
    """Rows of `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits` -> list of dicts (numbers as float; None when unparsable)."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def number(value):
            try:
                return float(value)
            except ValueError:
                return None
        rows.append(dict(name=parts[0], driver=parts[1], memory_used_mib=number(parts[2]),
                         memory_total_mib=number(parts[3]), utilization_pct=number(parts[4])))
    return rows


def gpu_busy(rows, apps, *, util=GPU_BUSY_UTIL, mib=GPU_BUSY_MIB):
    """GPU busy (gate): utilization >= util %, memory.used >= mib MiB or a listed compute process."""
    reasons = []
    for row in rows:
        if row["utilization_pct"] is not None and row["utilization_pct"] >= util:
            reasons.append(f"utilization {row['utilization_pct']:g} % >= {util:g}")
        if row["memory_used_mib"] is not None and row["memory_used_mib"] >= mib:
            reasons.append(f"memory.used {row['memory_used_mib']:g} MiB >= {mib:g}")
    if apps:
        reasons.append(f"{len(apps)} compute process(es) listed")
    return reasons


def gpu_sample(util=GPU_BUSY_UTIL, mib=GPU_BUSY_MIB):
    smi = nvidia_smi()
    if smi is None:
        return dict(available=False)
    out = dict(available=True, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        query = subprocess.run([smi, "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        apps = subprocess.run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(available=False, error=str(exc))
    out["gpus"] = parse_gpu_query(query.stdout)
    out["compute_apps"] = [line.strip() for line in apps.stdout.strip().splitlines()
                           if line.strip() and "No running" not in line]
    out["busy_reasons"] = gpu_busy(out["gpus"], out["compute_apps"], util=util, mib=mib)
    return out


def own_unit():
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::") and line.rstrip().endswith(".service"):
            return line.rstrip().rsplit("/", 1)[-1]
    return None


def active_units():
    """Active or activating user units named seq-vp-* or tag-* (other than the unit running this script)."""
    try:
        done = subprocess.run(["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend", "--no-pager",
                               "seq-vp-*", "tag-*"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(error=str(exc), units=[])
    mine = own_unit()
    units = []
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ("active", "activating") and parts[0] != mine:
            units.append(line.strip())
    return dict(units=units, own_unit=mine)


def mem_available_mib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])//1024
    except OSError:
        return None
    return None


def gate(args, *, when):
    record = dict(when=when, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), skipped=bool(args.skip_gate))
    if args.no_gpu_query:
        record["gpu"] = "not queried (--no-gpu-query)"
    else:
        samples = []
        for k in range(3):
            samples.append(gpu_sample(args.gpu_busy_util, args.gpu_busy_mib))
            if k < 2:
                time.sleep(1.)
        record["gpu"] = samples
    record["units"] = active_units()
    record["mem_available_mib"] = mem_available_mib()
    try:
        record["home_free_gb"] = shutil.disk_usage("/home").free/1e9
    except OSError:
        record["home_free_gb"] = None
    stops = []
    busy = []
    if isinstance(record["gpu"], list):
        for sample in record["gpu"]:
            busy += sample.get("busy_reasons", [])
    record["gpu_busy"] = sorted(set(busy))
    if "cuda" in args.paths and record["gpu_busy"] and not args.allow_shared_gpu:
        stops.append("GPU busy: " + "; ".join(record["gpu_busy"]) + " (--allow-shared-gpu to run anyway)")
    if record["units"].get("units") and not args.allow_busy_units:
        stops.append("active units: " + "; ".join(record["units"]["units"]) + " (--allow-busy-units)")
    if record["mem_available_mib"] is not None and record["mem_available_mib"] < MEM_FLOOR_MIB:
        stops.append(f"MemAvailable {record['mem_available_mib']} MiB < {MEM_FLOOR_MIB}")
    record["gpu_shared"] = bool("cuda" in args.paths and record["gpu_busy"])
    record["stops"] = [] if args.skip_gate else stops
    return record


def git_record():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return f"<git failed: {exc}>"
    status = git("status", "--porcelain", "--untracked-files=no")
    return dict(head=git("rev-parse", "HEAD").strip(), dirty=bool(status.strip()), status=status,
                diff=git("diff", "HEAD"))


THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
              "NUMBA_CACHE_DIR", "CUDA_VISIBLE_DEVICES")


def env_record():
    import scipy
    out = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               packages=dict(numpy=np.__version__, scipy=scipy.__version__),
               thread_env={k: os.environ.get(k) for k in THREAD_ENV}, cpu_count=os.cpu_count(),
               longdouble_nmant=int(np.finfo(np.longdouble).nmant))
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import numba
        out["packages"]["numba"] = numba.__version__
        out["numba"] = dict(config_threads=int(numba.config.NUMBA_NUM_THREADS), threads=int(numba.get_num_threads()))
    except Exception as exc:               # recorded, never fatal
        out["numba"] = dict(error=str(exc))
    return out


def torch_record(device=None):
    import torch
    out = dict(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()
               if torch.backends.cudnn.is_available() else None, torch_threads=torch.get_num_threads(),
               torch_interop_threads=torch.get_num_interop_threads())
    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        out.update(gpu=props.name, capability=f"{props.major}.{props.minor}", total_memory_bytes=int(props.total_memory),
                   multiprocessors=int(props.multi_processor_count))
    return out


# ----------------------------------------------------------------------------------------------
# Run record and inputs
# ----------------------------------------------------------------------------------------------

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_round(hyper):
    rows = [r for r in hyper.get("rounds", []) if r.get("level") == NATIVE_LEVEL]
    if not rows:
        raise SystemExit("run record: hyper.json has no native-level round")
    return rows[-1]


def log_summary(path):
    """Native-level counts of the run's log.jsonl (the weights of the report's model)."""
    path = Path(path)
    if not path.is_file():
        return None
    candidates, accepts, jd = [], [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("level") != NATIVE_LEVEL:
                continue
            if row.get("kind") == "candidate":
                candidates.append(row)
            elif row.get("kind") == "accept":
                accepts.append(row)
            elif row.get("kind") == "jd_scaling":
                jd += 1
    comparisons = [r["comparison"] for r in candidates if r.get("comparison")]
    seconds_gpu = [c.get("seconds_gpu") or 0. for c in comparisons]
    seconds_cpu = [c.get("seconds_cpu") or 0. for c in comparisons]
    cert = Counter(r.get("cert_status") for r in candidates)
    taus = [r["tau"] for r in accepts]
    return dict(candidates=len(candidates), accepted=len(accepts),
                cert_pass=int(cert.get("certified_feasible", 0)),
                cert_reject=len(candidates)-int(cert.get("certified_feasible", 0))-int(cert.get(None, 0)),
                not_certified=int(cert.get(None, 0)),
                evaluations=sum(1 for r in candidates if r.get("energy") is not None),
                comparisons=len(comparisons), fallbacks=sum(1 for c in comparisons if c.get("fallback")),
                comparison_seconds_gpu=summarize(seconds_gpu), comparison_seconds_gpu_sum=float(sum(seconds_gpu)),
                comparison_seconds_cpu_sum=float(sum(seconds_cpu)), jd_computations=jd,
                accepted_tau=summarize(taus), reasons=dict(Counter(str(r.get("reason")) for r in candidates)))


def load_run_record(run_dir):
    run_dir = Path(run_dir)
    config = _read_json(run_dir/"config.json")
    hyper = _read_json(run_dir/"hyper.json")
    row = native_round(hyper)
    timing = _read_json(run_dir/"timing.json") if (run_dir/"timing.json").is_file() else None
    native_seconds = None
    if timing:
        for level in (timing.get("solve_levels") or {}).get("levels", []):
            if level.get("level") == NATIVE_LEVEL:
                native_seconds = level.get("seconds")
    metric = row.get("metric") or {}
    jd = metric.get("jd_scaling") or {}
    return dict(run_dir=str(run_dir), args=config.get("args", {}), eta=config.get("eta"),
                tile_size=(config.get("solver") or {}).get("gauss_tile_size", 8192),
                metric_jd_scaling=(config.get("args") or {}).get("metric_jd_scaling", "affine_v1"),
                native=dict(sigma2_used=row.get("sigma2_used"), T_B=row.get("T_B"), G0=row.get("G0"),
                            c_dat=metric.get("c_dat"), b_P=metric.get("b_P"), counters=row.get("counters"),
                            accepted=row.get("accepted"), status=row.get("status"), seconds_round=row.get("seconds"),
                            jd_computations=len(jd.get("computations", [])) or None,
                            jd_seconds=jd.get("seconds")),
                native_level_seconds=native_seconds,
                timing=None if timing is None else dict(certificate_seconds=timing.get("certificate_seconds"),
                                                        certificate_calls=timing.get("certificate_calls"),
                                                        comparison_seconds=timing.get("comparison_seconds"),
                                                        solve=timing.get("solve"), resources=timing.get("resources")),
                gpu_comparison=None if "gpu_comparison" not in hyper else {
                    k: hyper["gpu_comparison"].get(k) for k in ("comparisons", "cpu_fallback", "seconds")},
                log=log_summary(run_dir/"log.jsonl"))


def resolved_inputs(args, record):
    a = record["args"]
    pick = lambda name, key: getattr(args, name) if getattr(args, name) is not None else a.get(key)   # noqa: E731
    out = dict(pair=pick("pair", "pair"), data_root=pick("data_root", "data_root"),
               allowlist=pick("allowlist", "allowlist"), tissue_root=pick("tissue_root", "tissue_root"),
               channel_source=a.get("channel_source", "none"), logvol_points=a.get("logvol_points", "gauss27"),
               cpu_threads=args.cpu_threads if args.cpu_threads is not None else int(a.get("cpu_threads", 4)),
               state=str(args.state) if args.state else str(Path(record["run_dir"])/"final_native.npz"))
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise SystemExit(f"inputs: {missing} unknown (give them or a run directory that records them)")
    if out["channel_source"] not in ("none", "tissue"):
        raise SystemExit("profile_native: channel_source none or tissue only (structure is not wired here)")
    return out


def load_native(inputs, record, logger):
    """Native ModelState of the run and the base u (read only), through the label firewall (load_pair)."""
    from bayesreg.cli.run_pair import prepare_channels, resolve_pair
    from bayesreg.io import load_pair
    tick = time.perf_counter()
    pair = load_pair(resolve_pair(inputs["data_root"], inputs["pair"]), allowlist=inputs["allowlist"])
    levels, _, _, _ = prepare_channels(pair, inputs["channel_source"], inputs["tissue_root"], logger)
    native = levels[-1]
    del levels
    seconds_inputs = time.perf_counter()-tick
    eta_record = record["eta"]
    eta = Eta(eta_record["alpha"], eta_record["nu"], tuple(eta_record["tau2"]), source=str(eta_record.get("source")))
    tick = time.perf_counter()
    host = _template(native, eta, native.grid.omega, int(record["tile_size"]), native.fixed.shape[0],
                     inputs["cpu_threads"], inputs["logvol_points"])
    state = host.replace(sigma2=np.asarray(record["native"]["sigma2_used"], dtype=np.float64))
    seconds_state = time.perf_counter()-tick
    with np.load(inputs["state"]) as data:
        u = np.array(data["u"], dtype=np.float64, order="C")
        stored = str(data["state_hash"]) if "state_hash" in data.files else None
    if u.shape != state.grid.shape+(3,):
        raise SystemExit(f"state: shape {u.shape} is not the native grid {state.grid.shape + (3,)}")
    digest = state_hash(u)
    if stored is not None and stored != digest:
        raise SystemExit("state: stored state_hash does not match the field")
    u.flags.writeable = False
    info = dict(pair=inputs["pair"], fixed_subject=pair.fixed_subject, moving_subject=pair.moving_subject,
                grid_shape_zyx=list(state.grid.shape), spacing_xyz=list(state.grid.spacing), n_cells=state.grid.n_cells,
                n_mask_nodes=int(state.node_index.size), n_channels=state.n_channels,
                logvol_points=state.logvol_points, tile_size=state.tile_size, cpu_threads=state.cpu_threads,
                sigma2=state.sigma2.tolist(), eta=eta.record(), state_path=inputs["state"], state_hash=digest,
                seconds_inputs=seconds_inputs, seconds_state=seconds_state)
    return state, u, info


# ----------------------------------------------------------------------------------------------
# Timing of one item
# ----------------------------------------------------------------------------------------------

class Recorder:
    """Times named items on one path and keeps their records (module docstring)."""

    def __init__(self, path, *, repeats, warmup, sync, device=None, transfers=None, items=None, sync_count=True,
                 torch_profiler=False, budget=None, logger=None):
        self.path, self.repeats, self.warmup = path, repeats, warmup
        self.sync, self.device, self.transfers = sync, device, transfers
        self.filters = items
        self.sync_count, self.torch_profiler = sync_count, torch_profiler
        self.budget = budget
        self.started = time.perf_counter()
        self.items = {}
        self.host_peak = HostPeak()
        self.logger = logger or logging.getLogger("spd")

    def wanted(self, name):
        return not self.filters or any(name.startswith(prefix) for prefix in self.filters)

    def over_budget(self):
        return self.budget is not None and time.perf_counter()-self.started > self.budget

    def _transfer_snapshot(self):
        t = self.transfers
        return None if t is None else (t.upload_calls, t.upload_bytes, t.download_calls, t.download_bytes)

    def _cuda(self):
        return self.device is not None and self.device.type == "cuda"

    def timed_call(self, fn):
        self.sync()
        tick = time.perf_counter()
        out = fn()
        self.sync()
        return time.perf_counter()-tick, out

    def item(self, name, fn, *, group, note="", repeats=None, warmup=None, inspect=None):
        """fn() -> value; inspect(value) -> dict of per-call facts (e.g. the comparison path), kept per repeat."""
        if not self.wanted(name):
            return None
        record = dict(group=group, note=note, error=None, skipped=None)
        if self.over_budget():
            record["skipped"] = "worker budget used up (--max-worker-seconds)"
            self.items[name] = record
            return record
        repeats = self.repeats if repeats is None else repeats
        warmup = self.warmup if warmup is None else warmup
        self.logger.info("%s: %s (%d warmup, %d repeats)", self.path, name, warmup, repeats)
        import_torch = self._cuda()
        try:
            if import_torch:
                import torch
                torch.cuda.reset_peak_memory_stats(self.device)
                base_alloc = torch.cuda.memory_allocated(self.device)
            self.host_peak.reset()
            warm, samples, facts = [], [], []
            for _ in range(warmup):
                seconds, value = self.timed_call(fn)
                warm.append(seconds)
                del value
            before = self._transfer_snapshot()
            for _ in range(repeats):
                seconds, value = self.timed_call(fn)
                samples.append(seconds)
                if inspect is not None:
                    facts.append(inspect(value))
                del value
            after = self._transfer_snapshot()
            record.update(summarize(samples) or {}, samples=samples, warmup_samples=warm,
                          host_peak_mb=self.host_peak.peak_mb())
            if facts:
                record["per_call"] = facts
            if before is not None and repeats:
                d = [(b-a)/repeats for a, b in zip(before, after)]
                record["transfers_per_call"] = dict(upload_calls=d[0], upload_bytes=d[1], download_calls=d[2],
                                                    download_bytes=d[3],
                                                    counted_by="bayesreg.backends.torch64.Device (explicit "
                                                               "upload/download only)")
            if import_torch:
                record["device_peak_allocated_mib"] = torch.cuda.max_memory_allocated(self.device)/2**20
                record["device_peak_reserved_mib"] = torch.cuda.max_memory_reserved(self.device)/2**20
                record["device_allocated_before_mib"] = base_alloc/2**20
                if self.sync_count:
                    record["sync_events"] = self._count_syncs(fn)
                if self.torch_profiler:
                    record["profiler"] = self._profile(fn)
        except Exception as exc:                         # recorded; the next item still runs (nothing retried)
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=6)
            self.logger.error("%s: %s failed: %s", self.path, name, record["error"])
        self.items[name] = record
        return record

    def _count_syncs(self, fn):
        import torch
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    value = fn()
                    del value
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            n = sum(1 for w in caught if "synchroniz" in str(w.message).lower())
            return dict(count=n, method='torch.cuda.set_sync_debug_mode("warn"), one extra call (experimental: not '
                                         'every synchronizing operation warns)')
        except Exception as exc:
            return dict(count=None, error=f"{type(exc).__name__}: {exc}")

    def _profile(self, fn):
        import torch
        try:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                value = fn()
                del value
                self.sync()
            rows = []
            launches = memcpy = 0
            for event in prof.key_averages():
                device_us = getattr(event, "self_device_time_total", None)
                if device_us is None:
                    device_us = getattr(event, "self_cuda_time_total", 0.)
                if "LaunchKernel" in event.key:
                    launches += event.count
                if "Memcpy" in event.key or "memcpy" in event.key:
                    memcpy += event.count
                rows.append((float(device_us or 0.), event.key, int(event.count)))
            rows.sort(reverse=True)
            return dict(kernel_launches=launches, memcpy_events=memcpy, n_event_kinds=len(rows),
                        top_device_us=[dict(name=k, count=c, self_device_us=t) for t, k, c in rows[:12]])
        except Exception as exc:
            return dict(error=f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------------------------
# Instrumented backend and metric for the iteration item (values untouched)
# ----------------------------------------------------------------------------------------------

class CallClock:
    """Records (name, start, end) of every proxied call; `sync` runs before the clock starts and before it stops."""

    def __init__(self, sync):
        self.sync, self.events = sync, []

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            self.sync()
            tick = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.sync()
                self.events.append((name, tick, time.perf_counter()))
        return timed


class TimedBackend:
    """Proxy of a solver backend: the TIMED methods go through the clock, everything else is the backend's own
    attribute (a missing attribute stays missing, so getattr(proxy, "stable_compare", None) is None for numpy)."""
    TIMED = ("evaluate", "gradient", "stable_compare", "stable_reduction", "confirm_gradient_norm", "certify",
             "c_dat", "build_metric")

    def __init__(self, backend, clock):
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_clock", clock)

    def __getattr__(self, name):
        value = getattr(self._backend, name)
        if name in self.TIMED and callable(value):
            return self._clock.wrap(name, value)
        return value


class TimedJDScaledMetric(JDScaledMetric):
    """JDScaledMetric whose solve (the whole S~ M^-1 S~ g, host scaling included) and refresh are clocked; run_pg's
    isinstance check and refresh rule see an ordinary JDScaledMetric."""

    def __init__(self, metric, grid, u, *, clock, **kwargs):
        self._clock = clock
        super().__init__(metric, grid, u, **kwargs)

    def solve(self, g):
        return self._clock.wrap("metric_solve", super().solve)(g)

    def refresh(self, u, **context):
        return self._clock.wrap("jd_refresh", super().refresh)(u, **context)


class TimedMetric:
    """The plain metric (metric_jd_scaling "off") with a clocked solve."""

    def __init__(self, metric, clock):
        self.base, self._clock = metric, clock

    def __getattr__(self, name):
        return getattr(self.base, name)

    def solve(self, g):
        return self._clock.wrap("metric_solve", self.base.solve)(g)


def segment_iterations(t_start, rows, events):
    """Per accepted iteration: wall clock from the previous accept row (or t_start) to its accept row, candidates and
    reasons in the window, seconds per proxied call name, and host_other = wall - sum of the proxied calls."""
    out = []
    lo = t_start
    candidates = []
    for row in rows:
        if row["kind"] == "candidate":
            candidates.append(row)
        elif row["kind"] == "accept":
            hi = row["t"]
            components = {name: 0. for name in ITERATION_COMPONENTS}
            calls = Counter()
            for name, a, b in events:
                if a >= lo and b <= hi:
                    components[name] = components.get(name, 0.)+(b-a)
                    calls[name] += 1
            wall = hi-lo
            out.append(dict(index=row["iteration"], wall=wall, candidates=len(candidates),
                            reasons=dict(Counter(str(c["reason"]) for c in candidates)),
                            rule=candidates[0].get("rule") if candidates else None, tau=row.get("tau"),
                            components=components, calls=dict(calls), host_other=wall-sum(components.values())))
            lo, candidates = hi, []
    return out


def summarize_iterations(runs):
    """Iteration 1 (no BB2 history) and iterations >= 2 (BB2) separately: wall, candidates, component medians and
    shares of the median wall."""
    groups = dict(first=[], bb2=[])
    for run in runs:
        for it in run:
            groups["first" if it["index"] == 1 else "bb2"].append(it)
    out = {}
    for key, rows in groups.items():
        if not rows:
            continue
        wall = summarize([r["wall"] for r in rows])
        comps = {}
        for name in ITERATION_COMPONENTS+("host_other",):
            values = [r["components"].get(name, 0.) if name != "host_other" else r["host_other"] for r in rows]
            s = summarize(values)
            if s["max"] == 0.:
                continue
            s["share_of_median_wall"] = s["median"]/wall["median"] if wall["median"] > 0 else None
            comps[name] = s
        out[key] = dict(wall=wall, candidates=summarize([r["candidates"] for r in rows]), components=comps)
    return out


# ----------------------------------------------------------------------------------------------
# Block timers (standalone: the geometry the block needs + its residual and square sums)
# ----------------------------------------------------------------------------------------------

def _numpy_block_tile(state, u, which, tile):
    if which in BLOCKS_GAUSS:
        geo = tile_geometry(u, state.ops, tile=tile)
        if which == "geometry_gauss":
            return np.sum(geo.J, dtype=LD)
        block = dict(diffusion=Diffusion, logvol_gauss=LogVolume, curl=Curl)[which]
    else:
        geo = tile_geometry(u, state.corner_ops, tile=tile)
        if which == "geometry_corners":
            return np.sum(geo.J, dtype=LD)
        block = LogVolumeCorners
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        r = block.residual(block.scale(state, tile), geo)
        return np.sum(r*r, dtype=LD)


def numpy_block(state, u, which):
    total = LD(0)
    with closing(ordered_map(partial(_numpy_block_tile, state, u, which), cell_tiles(state.grid, state.tile_size),
                             state.cpu_threads)) as tiles:
        for value in tiles:
            total += value
    return total


def numpy_data(state, u):
    point = numpy_channel_point(state, u)
    with np.errstate(over="raise", invalid="raise"):
        return [_node_sum(state, point.r[c]*point.r[c]) for c in range(state.n_channels)]


def torch_block(dstate, u_cm, which):
    import torch
    from bayesreg.backends.torch64.model import (_corner_geometry, _corner_scale, _geometry, _logvol_scale,
                                                  _tile_reduce)
    from bayesreg.backends.torch64.q1 import curl_of
    q1 = dstate.q1
    n_cells = dstate.grid.n_cells
    cell_sums = torch.empty(n_cells, dtype=torch.float64, device=u_cm.device)
    sD, sC = dstate.scale_diffusion[None, None, None, :], dstate.scale_curl
    for chunk in q1.chunks:
        lo, hi = q1.cell_range(chunk)
        if which in BLOCKS_GAUSS:
            G, _, _, J = _geometry(dstate, u_cm, chunk)
            if which == "geometry_gauss":
                cell_sums[lo:hi] = J.sum(dim=1)
            elif which == "diffusion":
                rD = sD*G
                cell_sums[lo:hi] = (rD*rD).sum(dim=(0, 2)).sum(dim=1)
            elif which == "logvol_gauss":
                rJ = _logvol_scale(dstate, chunk)*torch.log(J)
                cell_sums[lo:hi] = (rJ*rJ).sum(dim=1)
            else:
                curl = 0.
                for component in curl_of(G):
                    rC = sC*component
                    curl = curl+rC*rC
                cell_sums[lo:hi] = curl.sum(dim=1)
        else:
            _, _, _, Jc = _corner_geometry(dstate, u_cm, chunk)
            if which == "geometry_corners":
                cell_sums[lo:hi] = Jc.sum(dim=1)
            else:
                rJc = _corner_scale(dstate, chunk)*torch.log(Jc)
                cell_sums[lo:hi] = (rJc*rJc).sum(dim=1)
    return dstate.device.download(_tile_reduce(cell_sums, dstate.state.tile_size, dstate.n_tiles, n_cells))


# ----------------------------------------------------------------------------------------------
# Host worker: the fixed trial and the path-independent host items
# ----------------------------------------------------------------------------------------------

# DEF (2026-10-31): the JDScalingConfig default became jd_cell_estimate "certificate", which this profiler does not wire
# (it builds S~ without a CertOutcome).  SPD Stage A (runs/.., commit 7c5d97c) profiled the S~ of the "gauss27+corners"
# estimate, the default then; the explicit old value keeps the tool's behaviour bit for bit.
SPD_JD = JDScalingConfig(cell_estimate="gauss27+corners")


def pg_config(record):
    if record["metric_jd_scaling"] == "off":
        return PGConfig(metric_jd_scaling="off")          # DEF: no JDScalingConfig under "off"
    return PGConfig(metric_jd_scaling=record["metric_jd_scaling"], jd=SPD_JD)


def frozen_metric(backend, state, ev, record):
    """M of the native round (recorded c_dat, b_P; recomputed when the record lacks them)."""
    native = record["native"]
    c = native.get("c_dat")
    b = native.get("b_P")
    if c is None:
        c = backend.c_dat(state, ev)
    if b is None:
        b = backend.metric_bP(state, mode="principal")
    return backend.build_metric(state, c_dat=float(c), b_P=float(b)), float(c), float(b)


def pg_metric_of(metric, state, u, record, clock=None):
    pg = pg_config(record)
    if pg.metric_jd_scaling == "off":
        return metric if clock is None else TimedMetric(metric, clock)
    if clock is None:
        return JDScaledMetric(metric, state.grid, u, config=pg.jd, at="profile")
    return TimedJDScaledMetric(metric, state.grid, u, clock=clock, config=pg.jd, at="profile")


def make_trial(state, u_b, record, out, logger):
    """The fixed trial of every path (module docstring), with the numpy CPU reference; writes out/trial.npz."""
    backend = make_backend("numpy")
    tick = time.perf_counter()
    ev_b = backend.evaluate(state, u_b)
    if not hasattr(ev_b, "energy"):
        raise SystemExit(f"base state: F_B not defined ({ev_b.reason})")
    g = backend.gradient(state, ev_b)
    metric, c, b = frozen_metric(backend, state, ev_b, record)
    pg = pg_config(record)
    direction_metric = pg_metric_of(metric, state, u_b, record)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p = -direction_metric.solve(g)
        slope = float(g@p)
    theta = restrict_free(u_b, state.grid)
    bound = sampling_step_bound(state.grid, u_b, p)
    log = record.get("log") or {}
    tau_rec = (log.get("accepted_tau") or {}).get("median")
    tau = min(tau_rec if tau_rec else pg.initial_step, pg.domain_fraction*bound)
    attempts, rejected_tau, accepted = [], None, None
    for _ in range(pg.max_backtracks+1):
        u_t = inject_free(theta+tau*p, state.grid)
        cert = backend.certify(u_t, state.grid)
        row = dict(tau=tau, cert=cert.status, reason=cert.cert_reason, seconds=cert.seconds)
        if cert.passed:
            ev_t = backend.evaluate(state, u_t)
            row["energy"] = getattr(ev_t, "energy", None)
            if hasattr(ev_t, "energy"):
                attempts.append(row)
                accepted = (tau, u_t, ev_t)
                break
        elif rejected_tau is None:
            rejected_tau = tau
        attempts.append(row)
        tau /= 2
    if accepted is None:
        raise SystemExit("trial: no certified candidate with defined F_B in 26 halvings")
    tau, u_t, ev_t = accepted
    if rejected_tau is None:
        probe = tau
        for _ in range(TAU_MAX_DOUBLINGS):
            probe *= 2
            if probe >= bound:
                break
            cert = backend.certify(inject_free(theta+probe*p, state.grid), state.grid)
            attempts.append(dict(tau=probe, cert=cert.status, reason=cert.cert_reason, seconds=cert.seconds,
                                 probe="doubling for the rejected-candidate timing"))
            if not cert.passed:
                rejected_tau = probe
                break
    required = pg.c1*tau*abs(slope)
    reduction = backend.stable_reduction(state, ev_b, ev_t)
    path = Path(out)/"trial.npz"
    np.savez(path, p=p, tau=tau, rejected_tau=np.nan if rejected_tau is None else rejected_tau, slope=slope,
             required=required, base_hash=state_hash(u_b), trial_hash=state_hash(u_t))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = dict(file=str(path), sha256=digest, tau=tau, rejected_tau=rejected_tau, slope=slope, required=required,
                tau_start_source="median accepted native tau of log.jsonl" if tau_rec else "initial step",
                domain_bound=bound, attempts=attempts, base_energy=ev_b.energy, trial_energy=ev_t.energy,
                cpu_ared=reduction.ared, cpu_guard=reduction.guard, c_dat=c, b_P=b,
                gf_b_base=gf_b(g, state.grid.omega, state.omega0), base_hash=state_hash(u_b),
                trial_hash=state_hash(u_t), seconds=time.perf_counter()-tick)
    logger.info("trial: tau %.6g, rejected tau %s, ared %.6g", tau, rejected_tau, reduction.ared)
    return info, dict(g=g, p=p, theta=theta, u_t=u_t, direction_metric=direction_metric, rejected_tau=rejected_tau,
                      tau=tau)


def load_trial(out, state, u_b):
    path = Path(out)/"trial.npz"
    if not path.is_file():
        raise SystemExit("trial.npz missing: the host worker builds it (run through the driver)")
    with np.load(path) as data:
        p = np.array(data["p"])
        tau = float(data["tau"])
        required = float(data["required"])
        trial_hash = str(data["trial_hash"])
    theta = restrict_free(u_b, state.grid)
    u_t = inject_free(theta+tau*p, state.grid)
    if state_hash(u_t) != trial_hash:
        raise SystemExit("trial: the rebuilt trial state does not match trial.npz")
    u_t.flags.writeable = False
    return u_t, tau, required, p


def certificate_identity(outcome):
    record = outcome.record()
    record.pop("seconds", None)
    return json.dumps(record, sort_keys=True, default=float)


def run_host(args, rec, state, u_b, record, out, logger):
    info, t = make_trial(state, u_b, record, out, logger)
    grid = state.grid
    backend = make_backend("numpy")
    u_t = t["u_t"]
    import numba
    threads = dict(numba_config_threads=int(numba.config.NUMBA_NUM_THREADS), numba_threads=int(numba.get_num_threads()),
                   fallback_workers=int(backend.cert_config.fallback_workers), cpu_count=os.cpu_count())
    note = (f"CPU FP64 certify_map (C07-2); numba threads {threads['numba_threads']}, fallback_workers "
            f"{threads['fallback_workers']}")
    rec.item("certificate.pass", lambda: backend.certify(u_t, grid), group="certificate", note=note,
             inspect=lambda o: dict(status=o.status, seconds_inner=o.seconds))
    rec.item("certificate.pass_base", lambda: backend.certify(u_b, grid), group="certificate", note=note,
             inspect=lambda o: dict(status=o.status))
    if t["rejected_tau"] is not None:
        u_r = inject_free(t["theta"]+t["rejected_tau"]*t["p"], grid)
        rec.item("certificate.reject", lambda: backend.certify(u_r, grid), group="certificate",
                 note=note+f"; rejected candidate tau {t['rejected_tau']:.6g}",
                 inspect=lambda o: dict(status=o.status, reason=o.cert_reason))
    sweep = []
    if args.cert_thread_sweep:
        reference = certificate_identity(backend.certify(u_t, grid))
        original = numba.get_num_threads()
        try:
            for n in args.cert_thread_sweep:
                if n > numba.config.NUMBA_NUM_THREADS:
                    sweep.append(dict(threads=n, skipped=f"> NUMBA_NUM_THREADS {numba.config.NUMBA_NUM_THREADS}"))
                    continue
                numba.set_num_threads(n)
                same = []
                rec.item(f"certificate.pass@threads={n}", lambda: backend.certify(u_t, grid), group="certificate",
                         note=f"numba.set_num_threads({n})",
                         inspect=lambda o: same.append(certificate_identity(o) == reference) or None)
                sweep.append(dict(threads=n, identical_outcome=bool(same) and all(same)))
        finally:
            numba.set_num_threads(original)
    g, p, theta = t["g"], t["p"], t["theta"]
    tau = t["tau"]
    pg = pg_config(record)
    jd = SPD_JD                    # the S~ item is timed whatever the run's metric mode (before DEF: PGConfig().jd)
    rec.item("jd.scaling_factors", lambda: scaling_factors(u_b, grid, activation=jd.activation, floor=jd.floor),
             group="jd", note="C09.1-3a S~ estimate (host numpy; every path computes S~ on the host)")
    scale = getattr(t["direction_metric"], "scale", None)
    if scale is not None:
        rec.item("jd.apply_host", lambda: scale*(scale*g), group="jd", note="the two host products S~ (S~ g)")
    s = tau*p
    y = .37*s
    previous = (theta-s, g-y, p+y, tau)
    rec.item("pg.bb2_proposal", lambda: bb2_proposal(pg, theta, g, p, previous), group="host",
             note="C09.2-5 with a synthetic previous step (s.y > 0: the bb2 branch)",
             inspect=lambda v: dict(rule=v[1]))
    rec.item("pg.domain_bound", lambda: sampling_step_bound(grid, u_b, p), group="host", note="C09.2-6")
    rec.item("pg.slope", lambda: float(g@p), group="host", note="g.p")
    rec.item("pg.candidate", lambda: theta+tau*p, group="host", note="theta + tau p (per candidate)")
    candidate = theta+tau*p
    rec.item("pg.isfinite", lambda: bool(np.all(np.isfinite(candidate))), group="host", note="per candidate")
    rec.item("pg.inject_free", lambda: inject_free(candidate, grid), group="host", note="per candidate")
    rec.item("pg.readonly_copy", lambda: readonly_copy(candidate), group="host", note="per accepted step")
    rec.item("pg.step_ratio", lambda: step_ratio(candidate-theta, theta, grid), group="host",
             note="per accepted step")
    rec.item("pg.gf_b", lambda: gf_b(g, grid.omega, state.omega0), group="host", note="per gradient")
    rec.item("pg.state_hash", lambda: state_hash(u_t), group="host", note="sha256 of u (certify_map hashes u too)")
    rec.item("pg.node_field_copy",
             lambda: np.array(node_field(u_t, grid), dtype=np.float64, copy=True, order="C"), group="host",
             note="the host copy of Torch64Backend.evaluate")
    return dict(trial=info, certificate_threads=threads, cert_thread_sweep=sweep)


# ----------------------------------------------------------------------------------------------
# Backend workers
# ----------------------------------------------------------------------------------------------

def make_sync(device):
    if device is not None and device.type == "cuda":
        import torch

        def sync():
            torch.cuda.synchronize(device)
        return sync, f"torch.cuda.synchronize({device}) before the clock starts and before it stops"
    return (lambda: None), "none (CPU: every call returns when its work is done)"


def run_iteration(args, rec, path, backend, state, u_b, record, sync, logger, *, warmup):
    """run_pg from u_b through the timing proxies (module docstring); returns the iteration record."""
    if not rec.wanted("iteration") or args.iteration_steps < 1:
        return None
    if rec.over_budget():
        return dict(skipped="worker budget used up (--max-worker-seconds)")
    pg = pg_config(record)
    grid = state.grid
    theta = restrict_free(u_b, grid)
    cert = backend.certify(u_b, grid)                       # untimed: compiles numba in this process
    base = backend.evaluate(state, u_b)
    gradient = backend.gradient(state, base)
    metric, _, _ = frozen_metric(backend, state, base, record)
    target = record["native"].get("T_B") or 0.
    repeats = args.iteration_repeats if args.iteration_repeats is not None else args.repeats
    runs, warm_runs, errors = [], [], []
    cuda = rec.device is not None and rec.device.type == "cuda"
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats(rec.device)
    host_peak = HostPeak()
    host_peak.reset()
    for k in range(warmup+repeats):
        clock = CallClock(sync)
        proxy = TimedBackend(backend, clock)
        pg_metric = pg_metric_of(metric, state, u_b, record, clock=clock)
        rows = []

        def log(row):
            rows.append(dict(t=time.perf_counter(), kind=row.get("kind"), iteration=row.get("iteration"),
                             reason=row.get("reason"), rule=row.get("rule"), tau=row.get("tau")))
        objective = DirectObjective(proxy, state)
        logger.info("%s: iteration run %d/%d", path, k+1, warmup+repeats)
        sync()
        t0 = time.perf_counter()
        result = None
        try:
            result = run_pg(objective, base, theta=theta, cert=cert, metric=pg_metric, certify=proxy.certify,
                            target=target, max_steps=args.iteration_steps, config=pg, log=log, gradient=gradient)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        sync()
        total = time.perf_counter()-t0
        if result is None:
            status, accepted, final_hash = "exception", None, None
        else:
            status, accepted, final_hash = result.status, result.accepted, state_hash(result.point.u)
        del result
        entry = dict(status=status, accepted=accepted, final_state_hash=final_hash, total=total,
                     iterations=segment_iterations(t0, rows, clock.events))
        (warm_runs if k < warmup else runs).append(entry)
    return dict(steps=args.iteration_steps, repeats=repeats, warmup=warmup, target=target,
                runs=runs, warmup_runs=warm_runs, errors=errors, host_peak_mb=host_peak.peak_mb(),
                device_peak_allocated_mib=torch.cuda.max_memory_allocated(rec.device)/2**20 if cuda else None,
                summary=summarize_iterations([r["iterations"] for r in runs]),
                note="iteration 1 starts from the fixed state without BB2 history; iterations >= 2 use BB2")


def run_backend(args, rec, path, state, u_b, record, out, logger):
    u_t, tau, required, p = load_trial(out, state, u_b)
    torch_path = path != "numpy"
    setup = {}
    tick = time.perf_counter()
    if torch_path:
        backend = make_backend("torch64", device="cuda" if path == "cuda" else "cpu")
        device = backend.dev.device
        rec.transfers = backend.dev.transfers
    else:
        backend = make_backend("numpy")
        device = None
    sync, sync_text = make_sync(device)
    rec.sync, rec.device = sync, device
    setup["backend_seconds"] = time.perf_counter()-tick
    tick = time.perf_counter()
    backend.prepare(state)
    sync()
    setup["prepare_seconds"] = time.perf_counter()-tick
    if torch_path:
        t = backend.dev.transfers
        setup["prepare_transfers"] = dict(upload_calls=t.upload_calls, upload_bytes=t.upload_bytes)
    grid = state.grid
    ev_b = backend.evaluate(state, u_b)
    ev_t = backend.evaluate(state, u_t)
    for name, ev in (("base", ev_b), ("trial", ev_t)):
        if not hasattr(ev, "energy"):
            raise SystemExit(f"{path}: F_B at the {name} state not defined ({ev.reason})")
    g_b = backend.gradient(state, ev_b)
    sync()

    # ---- energy, by block ----
    if torch_path:
        T, S = backend._T, backend._S
        dstate = backend._lookup(state)[0]
        dev = backend.dev
        u_dev = dev.upload(u_t)
        u_cm = T.node_field_cm(u_dev, dstate)
        from bayesreg.backends.torch64.model import _channel_point

        def data_device():
            return _channel_point(dstate, u_cm)

        def data_all():
            point = _channel_point(dstate, u_cm)
            r = dev.download(point.r)
            with np.errstate(over="raise", invalid="raise"):
                return [_node_sum(state, r[c]*r[c]) for c in range(state.n_channels)]
        rec.item("energy.data", data_all, group="energy",
                 note="domain check + spline sampling + residual on the device, download r, host longdouble sum")
        rec.item("energy.data_device", data_device, group="energy", note="the device part of energy.data")
        block = lambda which: (lambda: torch_block(dstate, u_cm, which))   # noqa: E731
    else:
        rec.item("energy.data", lambda: numpy_data(state, u_t), group="energy",
                 note="channel_point (domain check, spline sampling, residual) + longdouble node sums")
        block = lambda which: (lambda: numpy_block(state, u_t, which))     # noqa: E731
    for which in BLOCKS_GAUSS:
        rec.item(f"energy.{which}", block(which), group="energy",
                 note="Gauss geometry" + ("" if which == "geometry_gauss" else f" + {which} residual and sums"))
    if state.logvol_corners:
        for which in BLOCKS_CORNERS:
            rec.item(f"energy.{which}", block(which), group="energy",
                     note="corner geometry" + ("" if which == "geometry_corners" else " + corner log-volume residual"))
    if torch_path:
        rec.item("kq.apply", lambda: T.stiffness_apply(u_cm, grid), group="energy",
                 note="K_Q u (stiffness_apply; the diffusion term of the gradient)")
        rec.item("energy.evaluate_device", lambda: T.evaluate(dstate, u_dev), group="energy",
                 note="TorchEvaluation from an uploaded u (no host copy, no upload)")
    else:
        rec.item("kq.apply", lambda: numpy_stiffness_apply(u_t, grid), group="energy",
                 note="K_Q u (stiffness_apply; the diffusion term of the gradient)")
    rec.item("energy.evaluate", lambda: backend.evaluate(state, u_t), group="energy",
             note="backend.evaluate as the solver calls it")

    # ---- gradient ----
    for blocks in (("channel",), ("diffusion",), ("logvol",), ("curl",)):
        rec.item(f"gradient.{blocks[0]}", lambda blocks=blocks: backend.gradient(state, ev_t, blocks=blocks),
                 group="gradient", note=f"backend.gradient(blocks={blocks})"
                 + (" (Gauss points and corners)" if blocks == ("logvol",) and state.logvol_corners else ""))
    rec.item("gradient.total", lambda: backend.gradient(state, ev_t), group="gradient",
             note="backend.gradient (all blocks; host free vector out)")
    if torch_path:
        rec.item("gradient.total_device", lambda: T.gradient(dstate, ev_t.device), group="gradient",
                 note="device gradient without the download")

    # ---- metric ----
    rec.item("metric.c_dat", lambda: backend.c_dat(state, ev_b), group="metric", note="C09.1 c_dat")
    metric, c, b = frozen_metric(backend, state, ev_b, record)
    rec.item("metric.build", lambda: backend.build_metric(state, c_dat=c, b_P=b), group="metric",
             note="build_metric (DST denominator on the host; torch: uploaded)")
    rec.item("metric.solve", lambda: metric.solve(g_b), group="metric",
             note="M^-1 g (DST-I)" + ("; host vector in and out (upload, device DST, download)" if torch_path else ""))
    if torch_path:
        g_dev = dev.upload(g_b)
        rec.item("metric.solve_device", lambda: metric.device_metric.solve(g_dev), group="metric",
                 note="M^-1 g on the device, no transfer")
    direction_metric = pg_metric_of(metric, state, u_b, record)
    rec.item("metric.jd_solve", lambda: direction_metric.solve(g_b), group="metric",
             note="the PG direction operator S~ M^-1 S~ g (S~ frozen; host scaling included)")

    # ---- comparison ----
    if torch_path:
        rec.item("compare.gpu", lambda: backend.stable_compare(state, ev_b, ev_t, required=required, energy="F_B",
                                                               context="pg"),
                 group="compare", note=f"C08-8 GPU path with the fallback rule (r = {required:.6g}); path per call",
                 inspect=lambda red: dict(path=red.path, fallback_reason=red.info.get("fallback_reason"),
                                          A_hat=red.ared, s_hat=red.guard))
        rec.item("compare.device_gauss_changes", lambda: S.device_gauss_changes(dstate, ev_b.device, ev_t.device),
                 group="compare", note="C08-8(a)(b): device elementwise changes + tile sums (two geometries per "
                                       "point set) + download")
        rec.item("compare.channel_host", lambda: [_node_change_sum(state, ev_b.device.r_host[ch],
                                                                   ev_t.device.r_host[ch])
                                                  for ch in range(state.n_channels)],
                 group="compare", note="C08-1 channel blocks on the host (longdouble)")
        A_hat = float(ev_b.energy-ev_t.energy)
        rec.item("compare.decision", lambda: S.fallback_rule(A_hat, S.gpu_guard(ev_b.energy, ev_t.energy), required),
                 group="compare", note="gpu_guard + fallback_rule (C08-8(c)(d))")
        if path == "cuda" and args.with_fallback_items:
            model_id = backend.model_identity(state)

            def cold():
                comparator = S.StableComparator(cache_size=2)
                return comparator._cpu_compare(state, ev_b, ev_t, model_id=model_id, energy="F_B", reference=None,
                                               extra=None, info={})
            warm_comparator = S.StableComparator(cache_size=2)
            warm_comparator.cpu_point(state, ev_b, model_id=model_id, base=True)

            def warm():
                warm_comparator.cache.clear()
                warm_comparator._reduction = None
                return warm_comparator._cpu_compare(state, ev_b, ev_t, model_id=model_id, energy="F_B",
                                                    reference=None, extra=None, info={})
            rec.item("compare.cpu_fallback_cold", cold, group="compare",
                     note="C08-8(d) CPU path, fresh comparator: both states on the CPU reference + C08-1..7")
            rec.item("compare.cpu_fallback_warm", warm, group="compare",
                     note="C08-8(d) CPU path, base from the base slot: trial on the CPU reference + C08-1..7")
    else:
        rec.item("compare.cpu_reference", lambda: backend.stable_reduction(state, ev_b, ev_t), group="compare",
                 note="C08-1..7 stable_reduction (CPU reference)")

    # ---- transfers ----
    if path == "cuda":
        rec.item("transfer.upload_u", lambda: dev.upload(u_t), group="transfer",
                 note=f"Device.upload of the node field ({u_t.nbytes/2**20:.1f} MiB)")
        rec.item("transfer.download_free", lambda: dev.download(g_dev), group="transfer",
                 note=f"Device.download of a free vector ({g_b.nbytes/2**20:.1f} MiB)")
    del ev_t
    iteration_warmup = args.iteration_warmup
    if iteration_warmup == "auto":
        iteration_warmup = 1 if path == "cuda" else 0
    iteration = run_iteration(args, rec, path, backend, state, u_b, record, sync, logger, warmup=int(iteration_warmup))
    extra = dict(setup=setup, sync=sync_text)
    if torch_path:
        extra["torch"] = torch_record(device)
        extra["comparator_summary"] = backend.comparison_summary()
        if path == "cuda":
            peaks = [v.get("device_peak_allocated_mib") for v in rec.items.values()]
            peaks = [v for v in peaks if v is not None]
            extra["device_peak_allocated_mib_max_over_items"] = max(peaks) if peaks else None
    return extra, iteration


def worker(args):
    out = Path(args.out)
    if not (out/"driver.json").is_file():
        raise SystemExit("--worker runs under the driver only (driver.json missing)")
    path = args.worker
    logger = logging.getLogger("spd")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    started = time.perf_counter()
    if args.numba_threads is not None:
        import numba
        if not 1 <= args.numba_threads <= numba.config.NUMBA_NUM_THREADS:
            raise SystemExit(f"--numba-threads {args.numba_threads}: 1..NUMBA_NUM_THREADS "
                             f"({numba.config.NUMBA_NUM_THREADS}) required")
        numba.set_num_threads(args.numba_threads)
    rec = Recorder(path, repeats=args.repeats, warmup=args.warmup, sync=lambda: None, items=args.items,
                   sync_count=not args.no_sync_count, torch_profiler=args.torch_profiler,
                   budget=args.max_worker_seconds, logger=logger)
    document = dict(schema=PATH_SCHEMA, path=path, inputs=None, pid=os.getpid(), env=env_record())
    try:
        record = load_run_record(args.run_dir)
        inputs = resolved_inputs(args, record)
        state, u_b, info = load_native(inputs, record, logger)
        document["inputs"] = info
        rec.started = time.perf_counter()                 # the worker budget counts the items, not the inputs
        if path == "host":
            document["host"] = run_host(args, rec, state, u_b, record, out, logger)
        else:
            extra, iteration = run_backend(args, rec, path, state, u_b, record, out, logger)
            document.update(extra)
            document["iteration"] = iteration
        document["status"] = "done"
    except SystemExit as exc:
        document["status"], document["error"] = "stopped", str(exc)
    except Exception as exc:
        document["status"], document["error"] = "exception", f"{type(exc).__name__}: {exc}"
        document["traceback"] = traceback.format_exc()
    document["items"] = rec.items
    document["elapsed_seconds"] = time.perf_counter()-started
    document["peak_rss_mb"] = lifetime_peak_rss_mb()
    document["budget"] = dict(max_worker_seconds=args.max_worker_seconds,
                              skipped=[k for k, v in rec.items.items() if v.get("skipped")])
    (out/f"path_{path}.json").write_text(json.dumps(document, indent=1, default=_json_default)+"\n", encoding="utf-8")
    return 0 if document["status"] == "done" else 1


def _json_default(value):
    if isinstance(value, (np.floating, np.longdouble)):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def median_of(paths, path, name):
    item = ((paths.get(path) or {}).get("items") or {}).get(name) or {}
    return item.get("median")


def model_rows(paths, record):
    """Item medians weighted by the run's native-level counts (a model, not a measurement)."""
    log = record.get("log") or {}
    native = record.get("native") or {}
    counters = native.get("counters") or {}
    if not log and not counters:
        return None
    n_cand = log.get("candidates", counters.get("candidates"))
    n_pass = log.get("cert_pass")
    n_reject = log.get("cert_reject", (counters.get("rejections") or {}).get("JACOBIAN_VIOLATION"))
    n_eval = log.get("evaluations", counters.get("energy_evaluations"))
    n_cmp = log.get("comparisons", n_eval)
    n_grad = counters.get("gradients", log.get("accepted"))
    n_solve = counters.get("metric_solves", log.get("accepted"))
    n_jd = native.get("jd_computations") or log.get("jd_computations") or 0
    n_iter = log.get("accepted", native.get("accepted"))
    n_fallback = log.get("fallbacks", 0)
    weights = dict(n_candidates=n_cand, n_cert_pass=n_pass, n_cert_reject=n_reject, n_evaluations=n_eval,
                   n_comparisons=n_cmp, n_gradients=n_grad, n_metric_solves=n_solve, n_jd_computations=n_jd,
                   n_accepted=n_iter, n_cpu_fallbacks=n_fallback)
    host = lambda name: median_of(paths, "host", name)    # noqa: E731
    per_candidate = sum(v for v in (host("pg.candidate"), host("pg.isfinite"), host("pg.inject_free")) if v)
    per_iteration = sum(v for v in (host("pg.bb2_proposal"), host("pg.domain_bound"), host("pg.slope"),
                                    host("pg.readonly_copy"), host("pg.step_ratio"), host("pg.gf_b")) if v)
    out = dict(weights=weights, observed_native_seconds=record.get("native_level_seconds"), paths={})
    for path in BACKEND_PATHS:
        if path not in paths or not (paths[path] or {}).get("items"):
            continue
        compare = "compare.cpu_reference" if path == "numpy" else "compare.gpu"
        rows = [("certificate (pass)", n_pass, host("certificate.pass")),
                ("certificate (reject)", n_reject, host("certificate.reject") or host("certificate.pass")),
                ("evaluate", n_eval, median_of(paths, path, "energy.evaluate")),
                ("stable comparison", n_cmp, median_of(paths, path, compare)),
                ("gradient", n_grad, median_of(paths, path, "gradient.total")),
                ("direction (S~ M^-1 S~ g)", n_solve, median_of(paths, path, "metric.jd_solve")),
                ("JD scaling S~", n_jd, host("jd.scaling_factors")),
                ("host per candidate", n_cand, per_candidate or None),
                ("host per iteration", n_iter, per_iteration or None)]
        if path == "cuda" and n_fallback:
            rows.append(("CPU fallback (warm)", n_fallback, median_of(paths, path, "compare.cpu_fallback_warm")))
        table = []
        for name, count, seconds in rows:
            total = None if count is None or seconds is None else count*seconds
            table.append(dict(component=name, count=count, median_seconds=seconds, total_seconds=total))
        known = sum(r["total_seconds"] for r in table if r["total_seconds"] is not None)
        for r in table:
            r["share"] = None if r["total_seconds"] is None or known <= 0 else r["total_seconds"]/known
        observed = record.get("native_level_seconds")
        out["paths"][path] = dict(rows=table, model_seconds=known,
                                  ratio_observed_over_model=None if not observed or known <= 0 else observed/known)
    return out


def bottleneck_statements(profile):
    """Measured facts only (shares of medians); no inference."""
    lines = []
    gate_start = profile.get("gate_start") or {}
    if gate_start.get("gpu_shared"):
        lines.append("GPU 在开始时被占用（" + "; ".join(gate_start.get("gpu_busy", [])) + "）：cuda 数字不作计时结论。")
    for path in BACKEND_PATHS:
        doc = profile["paths"].get(path) or {}
        summary = ((doc.get("iteration") or {}).get("summary") or {}).get("bb2")
        if not summary:
            continue
        wall = summary["wall"]
        parts = sorted(((v["median"], k, v.get("share_of_median_wall")) for k, v in summary["components"].items()),
                       reverse=True)
        text = "，".join(f"{k} {m:.3g} s（{(s or 0)*100:.0f}%）" for m, k, s in parts[:5])
        lines.append(f"{path}：BB2 阶段一次接受步 wall 中位数 {wall['median']:.3g} s（IQR {wall['iqr']:.3g}，n = "
                     f"{wall['n']}，候选数中位数 {summary['candidates']['median']:g}）；分项中位数（占 wall 中位数）："
                     f"{text}。")
    model = profile.get("model") or {}
    observed = model.get("observed_native_seconds")
    for path, entry in (model.get("paths") or {}).items():
        top = sorted((r for r in entry["rows"] if r["total_seconds"] is not None), key=lambda r: -r["total_seconds"])
        text = "，".join(f"{r['component']} {r['share']*100:.0f}%" for r in top[:3])
        tail = "" if not observed else f"；运行记录的原生层实测 {observed:.0f} s"
        lines.append(f"模型（{path}，分项中位数按运行的原生层计数加权，不是测量）：合计 {entry['model_seconds']:.0f} s{tail}；"
                     f"前三项 {text}。")
    log = (profile.get("record") or {}).get("log") or {}
    cmp_stats = log.get("comparison_seconds_gpu")
    cuda_cmp = median_of(profile["paths"], "cuda", "compare.gpu")
    if cmp_stats and cuda_cmp is not None:
        lines.append(f"同一比较的两次观测：运行记录（log.jsonl 原生层 {cmp_stats['n']} 次）seconds_gpu 中位数 "
                     f"{cmp_stats['median']:.3g} s；本次 compare.gpu 中位数 {cuda_cmp:.3g} s（固定状态，单次 profiler 运行）。")
    return lines


def num(value, spec, dash="—"):
    """format(value, spec), or the dash for None."""
    return dash if value is None else format(value, spec)


def fmt(item):
    if not item:
        return "—"
    if item.get("skipped"):
        return "skipped"
    if item.get("error"):
        return "error"
    if item.get("median") is None:
        return "—"
    return f"{num(item['median'], '.4g')} [{num(item['iqr'], '.2g')}]"


def render_md(profile):
    paths = profile["paths"]
    present = [p for p in BACKEND_PATHS if p in paths]
    inputs = profile.get("inputs") or {}
    config = profile["config"]
    lines = ["# SPD 原生层分项测速（profile_native.py）", ""]
    lines += [f"- 生成：{profile['created_utc']}（UTC）；代码 {profile['code']['head']}"
              + ("（工作区有改动）" if profile["code"]["dirty"] else ""),
              f"- 状态：{inputs.get('state_path')}（state_hash {str(inputs.get('state_hash'))[:16]}…），"
              f"对 {inputs.get('pair')}，网格 {inputs.get('grid_shape_zyx')}，{inputs.get('logvol_points')}，"
              f"cpu_threads {inputs.get('cpu_threads')}",
              f"- 每项：预热 {config['warmup']} 次（单独记录）后 {config['repeats']} 次，报中位数 [IQR]，单位 s；"
              "固定状态测速，不能换算成整次配准的倍速。",
              "- 主机峰值：清零 VmHWM 后该项期间的进程峰值 RSS（含常驻部分）。", ""]
    if "cuda" in present:
        lines[-1:-1] = ["- GPU 计时：每次调用前后 torch.cuda.synchronize(device)。"]
    gate_start = profile.get("gate_start") or {}
    if gate_start.get("gpu_shared"):
        lines += ["> **GPU 共用**：开始时 GPU 忙（" + "; ".join(gate_start.get("gpu_busy", []))
                  + "）。cuda 数字只作记录，不作计时结论。", ""]
    trial = profile.get("trial") or {}
    if trial:
        lines += [f"- 试探态：tau {num(trial.get('tau'), '.6g')}（起点：{trial.get('tau_start_source')}），被拒 tau "
                  f"{num(trial.get('rejected_tau'), '.6g')}，CPU ared {num(trial.get('cpu_ared'), '.6g')}，"
                  f"r = c1 tau |g.p| = {num(trial.get('required'), '.6g')}", ""]
    lines += ["## 分项（中位数 [IQR]，s）", "", "| 项 | " + " | ".join(present) + " | 说明 |",
              "|---|" + "---|"*len(present) + "---|"]
    names = []
    for p in present:
        for name in (paths[p].get("items") or {}):
            if name not in names:
                names.append(name)
    for name in names:
        note = next(((paths[p].get("items") or {}).get(name, {}).get("note") for p in present
                     if (paths[p].get("items") or {}).get(name)), "")
        lines.append(f"| {name} | " + " | ".join(fmt((paths[p].get("items") or {}).get(name)) for p in present)
                     + f" | {note} |")
    lines += ["", "增量（块 − 几何，中位数之差；单独计时之差，不是同一次调用内的拆分）：", ""]
    for p in present:
        items = paths[p].get("items") or {}
        geo = (items.get("energy.geometry_gauss") or {}).get("median")
        cgeo = (items.get("energy.geometry_corners") or {}).get("median")
        parts = []
        for which, base in (("diffusion", geo), ("logvol_gauss", geo), ("curl", geo), ("logvol_corners", cgeo)):
            m = (items.get(f"energy.{which}") or {}).get("median")
            if m is not None and base is not None:
                parts.append(f"{which} {m-base:.3g}")
        if parts:
            lines.append(f"- {p}：几何 {num(geo, '.3g')}，角点几何 {num(cgeo, '.3g')}；" + "，".join(parts))
    host = paths.get("host") or {}
    if host.get("items"):
        lines += ["", "## 主机项（与路径无关；host 进程）", "", "| 项 | 中位数 [IQR] | 说明 |", "|---|---|---|"]
        for name, item in host["items"].items():
            lines.append(f"| {name} | {fmt(item)} | {item.get('note', '')} |")
        threads = (host.get("host") or {}).get("certificate_threads")
        if threads:
            lines += ["", f"证书线程：{threads}"]
        sweep = (host.get("host") or {}).get("cert_thread_sweep")
        if sweep:
            lines.append(f"线程扫描：{sweep}")
    lines += ["", "## 资源", ""]
    cuda = paths.get("cuda") or {}
    if cuda.get("items"):
        lines += ["cuda 各项（一次调用的均值；同步次数为一次额外调用）：", "",
                  "| 项 | 显存峰值 MiB | 主机峰值 MB | 上传 次/MiB | 下载 次/MiB | 同步次数 |", "|---|---|---|---|---|---|"]
        for name, item in cuda["items"].items():
            t = item.get("transfers_per_call") or {}
            s = item.get("sync_events") or {}
            up = num(t.get("upload_calls"), "g") + "/" + num(None if not t else (t.get("upload_bytes") or 0)/2**20, ".1f")
            down = (num(t.get("download_calls"), "g") + "/"
                    + num(None if not t else (t.get("download_bytes") or 0)/2**20, ".1f"))
            lines.append(f"| {name} | {num(item.get('device_peak_allocated_mib'), '.0f')} | "
                         f"{num(item.get('host_peak_mb'), '.0f')} | {up} | {down} | {num(s.get('count'), 'g')} |")
    else:
        lines.append("（未跑 cuda 路径）")
    lines.append("")
    for p in ["host"]+present:
        doc = paths.get(p) or {}
        if not doc:
            continue
        peaks = [(v.get("host_peak_mb"), k) for k, v in (doc.get("items") or {}).items() if v.get("host_peak_mb")]
        top = max(peaks) if peaks else None
        lines.append(f"- {p}：进程峰值 RSS {num(doc.get('peak_rss_mb'), '.0f')} MB"
                     + ("" if top is None else f"（单项最高 {top[0]:.0f} MB：{top[1]}）")
                     + (f"，显存单项最高 {num(doc.get('device_peak_allocated_mib_max_over_items'), '.0f')} MiB"
                        if p == "cuda" else "")
                     + f"，用时 {num(doc.get('elapsed_seconds'), '.0f')} s，状态 {doc.get('status')}"
                     + (f"，错误 {doc.get('error')}" if doc.get("error") else ""))
        it = doc.get("iteration") or {}
        if it.get("host_peak_mb") is not None or it.get("device_peak_allocated_mib") is not None:
            lines.append(f"  - iteration：主机峰值 {num(it.get('host_peak_mb'), '.0f')} MB，显存峰值 "
                         f"{num(it.get('device_peak_allocated_mib'), '.0f')} MiB")
    lines += ["", "## 一整次迭代（run_pg，计时代理）", ""]
    for p in present:
        summary = (paths[p].get("iteration") or {}).get("summary") or {}
        for key, title in (("first", "迭代 1（无 BB2 历史）"), ("bb2", "迭代 ≥ 2（BB2）")):
            s = summary.get(key)
            if not s:
                continue
            lines.append(f"- {p} {title}：wall 中位数 {num(s['wall']['median'], '.4g')} s [IQR "
                         f"{num(s['wall']['iqr'], '.2g')}]，n = {s['wall']['n']}，候选数中位数 "
                         f"{num(s['candidates']['median'], 'g')}")
            for name, v in sorted(s["components"].items(), key=lambda kv: -kv[1]["median"]):
                share = v.get("share_of_median_wall")
                lines.append(f"  - {name}：{num(v['median'], '.4g')} s（{num(None if share is None else 100*share, '.1f')}%）")
    model = profile.get("model")
    if model:
        lines += ["", "## 模型：分项中位数 × 运行的原生层计数（不是测量）", "",
                  f"权重：{model['weights']}；运行记录的原生层实测 {num(model.get('observed_native_seconds'), '.1f')} s", ""]
        for p, entry in model["paths"].items():
            lines += [f"### {p}（模型合计 {num(entry['model_seconds'], '.1f')} s；实测/模型 = "
                      f"{num(entry['ratio_observed_over_model'], '.3g')}）", "",
                      "| 成分 | 次数 | 中位数 s | 合计 s | 占比 |", "|---|---|---|---|---|"]
            for r in entry["rows"]:
                lines.append(f"| {r['component']} | {num(r['count'], 'g')} | {num(r['median_seconds'], '.4g')} | "
                             f"{num(r['total_seconds'], '.1f')} | "
                             f"{num(None if r['share'] is None else 100*r['share'], '.1f')}% |")
            lines.append("")
    lines += ["## 主要瓶颈（只复述测量）", ""]
    lines += [f"- {s}" for s in profile.get("bottleneck", [])] or ["- （无可用测量）"]
    lines += ["", "解释与建议由总负责人写进 result.md；推断须标〔推断〕。", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--allowlist", default=None)
    parser.add_argument("--tissue-root", default=None)
    parser.add_argument("--paths", default="cuda,torch-cpu,numpy")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--items", default=None, help="comma-separated item name prefixes (default: all)")
    parser.add_argument("--iteration-steps", type=int, default=3)
    parser.add_argument("--iteration-repeats", type=int, default=None)
    parser.add_argument("--iteration-warmup", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--numba-threads", type=int, default=None)
    parser.add_argument("--cert-thread-sweep", default=None)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--with-fallback-items", action="store_true")
    parser.add_argument("--no-sync-count", action="store_true")
    parser.add_argument("--max-worker-seconds", type=float, default=None)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--gpu-busy-util", type=float, default=GPU_BUSY_UTIL)
    parser.add_argument("--gpu-busy-mib", type=float, default=GPU_BUSY_MIB)
    parser.add_argument("--allow-busy-units", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--no-gpu-query", action="store_true")
    parser.add_argument("--keep-trial", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.paths = [p for p in args.paths.split(",") if p]
    bad = [p for p in args.paths if p not in BACKEND_PATHS]
    if bad:
        parser.error(f"--paths: unknown {bad} (choose from {BACKEND_PATHS}; host always runs first)")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats >= 1 and --warmup >= 0 required")
    args.items = [i for i in args.items.split(",") if i] if args.items else None
    args.cert_thread_sweep = [int(n) for n in args.cert_thread_sweep.split(",")] if args.cert_thread_sweep else None
    if args.iteration_warmup != "auto":
        args.iteration_warmup = int(args.iteration_warmup)
    return args


def driver(args, argv):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty (nothing is overwritten)")
    code = git_record()
    if code["dirty"] and not args.allow_dirty:
        raise SystemExit("the code tree has tracked changes (commit first, or --allow-dirty for development)")
    start_gate = gate(args, when="start")
    if start_gate["stops"]:
        raise SystemExit("STOP (gate): " + " | ".join(start_gate["stops"]))
    out.mkdir(parents=True, exist_ok=True)
    record = load_run_record(args.run_dir)
    inputs = resolved_inputs(args, record)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out/"cmd.txt").write_text(" ".join([sys.executable, "-B", str(Path(__file__).resolve()), *argv])
                               + f"\n# cwd {os.getcwd()}\n", encoding="utf-8")
    (out/"git.txt").write_text(f"HEAD {code['head']}\ndirty {code['dirty']}\n\n# git status --porcelain\n"
                               f"{code['status']}\n# git diff HEAD\n{code['diff']}", encoding="utf-8")
    (out/"env.txt").write_text(json.dumps(env_record(), indent=1)+"\n", encoding="utf-8")
    (out/"data.txt").write_text(json.dumps(dict(inputs=inputs, run_dir=str(args.run_dir)), indent=1)+"\n",
                                encoding="utf-8")
    config = dict(paths=["host"]+args.paths, repeats=args.repeats, warmup=args.warmup, items=args.items,
                  iteration_steps=args.iteration_steps, iteration_repeats=args.iteration_repeats or args.repeats,
                  iteration_warmup=args.iteration_warmup, cert_thread_sweep=args.cert_thread_sweep,
                  numba_threads=args.numba_threads,
                  torch_profiler=args.torch_profiler, sync_count=not args.no_sync_count,
                  with_fallback_items=args.with_fallback_items,
                  max_worker_seconds=args.max_worker_seconds, cpu_threads=inputs["cpu_threads"])
    (out/"config.json").write_text(json.dumps(config, indent=1)+"\n", encoding="utf-8")
    (out/"driver.json").write_text(json.dumps(dict(created_utc=created, pid=os.getpid()))+"\n", encoding="utf-8")
    paths, workers = {}, []
    for path in ["host"]+args.paths:
        if path != "host" and (paths.get("host") or {}).get("status") != "done":
            paths[path] = dict(status="not_run", error="the host worker did not finish (no trial state)")
            continue
        env = dict(os.environ)
        if path != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ""
        command = [sys.executable, "-B", str(Path(__file__).resolve()), *argv, "--worker", path]
        tick = time.time()
        with open(out/f"worker_{path}.log", "w", encoding="utf-8") as handle:
            done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        workers.append(dict(path=path, exit_code=done.returncode, seconds=time.time()-tick,
                            cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES")))
        document_path = out/f"path_{path}.json"
        paths[path] = (json.loads(document_path.read_text(encoding="utf-8")) if document_path.is_file()
                       else dict(status="no_output", error=f"worker exit {done.returncode}, see worker_{path}.log"))
    end_gate = gate(args, when="end")
    host_doc = paths.get("host") or {}
    profile = dict(schema=SCHEMA, created_utc=created, code=dict(head=code["head"], dirty=code["dirty"]),
                   config=config, gate_start=start_gate, gate_end=end_gate, workers=workers, record=record,
                   inputs=host_doc.get("inputs") or dict(state_path=inputs["state"], pair=inputs["pair"]),
                   trial=(host_doc.get("host") or {}).get("trial"), paths=paths)
    profile["model"] = model_rows(paths, record)
    profile["bottleneck"] = bottleneck_statements(profile)
    (out/"profile.json").write_text(json.dumps(profile, indent=1, default=_json_default)+"\n", encoding="utf-8")
    (out/"profile.md").write_text(render_md(profile), encoding="utf-8")
    trial = out/"trial.npz"
    if trial.exists() and not args.keep_trial:
        trial.unlink()
    print(f"wrote {out/'profile.json'} and {out/'profile.md'}")
    return 0 if all(w["exit_code"] == 0 for w in workers) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.worker is not None:
        if args.worker not in ("host",)+BACKEND_PATHS:
            raise SystemExit(f"--worker: unknown path {args.worker}")
        return worker(args)
    return driver(args, argv)


if __name__ == "__main__":
    sys.exit(main())
