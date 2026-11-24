"""V1 (b): floating-point state of the processes that run the certificate (rounding mode, FTZ/DAZ), on this machine.

Read-only probe; no project file is written.  Each scenario runs in a fresh subprocess (python -B):
  A  jd_curl's certificate modules only (as the review unit and the evaluation process: no torch at module level);
  B  torch (CPU only: torch.cuda is never touched) plus the native libraries of the repair process and of the
     evaluation (scipy, nibabel, surface_distance, numba, llvmlite), imported BEFORE jd_curl; then some torch CPU kernels
     (float64 / float32 matmul, conv3d, linalg.solve) and a NumPy BLAS matmul in the main thread;
  C  positive control: torch.set_flush_denormal(True) in the main thread (sets FTZ and DAZ there), to show that the
     probe reads MXCSR correctly, what the run-time smoke tests catch, and what a thread created before / after inherits.
  D  the smoke tests under each of the four IEEE rounding modes (fesetround; restored to nearest at once).
In scenarios A and B a small certify_map runs first (so jd_curl's persistent pools exist: jd-curl-scan, jd-curl-edge-hull),
then the state is read in: the main thread, a new threading.Thread, a jd-curl-scan pool thread (thread_chunks.run_chunks),
the jd-curl-edge-hull pool thread (feasibility._kernel_pool) and a fresh ThreadPoolExecutor thread (as the fallback pool).
MXCSR is read with glibc fegetenv (x86-64 fenv_t: x87 control word at offset 0, MXCSR at offset 28).
Usage: python -B v1_fpstate.py            (driver: runs A, B, C, D and prints JSON)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def fpstate():
    import ctypes
    import ctypes.util
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    env = (ctypes.c_ubyte*32)()
    rc = libm.fegetenv(ctypes.byref(env))
    raw = bytes(env)
    cw = int.from_bytes(raw[0:2], "little")
    mx = int.from_bytes(raw[28:32], "little")
    return dict(fegetenv_rc=rc, fegetround=libm.fegetround(), x87_cw=hex(cw), x87_rc=(cw >> 10) & 3,
                mxcsr=hex(mx), mxcsr_rc=(mx >> 13) & 3, ftz=bool((mx >> 15) & 1), daz=bool((mx >> 6) & 1))


def arithmetic_checks():
    """The two run-time smoke tests of jd_curl, evaluated in the calling thread, plus a direct subnormal probe."""
    import numpy as np
    from jd_curl.three_d import interval as iv
    from jd_curl.three_d import compiled_enclosure as ce
    tiny = np.float64(np.finfo(np.float64).tiny)
    sub = np.nextafter(np.float64(0), np.float64(1))
    return dict(numpy_smoke=bool(iv.arithmetic_supported()), numba_smoke=bool(ce.arithmetic_supported(np.float64(0.))),
                numpy_tiny_half=float(np.divide(tiny, np.float64(2))), numpy_sub_times_one=float(np.multiply(sub, np.float64(1))))


def probe_all():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from jd_curl.three_d import feasibility as F
    from jd_curl.three_d import thread_chunks as tc

    def here(tag):
        d = dict(thread=tag, name=threading.current_thread().name)
        d.update(fpstate())
        d.update(arithmetic_checks())
        return d

    out = [here("main")]
    box = []
    t = threading.Thread(target=lambda: box.append(here("new threading.Thread")))
    t.start(); t.join()
    out += box
    res = tc.run_chunks(lambda lo, hi: here(f"jd-curl-scan chunk {lo}"), 2, workers=2)
    out += res
    out.append(F._kernel_pool().submit(here, "jd-curl-edge-hull (feasibility._kernel_pool)").result())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-cert-probe") as pool:
        out.append(pool.submit(here, "fresh ThreadPoolExecutor (as the fallback pool)").result())
    return out


def small_certificate():
    import numpy as np
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig, certify_map
    g = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit="mm")
    u = np.zeros(g.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = .3*np.random.default_rng(7).uniform(-1, 1, u[1:-1, 1:-1, 1:-1].shape)
    r = certify_map(u, g, config=PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536))
    return dict(status=r.status, reason=r.reason, n_fast=r.n_fast_cells, n_pre=r.n_prescreen_cells, n_fb=r.n_fallback_cells,
                min_lb=r.min_jd_lower_bound)


def loaded_libraries():
    libs = set()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and ".so" in parts[-1]:
                libs.add(os.path.basename(parts[-1]))
    return sorted(libs)


def rounding_modes():
    """Scenario D: what the two run-time smoke tests report under each IEEE rounding mode (glibc fesetround sets the x87
    and the SSE rounding control).  Only reads; the mode is restored to nearest before returning."""
    import ctypes
    import ctypes.util
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    out = {}
    try:
        for label, mode in (("FE_TONEAREST", 0x000), ("FE_DOWNWARD", 0x400), ("FE_UPWARD", 0x800), ("FE_TOWARDZERO", 0xc00)):
            rc = libm.fesetround(mode)
            out[label] = dict(fesetround_rc=rc, state=fpstate(), **arithmetic_checks())
    finally:
        libm.fesetround(0)
    out["restored"] = fpstate()
    return out


def scenario(name):
    rec = dict(scenario=name, pid=os.getpid())
    rec["at_start"] = fpstate()
    if name == "D":
        rec["rounding_modes"] = rounding_modes()
        return rec
    if name in ("B", "C"):
        import torch
        import scipy.ndimage, scipy.interpolate  # noqa: F401
        import nibabel  # noqa: F401
        import surface_distance  # noqa: F401
        import numba, llvmlite  # noqa: F401
        rec["torch"] = torch.__version__
        rec["torch_cuda_initialized"] = bool(torch.cuda.is_initialized())
        rec["after_imports"] = fpstate()
        if name == "B":
            # some torch / BLAS CPU work in the main thread (the repair process runs its optimizer on the GPU; its CPU-side
            # torch calls are not enumerated here, this only shows that ordinary CPU kernels leave MXCSR alone)
            import numpy as np
            a64 = torch.randn(256, 256, dtype=torch.float64)
            a32 = torch.randn(256, 256, dtype=torch.float32)
            _ = (a64 @ a64).sum().item(), (a32 @ a32).sum().item()
            _ = torch.nn.functional.conv3d(torch.randn(1, 2, 16, 16, 16), torch.randn(3, 2, 3, 3, 3)).sum().item()
            _ = torch.linalg.solve(a64 + 256*torch.eye(256, dtype=torch.float64), a64).sum().item()
            _ = float((np.random.default_rng(0).uniform(size=(300, 300)) @ np.ones((300, 300))).sum())
            rec["after_torch_and_blas_cpu_work"] = fpstate()
    if name == "C":
        # threads created BEFORE the switch: a plain thread parked on an event
        import threading
        ev, early = threading.Event(), []

        def parked():
            ev.wait()
            early.append(dict(fpstate(), **arithmetic_checks()))
        t = threading.Thread(target=parked, name="created-before-ftz")
        t.start()
        rec["set_flush_denormal_returned"] = bool(torch.set_flush_denormal(True))
        rec["after_set_flush_denormal"] = fpstate()
        ev.set(); t.join()
        rec["thread_created_before_switch"] = early[0]
        late = []
        t2 = threading.Thread(target=lambda: late.append(dict(fpstate(), **arithmetic_checks())), name="created-after-ftz")
        t2.start(); t2.join()
        rec["thread_created_after_switch"] = late[0]
        rec["certificate_under_ftz"] = small_certificate()
        # what FTZ/DAZ does to one outward-rounded interval product (Lemma A.3 needs gradual underflow)
        import numpy as np
        from fractions import Fraction
        from jd_curl.three_d.interval import Interval
        a = Interval.point(np.array([-3e-160]))
        b = Interval.point(np.array([5e-160]))
        prod = a*b
        exact = Fraction(-3e-160)*Fraction(5e-160)
        rec["ftz_interval_product"] = dict(lo=float(prod.lo[0]), hi=float(prod.hi[0]), exact="see gradual_interval_product (printed after the reset)",
                                           contains_exact=bool(Fraction(float(prod.lo[0])) <= exact <= Fraction(float(prod.hi[0]))))
        torch.set_flush_denormal(False)
        rec["after_reset"] = fpstate()
        prod = a*b
        rec["gradual_interval_product"] = dict(lo=float(prod.lo[0]), hi=float(prod.hi[0]), exact=float(exact),
                                               contains_exact=bool(Fraction(float(prod.lo[0])) <= exact <= Fraction(float(prod.hi[0]))))
        rec["probes"] = None
    else:
        rec["certificate"] = small_certificate()
        rec["after_certificate_main"] = fpstate()
        rec["probes"] = probe_all()
    if name == "B":
        libs = loaded_libraries()
        rec["n_loaded_so"] = len(libs)
        rec["loaded_so_sample"] = [x for x in libs if any(k in x for k in ("torch", "omp", "mkl", "openblas", "llvmlite", "gomp", "cuda", "cudnn", "c10"))]
    return rec


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--scenario":
        print(json.dumps(scenario(sys.argv[2]), sort_keys=True))
        return 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    for name in ("A", "B", "C", "D"):
        p = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--scenario", name], env=env,
                           capture_output=True, text=True)
        print(f"### scenario {name}: exit {p.returncode}")
        if p.stdout.strip():
            print(json.dumps(json.loads(p.stdout.strip().splitlines()[-1]), indent=1, sort_keys=True))
        if p.returncode:
            print(p.stderr[-3000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
