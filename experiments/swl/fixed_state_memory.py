"""C04-6 gate 12 (memory; work package SWLI, 2026-10-31): the fixed-state measurement.  First run 10-31 (lead, code
2d81e3d, old script, one pair, brain first; /home/rmcfz/seq_vp_tag_results/swl_gate12): host peak brain 7019 MiB,
support_shared 7206 MiB (+187 MiB; fails the gate as then written; kept as it is); device peaks equal.  Revised by work
package SWLM (below, and reviews/SWLM_复核_20261101.md).  The lead runs it when the GPU is idle and no other
native-scale run is going on (machine section of the project rules; one arm per process): experiments/swl/gate12_repeat.sh (an
uncounted warm-up run, then ABBA pairs, the three criteria of the revised gate).

    BAYESREG_MEMTRACE=$OUT/<arm>.memtrace.jsonl BAYESREG_MEMTRACE_RESET_HWM=1 \\
    python -B experiments/swl/fixed_state_memory.py --arm brain|support_shared --out $OUT/<arm>.json \\
        [--seed-npz <a native final_native.npz, e.g. mem_v1/M03/run/final_native.npz>] [--device cuda] [--cpu-threads 4]
        [--data-root DIR] [--allowlist PATH]

What one arm does (the contract's "prepare + 层首（v 与 σ）+ G₀ + 一次求值（含梯度与 c_dat）" on one native seed state),
with the holdings and the heap releases of run_pair / solve_multilevel at the native level (SWLM):
  1. the development pair stage_L2R_val_v1 0438 <- 0439 with the H2 configuration of experiments/mem/measure.sh
     (structure channels unet35_v1/pred, tissue atropos_v1, gauss27+closed27 beta_S 1 Simpson, torch64 cuda,
     affine_v1 / certificate), channels and pyramid by run_pair.prepare_channels; only the native LevelInput is kept;
     then release_free_heap() as run_pair after the channel preparation;
  2. the seed u: --seed-npz (its "u", a certified native state; recommended, the gate's "原生层种子状态") or u = 0;
  3. memtrace marks: swl.start; template + backend.prepare (swl.prepare; the LevelInput is dropped once the template
     is built, as solve_multilevel(release_inputs=True) replaces it by grid_only); the seed evaluation and the
     commit's free vector theta (held as solve_multilevel's Commit holds it) (swl.seed_evaluated; support_shared has
     already dropped the seed evaluation there, as the solver does before the support pass; brain still holds it for
     its level start); the level start (brain: GaussianMStep.level_start; support_shared:
     bayesreg.model.support.support_level on the CPU reference, release_free_heap() as the solver after the support
     pass, SupportMStep.level_start, the level fields with v) (swl.level_start: under support_shared this segment is
     the support pass alone); the round state's prepare (swl.state); G0 = g0_of (swl.g0; then release_free_heap() as
     the solver after G0); one evaluation with gradient and c_dat (swl.eval).  Every mark also records
     /proc/self/smaps_rollup (smaps_rss_mb, smaps_anon_mb, smaps_anon_huge_mb, smaps_locked_mb, smaps_swap_mb);
  4. the JSON: per mark the memtrace fields (host rss / hwm since the previous mark with the HWM reset, device
     allocated / peak, glibc mallinfo2, smaps_rollup), the overall host peak (max hwm_mb) and device peak (max
     cuda_peak_mb), the allocated heap per mark (malloc_live_mb = malloc_inuse_mb + malloc_mmap_mb, mallinfo2
     uordblks + hblkhd: bytes held by allocations), v_bytes, the level-start record, page_settings, argv.
Reading the allocator fields after a release: malloc_trim hands free pages back with MADV_DONTNEED but glibc keeps them
on its free lists, so mallinfo2's arena (system bytes) and free (fordblks) still count them.  At marks after a release
(swl.level_start of support_shared, swl.g0, swl.eval) malloc_free_mb is therefore not resident free memory; the
resident anonymous memory is smaps_anon_mb.  The "Memory peak" line systemd-run prints for these units is not usable
(548 KiB in the first run), and cg_mb is not comparable between the arms (page cache is charged to the unit that read
the files first): use hwm_mb, malloc_live_mb and smaps_*.
Gate 12 (contract C04-6; measurement design changed by the lead after the failed first run, adopted 11-01 after the
second SWLM review, d7c7476; the contract text rules): an uncounted warm-up run, then >= 5 ABBA pairs, 4 threads;
(a) allocated heap (malloc_live_mb) support_shared - brain <= v_bytes + 8 MiB at swl.level_start, swl.state, swl.g0
and swl.eval, per pair; (b) device allocated peak difference <= 32 MiB, per pair; (c) median host peak difference <=
64 MiB, the host peak taken over the swl.* segments only (peaks.host_hwm_swl_mb); "未决" when brain's range exceeds
128 MiB, taking precedence over a failed (c).  experiments/swl/gate12_summary.py computes them.  The whole-run part of
gate 12 is experiments/mem/measure.sh with the M03 command and --extra "--sigma-domain support_shared" (recorded only).

SWLM findings on the first run (memtrace fields of swl_gate12/*.json; work package SWLM, 11-01): the allocations
held (malloc_inuse_mb; malloc_mmap_mb is equal) differ by +28.4 MiB from swl.level_start on (v = 26.0 MiB, the rest a
constant 2.4 MiB); the rest of the RSS difference is free memory that glibc keeps in its arenas (malloc_free_mb +39 MiB
at swl.level_start, +183 MiB at swl.g0), not held objects.  The +39 MiB at the level start matches the support pass's
two length-n float64 temporaries (main arena) and the CPU workers' spline chunks (their arenas); the solver now hands
them back right after the pass (and so does this script).  Whether the further ~145 MiB retained during the threaded
CPU G0 is run-to-run variation or systematic is open, to be settled by the alternating re-measurement: the first run
has one order only (brain first), and before the arms diverged (swl.start) the two processes already differed by
318 MiB of transparent huge pages at equal RSS (an order or system-state effect).  The synthetic CPU runs
(/home/rmcfz/seq_vp_tag_results/swlm_synthetic_20261101; 681k mask nodes, v 8.4 MiB, numpy backend) do not settle it:
they ran while the PILOT native runs were going on (10-31 22:29-23:22 CDT; the machine was not idle); the "7 runs" of
brain came from three batches and two versions of the synthetic script (start RSS 316.6 vs 299.3 MiB; all invoked with
--warm per the record, but the JSON files do not record the command line); there the overall host peak of
brain ranged over 142 MiB (4 threads; 19 MiB over 2 runs with 1 thread) while the allocated heap differed by
+10.4..+10.8 MiB between the arms in every run; and the release after the support pass did not measurably lower the
peak (difference of the mean arm differences without and with it 32 +- 38 MiB; in the interleaved batch the sign was
reversed).  page_settings in the JSON records NumPy's hugepage madvise and the kernel's THP settings of the run.
This script reads data/ (development pair only: hard constraint 6 -- 0438 and 0439 are in stage_L2R_val_v1) and writes
only --out (must not exist).
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PAIR = "stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1"
SEG = Path("/home/rmcfz/seq_vp_tag_results/seg")
ARMS = ("brain", "support_shared", "support")


def smaps_rollup():
    """Resident memory of the process from /proc/self/smaps_rollup (MiB): Rss, Anonymous, AnonHugePages, Locked, Swap
    as smaps_rss_mb, smaps_anon_mb, smaps_anon_huge_mb, smaps_locked_mb, smaps_swap_mb; {} where unreadable."""
    keys = dict(Rss="smaps_rss_mb", Anonymous="smaps_anon_mb", AnonHugePages="smaps_anon_huge_mb",
                Locked="smaps_locked_mb", Swap="smaps_swap_mb")
    out = {}
    try:
        with open("/proc/self/smaps_rollup", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in keys:
                    out[keys[key]] = int(rest.split()[0])/1024.
    except (OSError, ValueError, IndexError):
        return {}
    return out


def measure_arm(arm, inputs, eta, backend, seed, *, theta=.05, cpu_threads=4, mark=None):
    """One arm of the fixed-state sequence (module docstring, step 3); returns (g0, c_dat, energy, gradient norm,
    level-start record or None).  inputs: a list holding the native LevelInput only, handed over by the caller (it
    keeps no other reference): the entry is replaced by None once the template is built (solve_multilevel's
    release_inputs).  mark(label): the memtrace mark (default bayesreg.memtrace.mark with arm=... and smaps_rollup)."""
    from jd_curl.three_d.boundary import restrict_free
    from bayesreg import memtrace
    from bayesreg.heap import release_free_heap
    from bayesreg.model.support import support_level
    from bayesreg.readonly import freeze_new
    from bayesreg.solve import GaussianMStep, SupportMStep
    from bayesreg.solve.multilevel import _template
    from bayesreg.solve.stopping import g0_of

    if arm not in ARMS:
        raise ValueError(f"arm: one of {ARMS}")
    if mark is None:
        def mark(label):
            if memtrace.enabled():
                memtrace.mark(label, arm=arm, **smaps_rollup())
    native = inputs[0]
    grid, n_channels = native.grid, native.fixed.shape[0]
    mark("swl.start")
    host = _template(native, eta, grid.omega, 8192, n_channels, cpu_threads, "gauss27+closed27",
                     logvol_closed27_share=1., closed27_weights="simpson")
    inputs[0] = native = None                                       # release_inputs: the moving stack goes
    template = backend.prepare(host)
    mark("swl.prepare")
    point = backend.evaluate(template, seed)
    theta_free = freeze_new(restrict_free(seed, grid))             # solve_multilevel's Commit holds it for the level
    record = None
    if arm != "brain":
        del point                    # the solver drops seed_point before the support pass; brain's level start reads it
    mark("swl.seed_evaluated")
    if arm == "brain":
        sigma2 = GaussianMStep().level_start(template, point)
        del point
        level_fields = {}
    else:
        level = support_level(host, seed, theta, sigma_domain=arm, level=3)
        release_free_heap()                                        # solve_multilevel after the support pass (SWLM)
        start = SupportMStep(arm).level_start(level)
        sigma2 = start.internal
        # as solve_multilevel since review SWLI important 1: only the fields, no second resident host ModelState
        level_fields = dict(sigma_domain=arm, support_threshold=theta, support_v=level.v)
        record = dict(level.record(), sigma2_start=start.record)
    mark("swl.level_start")
    state = backend.prepare(host.replace(sigma2=np.asarray(sigma2), **level_fields))
    mark("swl.state")
    g0 = g0_of(backend, state)
    release_free_heap()                                            # solve_multilevel after G0 (MEM)
    mark("swl.g0")
    ev = backend.evaluate(state, seed)
    gradient = backend.gradient(state, ev)
    c_dat = backend.c_dat(state, ev)
    mark("swl.eval")
    del theta_free, template
    return g0, c_dat, ev.energy, float(np.linalg.norm(gradient)), record


def page_settings():
    """The settings behind the RSS of NumPy arrays (recorded, not changed): NumPy's madvise(MADV_HUGEPAGE) on large
    buffers (environment NUMPY_MADVISE_HUGEPAGE), the kernel's THP mode and khugepaged's max_ptes_none (with 511 a
    collapse fills absent pages of a 2 MiB range, so freed and trimmed arena pages can become resident again)."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    try:
        madvise = bool(np._core.multiarray._get_madvise_hugepage())          # NumPy 2 (np.core is deprecated)
    except AttributeError:
        madvise = None
    base = "/sys/kernel/mm/transparent_hugepage/"
    return dict(numpy_madvise_hugepage=madvise, thp_enabled=read(base+"enabled"), thp_defrag=read(base+"defrag"),
                khugepaged_max_ptes_none=read(base+"khugepaged/max_ptes_none"))


def main(argv=None):
    from bayesreg import memtrace
    from bayesreg.cli import run_pair
    from bayesreg.heap import release_free_heap
    from bayesreg.io import load_pair
    from bayesreg.solve import eta_temporary, make_backend

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-npz", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--theta", type=float, default=.05)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--data-root", default=None, help="run_pair --data-root (default run_pair's, under the code tree)")
    parser.add_argument("--allowlist", default=None, help="run_pair --allowlist (default run_pair's)")
    ns = parser.parse_args(argv)
    out = Path(ns.out)
    if out.exists():
        raise SystemExit(f"{out}: exists (nothing is overwritten)")
    if not memtrace.enabled():
        raise SystemExit("set BAYESREG_MEMTRACE (and BAYESREG_MEMTRACE_RESET_HWM=1) for this measurement")
    logger = logging.getLogger("swl.fixed_state_memory")
    logging.basicConfig(level=logging.INFO)
    extra = ((["--data-root", ns.data_root] if ns.data_root else [])
             + (["--allowlist", ns.allowlist] if ns.allowlist else []))
    args = run_pair.parse_args(["--pair", PAIR, "--channel-source", "structure", "--out", str(out)+".unused",
                                "--tissue-root", str(SEG/"atropos_v1"), "--structure-root", str(SEG/"unet35_v1"/"pred"),
                                "--structure-infer", "never"]+extra)
    pair = load_pair(run_pair.resolve_pair(args.data_root, PAIR), allowlist=args.allowlist)
    structure = run_pair.structure_source(args)
    inputs, pt_f, mask_f, _ = run_pair.prepare_channels(pair, "structure", args.tissue_root, logger,
                                                       structure=structure)
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    native = [inputs[-1]]
    del inputs, pt_f, mask_f
    release_free_heap()                                            # run_pair after the channel preparation (MEM)
    if ns.seed_npz:
        with np.load(ns.seed_npz) as data:
            seed = np.array(data["u"])
    else:
        seed = np.zeros(native[0].grid.shape+(3,))
    backend = make_backend("torch64", device=ns.device, logvol_points="gauss27+closed27")
    g0, c_dat, energy, gradient_norm, record = measure_arm(ns.arm, native, eta, backend, seed, theta=ns.theta,
                                                           cpu_threads=ns.cpu_threads)
    lines = [json.loads(line) for line in Path(memtrace._path).read_text().splitlines()]
    swl = [r for r in lines if str(r.get("label", "")).startswith("swl.")]
    peaks = dict(host_hwm_mb=max(r.get("hwm_mb") or 0 for r in lines),
                 host_hwm_lifetime_mb=max(r.get("hwm_lifetime_mb") or 0 for r in lines),
                 device_peak_mb=max(r.get("cuda_peak_mb") or 0 for r in lines),
                 host_hwm_swl_mb=max(r.get("hwm_mb") or 0 for r in swl),             # the gate's host peak (c)
                 device_peak_swl_mb=max(r.get("cuda_peak_mb") or 0 for r in swl))
    live = {r["label"]: (r.get("malloc_inuse_mb") or 0)+(r.get("malloc_mmap_mb") or 0) for r in lines
            if str(r.get("label", "")).startswith("swl.")}
    v_bytes = None if record is None else int(np.prod(record["v_shape"]))
    result = dict(arm=ns.arm, pair=PAIR, seed_npz=ns.seed_npz, theta=ns.theta if ns.arm != "brain" else None,
                  cpu_threads=ns.cpu_threads, argv=sys.argv, code_root=str(REPO), data_root=args.data_root,
                  allowlist=args.allowlist, g0=g0, c_dat=c_dat, energy=energy, gradient_norm=gradient_norm,
                  peaks=peaks, malloc_live_mb=live, v_bytes=v_bytes, memtrace=lines, level_start=record,
                  page_settings=page_settings(),
                  gate=("C04-6 gate 12 (adopted 11-01, d7c7476): (a) malloc_live support_shared - brain <= v_bytes "
                        "+ 8 MiB at swl.level_start / state / g0 / eval, per pair; (b) device peak difference <= 32 "
                        "MiB, per pair; (c) median host peak (swl.* segments) difference <= 64 MiB over >= 5 ABBA "
                        "pairs, undecided when brain's range > 128 MiB; the contract text rules"))
    out.write_text(json.dumps(result, indent=1, default=str)+"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
