"""V1 (b): the compiled code of the certificate's three paths on this machine (no contraction, no reassociation,
no fast-math, no FTZ/DAZ or rounding-mode instructions).

Read-only.  The kernels are compiled in this process for the host CPU (cache=False, as in a run without jit_cache;
the runs' jit_cache stores the machine code of an identical compile, keyed by CPU name and features), by calling the
same entry points the certificate calls (certify_map with configuration C, soa_margins, the single-cell enclosures).
For every Numba dispatcher of the three paths and every compiled signature it reports, from the optimized LLVM IR:
  - fast-math flags on any floating-point instruction or call (fast nnan ninf nsz arcp contract afn reassoc);
  - llvm.fma / llvm.fmuladd intrinsics, llvm.*.reduce.f* intrinsics;
  - function attributes naming *-fp-math (unsafe / no-nans / no-infs / no-signed-zeros / approx-func / denormal);
  - the data flow: fadd/fsub instructions with an operand defined by an fmul (the only places where a contraction
    into an FMA could happen at all), and calls to nextafter (numba lowers np.nextafter to @numba_nextafter, numba/_helperlib.c:213, which returns glibc nextafter);
  - binary64 only: no float / half / x86_fp80 / fp128 arithmetic, no fptrunc / fpext;
and from the assembly: FMA instructions (v[f]n?m(add|sub)...), MXCSR / x87 control-word writes (ldmxcsr, fldcw),
approximate instructions (rcp/rsqrt), x87 arithmetic, single-precision (ss/ps) arithmetic or conversions, and the number of
packed vs scalar double arithmetic instructions.
A positive control compiles a*b+c with fastmath False / True / {'contract'} and a sum loop with fastmath False / True.
Full IR and asm go to a temporary directory (sha256 printed); nothing in the repository is written.
Usage: python -B v1_codegen.py
"""
import hashlib
import os
import re
import sys
import tempfile

import numpy as np

FLAGS = r"(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)"
FP_FLAG = re.compile(r"=\s*(?:fadd|fsub|fmul|fdiv|frem|fneg|fcmp|call|select|phi)\s+(?:" + FLAGS + r"\s+)*" + FLAGS + r"\b")
FP_FLAG_NOASSIGN = re.compile(r"^\s*call\s+(?:" + FLAGS + r"\s+)+")
FMA_INTR = re.compile(r"@llvm\.(?:fma|fmuladd)\.")
REDUCE_INTR = re.compile(r"@llvm\.(?:experimental\.)?vector\.reduce\.f\w+")
FPMATH_ATTR = re.compile(r'"([\w-]*fp-math[\w-]*)"="([^"]*)"')
FMA_ASM = re.compile(r"^\s*v?f(?:n)?m(?:add|sub)\w*", re.IGNORECASE | re.MULTILINE)
CTRL_ASM = re.compile(r"^\s*(?:v?ldmxcsr|fldcw|fldenv|frstor|fxrstor|xrstor)\b", re.MULTILINE)
APPROX_ASM = re.compile(r"^\s*v?(?:rcp|rsqrt)\w*", re.MULTILINE)
PACKED = re.compile(r"^\s*v(?:add|sub|mul|div)pd\b", re.MULTILINE)
SCALAR = re.compile(r"^\s*v?(?:add|sub|mul|div)sd\b", re.MULTILINE)
X87 = re.compile(r"^\s*f(?:add|sub|subr|mul|div|divr|ld|st|ild|ist)p?(?:s|l|t|ll|q)?\b", re.MULTILINE)   # AT&T suffixes (fldl, fstpl, fmull)
SINGLE_ASM = re.compile(r"^\s*v?(?:add|sub|mul|div|sqrt|cvtsd2|cvtpd2)(?:ss|ps)\b|^\s*v?cvt(?:sd2ss|pd2ps)\b", re.MULTILINE)
F32_IR = re.compile(r"=\s*(?:fadd|fsub|fmul|fdiv|fptrunc|fpext)\b[^\n]*\b(?:float|half|x86_fp80|fp128)\b")
DEF = re.compile(r"^\s*(%[-\w.$\"]+)\s*=\s*(\w+)\b")
OPER = re.compile(r"%[-\w.$\"]+")


def ir_dataflow(ir):
    """Per function: fadd/fsub with an fmul operand; nextafter calls."""
    fused_candidates, nextafter_calls, n_fadd, n_fmul = 0, 0, 0, 0
    examples = []
    for body in ir.split("\ndefine ")[1:]:
        kind = {}
        lines = body.splitlines()
        for line in lines:
            m = DEF.match(line)
            if m:
                kind[m.group(1)] = m.group(2)
        for line in lines:
            m = DEF.match(line)
            if ("@numba_nextafter" in line or "@nextafter" in line) and "call" in line:
                nextafter_calls += 1
            if not m:
                continue
            op = m.group(2)
            if op == "fmul":
                n_fmul += 1
            if op in ("fadd", "fsub"):
                n_fadd += 1
                rhs = line.split("=", 1)[1]
                if any(kind.get(o) == "fmul" for o in OPER.findall(rhs)):
                    fused_candidates += 1
                    if len(examples) < 3:
                        examples.append(line.strip()[:160])
    return dict(fadd_fsub=n_fadd, fmul=n_fmul, fadd_fsub_with_fmul_operand=fused_candidates, nextafter_calls=nextafter_calls,
                examples=examples)


def inspect(name, dispatcher, outdir):
    rows = []
    for sig in dispatcher.signatures:
        ir = dispatcher.inspect_llvm(sig)
        asm = dispatcher.inspect_asm(sig)
        tag = re.sub(r"[^\w]+", "_", f"{name}_{sig}")[:150]
        for ext, text in (("ll", ir), ("s", asm)):
            with open(os.path.join(outdir, f"{tag}.{ext}"), "w") as f:
                f.write(text)
        flag_lines = [ln.strip()[:160] for ln in ir.splitlines() if FP_FLAG.search(ln) or FP_FLAG_NOASSIGN.search(ln)]
        attrs = sorted(set(FPMATH_ATTR.findall(ir)))
        row = dict(kernel=name, signature=str(sig), targetoptions_fastmath=dispatcher.targetoptions.get("fastmath"),
                   ir_sha256=hashlib.sha256(ir.encode()).hexdigest()[:16], asm_sha256=hashlib.sha256(asm.encode()).hexdigest()[:16],
                   ir_lines=ir.count("\n"), asm_lines=asm.count("\n"),
                   ir_fastmath_flag_lines=len(flag_lines), ir_fastmath_examples=flag_lines[:3],
                   ir_fma_intrinsics=len(FMA_INTR.findall(ir)), ir_vector_reduce_intrinsics=sorted(set(REDUCE_INTR.findall(ir))),
                   ir_fp_math_attributes=attrs, ir_vector_double_types=ir.count("x double>"),
                   asm_fma=len(FMA_ASM.findall(asm)), asm_fma_examples=[m.strip() for m in FMA_ASM.findall(asm)][:3],
                   asm_control_writes=[m.strip() for m in CTRL_ASM.findall(asm)][:5], asm_approx=len(APPROX_ASM.findall(asm)),
                   asm_packed_pd_arith=len(PACKED.findall(asm)), asm_scalar_sd_arith=len(SCALAR.findall(asm)),
                   ir_non_binary64_fp_ops=len(F32_IR.findall(ir)), ir_fptrunc_fpext=len(re.findall(r"=\s*(?:fptrunc|fpext)\b", ir)),
                   asm_x87_arith=len(X87.findall(asm)), asm_single_precision=len(SINGLE_ASM.findall(asm)))
        row.update(ir_dataflow(ir))
        rows.append(row)
    if not rows:
        rows.append(dict(kernel=name, signature=None, note="not compiled in this process"))
    return rows


def compile_everything():
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d import feasibility as F
    from jd_curl.three_d import compiled_enclosure as ce
    from jd_curl.three_d import soa_enclosure as soa
    from jd_curl.three_d import certificate_prescreen as cps
    from jd_curl.three_d import compiled_fallback as cf
    from jd_curl.three_d import fast_checks as fc
    from jd_curl.three_d import interpolation
    cfg = F.PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)
    g = make_grid_3d(nx=9, ny=8, nz=7, hx=1., hy=1., hz=1., unit="mm")
    rng = np.random.default_rng(11)
    out = {}
    for scale in (.05, .3, .45):
        u = np.zeros(g.shape+(3,))
        u[1:-1, 1:-1, 1:-1] = scale*rng.uniform(-1, 1, u[1:-1, 1:-1, 1:-1].shape)
        r = F.certify_map(u, g, config=cfg)
        out[scale] = (r.status, r.n_fast_cells, r.n_prescreen_cells, r.n_fallback_cells, r.max_depth)
    # the SoA execution (not selected by the method's processes; compiled here only to inspect it)
    soa.soa_margins(u, g, 0.)
    # the single-cell enclosures (certify_cell; not on certify_map's path)
    v = rng.uniform(-.1, .1, (3, 8, 3))
    F.compiled_jacobian_enclosure(v, (1., 1., 1.), kernel="numba-directed-edge-hull-v1")
    F.compiled_jacobian_enclosure(v, (1., 1., 1.), kernel="numba-hoisted-edge-hull-v1")
    # the domain test (Remark 4.5, not part of Thm 4.4): force the compiled scan on a large array
    big = make_grid_3d(nx=104, ny=104, nz=104, hx=1., hy=1., hz=1., unit="mm")
    interpolation.check_map_domain(np.zeros(big.shape+(3,)), big)
    kernels = [
        ("(i) compiled_enclosure._chunk_hoisted_margins [used: libm execution]", ce._chunk_hoisted_margins),
        ("(i) compiled_enclosure.arithmetic_supported [smoke test]", ce.arithmetic_supported),
        ("(i) soa_enclosure._soa_rows [SoA execution, not selected]", soa._soa_rows),
        ("(i) soa_enclosure._row_entries", soa._row_entries),
        ("(i) soa_enclosure._row_det", soa._row_det),
        ("(i) soa_enclosure._transpose", soa._transpose),
        ("(i) compiled_enclosure._hoisted_bounds [certify_cell only]", ce._hoisted_bounds),
        ("(i) compiled_enclosure._bounds [certify_cell only]", ce._bounds),
        ("(ii) certificate_prescreen kernel [used]", cps._kernel()),
        ("(iii) compiled_fallback._fallback_kernel [used]", cf._fallback_kernel),
        ("(R4.5) fast_checks._domain_counts [domain test, not Thm 4.4]", getattr(fc, "_domain_counts", None)),
    ]
    return out, kernels


def positive_controls():
    from numba import njit

    def mad(a, b, c):
        return a*b+c

    def total(x):
        acc = 0.
        for i in range(x.shape[0]):
            acc += x[i]
        return acc
    rows = []
    for label, fm in (("fastmath=False", False), ("fastmath=True", True), ("fastmath={'contract'}", {"contract"})):
        f = njit(fastmath=fm)(mad)
        f(1., 2., 3.)
        asm = f.inspect_asm(f.signatures[0])
        ir = f.inspect_llvm(f.signatures[0])
        a, b, c = 0.1, 10., -1.
        rows.append(dict(control="a*b+c", option=label, asm_fma=len(FMA_ASM.findall(asm)), ir_flag_lines=sum(bool(FP_FLAG.search(l)) for l in ir.splitlines()),
                         value_hex=f(a, b, c).hex(), python_value_hex=(a*b+c).hex()))
    def f32(a, b):
        return np.float32(a)*np.float32(b)
    f = njit(f32)
    f(1., 2.)
    asm = f.inspect_asm(f.signatures[0])
    ir = f.inspect_llvm(f.signatures[0])
    rows.append(dict(control="float32 product", option="fastmath=False", asm_single_precision=len(SINGLE_ASM.findall(asm)),
                     ir_non_binary64_fp_ops=len(F32_IR.findall(ir)), ir_fptrunc_fpext=len(re.findall(r"=\s*(?:fptrunc|fpext)\b", ir))))
    x = np.random.default_rng(3).uniform(-1, 1, 1001)
    for label, fm in (("fastmath=False", False), ("fastmath=True", True)):
        f = njit(fastmath=fm)(total)
        val = f(x)
        asm = f.inspect_asm(f.signatures[0])
        seq = 0.
        for t in x:
            seq += t
        rows.append(dict(control="sequential sum", option=label, asm_packed_addpd=len(re.findall(r"^\s*vaddpd\b", asm, re.M)),
                         asm_scalar_addsd=len(re.findall(r"^\s*vaddsd\b", asm, re.M)), equals_python_sequential_sum=val == seq))
    return rows


def target_info():
    import llvmlite
    import llvmlite.binding as ll
    import numba
    from numba.core import config
    from numba.core.registry import cpu_target
    cg = cpu_target.target_context.codegen()
    feats = ll.get_host_cpu_features()
    return dict(numba=numba.__version__, llvmlite=llvmlite.__version__, llvm=".".join(map(str, ll.llvm_version_info)),
                host_cpu=ll.get_host_cpu_name(), host_fma=bool(feats.get("fma")), host_avx2=bool(feats.get("avx2")),
                host_avx512f=bool(feats.get("avx512f")), numba_CPU_NAME=config.CPU_NAME, numba_CPU_FEATURES=config.CPU_FEATURES,
                numba_OPT=str(config.OPT), numba_USING_SVML=config.USING_SVML, codegen_tm_features_has_fma="+fma" in (cg._tm_features or ""),
                target_triple=ll.get_process_triple())


def main():
    outdir = tempfile.mkdtemp(prefix="v1_codegen_")
    print("# full IR / asm written to", outdir, "(temporary, not committed)")
    print("# target:", target_info())
    statuses, kernels = compile_everything()
    print("# certify_map warm-up (scale: status, n_fast, n_prescreen, n_fallback, max_depth):", statuses)
    total_bad = 0
    for name, disp in kernels:
        if disp is None:
            print(f"== {name}: not defined")
            continue
        for row in inspect(name, disp, outdir):
            print("==", row.pop("kernel"))
            for k, v in row.items():
                print(f"   {k}: {v}")
            if row.get("signature") is not None:
                total_bad += (row["ir_fastmath_flag_lines"]+row["ir_fma_intrinsics"]+row["asm_fma"]+len(row["asm_control_writes"])+row["asm_approx"]
                              + row["ir_non_binary64_fp_ops"]+row["ir_fptrunc_fpext"]+row["asm_x87_arith"]+row["asm_single_precision"])
    print("# positive controls (the same detectors on code that is allowed to contract / reassociate)")
    for row in positive_controls():
        print("  ", row)
    print(f"# TOTAL over the certificate kernels: fast-math flag lines + FMA intrinsics + FMA instructions + MXCSR/x87 control writes + approx"
          f" + non-binary64 FP ops (IR float/half/x86_fp80/fp128, fptrunc/fpext; asm x87 arithmetic, ss/ps arithmetic) = {total_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
