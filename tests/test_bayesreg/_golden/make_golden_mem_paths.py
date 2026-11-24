"""Generate the digests of the solver paths the MEM golden did not reach (work package FUP1; review
MEM_复核_20261031 important 1): CPU fallback, injected device faults, option (b), gauss27+corners, gauss27 with the
metric scaling off over rounds 1,1,1,3, and (regression only) the C08-8(f) GF recheck.  The cases and the digest are
those of tests/test_bayesreg/solve/_mem_paths_digest.py.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_mem_paths.py --ref 9747553 --cases golden \\
        --out tests/test_bayesreg/_golden/golden_9747553_mem_paths.json
    python tests/test_bayesreg/_golden/make_golden_mem_paths.py --ref d046b77 --cases gf \\
        --out tests/test_bayesreg/_golden/golden_d046b77_mem_gf.json
    python tests/test_bayesreg/_golden/make_golden_mem_paths.py --ref worktree --cases golden   # print only

--cases golden on 9747553, the code just before the MEM memory changes (main facf4b8 plus the read-only memtrace marks;
the reference of golden_9747553_mem.json): the test compares the current tree with the code before MEM.
--cases gf on the current code: on 9747553 the GF recheck inside run_pg raised TypeError (fixed by GFR, 34ddaef), so
these digests only guard against later changes; they do not show bitwise equality with the code before MEM.

How (--ref <commit>), as make_golden_mem.py: `git archive <commit> bayesreg jd_curl tests/test_bayesreg configs` into a
new temporary directory; the two digest modules of THIS tree (_mem_digest.py, _mem_paths_digest.py; the second did not
exist at the reference) and this script are copied into it; a child Python with cwd = PYTHONPATH = that directory, one
BLAS / OpenMP / MKL / numba thread, CUDA_VISIBLE_DEVICES and BAYESREG_MEMTRACE empty runs this script with --compute and
prints the digests; the directory is removed.  Nothing in the repository is written except --out, which must not exist
(hard constraint 4).
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[3]
DIGEST_MODULES = ("tests/test_bayesreg/solve/_mem_digest.py", "tests/test_bayesreg/solve/_mem_paths_digest.py")
THIS = "tests/test_bayesreg/_golden/make_golden_mem_paths.py"
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                     NUMBA_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                     BAYESREG_MEMTRACE="")
CASE_SETS = ("golden", "gf")


def compute(case_set):
    import numpy as np
    import torch
    from tests.test_bayesreg.solve import _mem_paths_digest as P
    tick = time.perf_counter()
    digests = P.compute(case_set)
    import bayesreg
    return dict(digests=digests, cases={name: dict(backend=b, options=list(o), patches=list(p))
                                        for name, (b, o, p) in P.CASE_SETS[case_set].items()},
                seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(ref, case_set):
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    env = dict(os.environ, **SINGLE_THREAD)
    if ref == "worktree":
        commit = None
        env["PYTHONPATH"] = str(REPO)
        out = subprocess.run([sys.executable, "-B", str(REPO/THIS), "--compute", "--cases", case_set], cwd=REPO,
                             env=env, capture_output=True, text=True, check=True).stdout
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_mem_paths_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl",
                                      "tests/test_bayesreg", "configs"], cwd=REPO, capture_output=True,
                                     check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            for rel in (*DIGEST_MODULES, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = subprocess.run([sys.executable, "-B", str(root/THIS), "--compute", "--cases", case_set], cwd=root,
                                 env=env, capture_output=True, text=True, check=True).stdout
    child = json.loads(out.strip().splitlines()[-1])
    purpose = ("digests of the solver paths the MEM golden did not reach (CPU fallback, device faults, option (b), "
               "gauss27+corners, gauss27 metric off rounds 1,1,1,3) on the code before MEM: bitwise gate"
               if case_set == "golden" else
               "regression digests of the C08-8(f) GF recheck path on the current code (the code before MEM raised "
               "TypeError there): they guard later changes only, they do not show equality with the code before MEM")
    return dict(purpose=purpose, case_set=case_set, command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty,
                digest_modules={rel: _sha256(REPO/rel) for rel in DIGEST_MODULES},
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg configs (temporary "
                               "directory)" if commit else "this working tree",
                child_bayesreg_path=child["bayesreg_path"], environment=child["environment"],
                seconds=child["seconds"], cases=child["cases"], digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="9747553", help="commit to extract (default 9747553) or 'worktree'")
    parser.add_argument("--cases", choices=CASE_SETS, default="golden")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--compute", action="store_true", help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.compute:
        print(json.dumps(compute(ns.cases)))
        return 0
    if ns.out is not None and Path(ns.out).exists():
        raise SystemExit(f"{ns.out}: exists (nothing is overwritten)")
    record = generate(ns.ref, ns.cases)
    text = json.dumps(record, indent=1, sort_keys=True)+"\n"
    if ns.out is not None:
        with open(ns.out, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
