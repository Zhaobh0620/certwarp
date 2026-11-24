"""Generate the golden trajectory digests of the MEM gate (work package MEM; memory-only changes, bitwise unchanged).

The digests of tests/test_bayesreg/solve/_mem_digest.py (run_pair end to end with 14 structure channels and the solver
on four synthetic levels; numpy and torch64 on the cpu device) computed on the code of a commit BEFORE the MEM memory
changes, for tests/test_bayesreg/solve/test_mem_bitwise.py to compare the current tree against.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_mem.py --ref 9747553 --out tests/test_bayesreg/_golden/golden_9747553_mem.json
    python tests/test_bayesreg/_golden/make_golden_mem.py --ref worktree        # the current files (print only)

How (--ref <commit>): `git archive <commit> bayesreg jd_curl tests/test_bayesreg configs` is extracted into a new
temporary directory; the digest module of THIS tree (_mem_digest.py; it did not exist at the reference) and this
script are copied into it; a child Python with cwd = PYTHONPATH = that directory, one BLAS / OpenMP / MKL / numba
thread and CUDA_VISIBLE_DEVICES empty runs this script with --compute and prints the digests; the directory is removed.
The reference 9747553 is main at facf4b8 (DEF and EBX merged) plus the read-only memory marks of bayesreg.memtrace
(default off; they touch no array), i.e. the code just before the MEM memory changes.  Before the rebase onto facf4b8
the same digests were generated on 6fb91bd (8a4d060 plus the marks: the code of the smoke v2 runs, bayesreg/jd_curl/
configs unchanged since 7c5d97c); both generations gave identical digests (every run passes its options explicitly),
see runs/20261031-mem/result.md.  Nothing in the repository is
written except --out, which must not exist (hard constraint 4).
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
DIGEST_MODULE = "tests/test_bayesreg/solve/_mem_digest.py"
THIS = "tests/test_bayesreg/_golden/make_golden_mem.py"
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                     NUMBA_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                     BAYESREG_MEMTRACE="")


def compute(include_release):
    import numpy as np
    import torch
    from tests.test_bayesreg.solve import _mem_digest as D
    tick = time.perf_counter()
    digests = D.compute(include_release)
    import bayesreg
    return dict(digests=digests, seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(ref):
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    env = dict(os.environ, **SINGLE_THREAD)
    if ref == "worktree":
        commit = None
        env["PYTHONPATH"] = str(REPO)
        out = subprocess.run([sys.executable, "-B", str(REPO/THIS), "--compute", "--release"], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True).stdout
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_mem_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl", "tests/test_bayesreg",
                                      "configs"], cwd=REPO, capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            for rel in (DIGEST_MODULE, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = subprocess.run([sys.executable, "-B", str(root/THIS), "--compute"], cwd=root, env=env,
                                 capture_output=True, text=True, check=True).stdout
    child = json.loads(out.strip().splitlines()[-1])
    return dict(purpose="golden trajectory digests of the MEM gate (memory-only changes must leave every value unchanged)",
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty, digest_module=DIGEST_MODULE,
                digest_module_sha256=_sha256(REPO/DIGEST_MODULE),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg configs (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="9747553", help="commit to extract (default 9747553) or 'worktree'")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--compute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--release", action="store_true", help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.compute:
        print(json.dumps(compute(ns.release)))
        return 0
    if ns.out is not None and Path(ns.out).exists():
        raise SystemExit(f"{ns.out}: exists (nothing is overwritten)")
    record = generate(ns.ref)
    text = json.dumps(record, indent=1, sort_keys=True)+"\n"
    if ns.out is not None:
        with open(ns.out, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
