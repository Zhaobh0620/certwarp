"""Generate the golden digests of the JDTI gate 1 (contract C09.1-3a certificate_bernstein, gate 1): short run_pg on the
"mid" / "off" edge-dip quadratics with the real certificate (jd_cell_estimate certificate, closed27, gauss27+corners)
and the outer solvers of options (a) and (b) with certificate (tests/test_bayesreg/solve/_jdti_digest.py), on the code
before JDTI (main 96c8d35), numpy 2.0.2, one BLAS / OpenMP / numba thread, CUDA_VISIBLE_DEVICES empty.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_jdti.py --ref 96c8d35 \\
        --out tests/test_bayesreg/_golden/golden_96c8d35_jdti.json
    python tests/test_bayesreg/_golden/make_golden_jdti.py --ref worktree        # the current files instead

How (--ref <commit>): as make_golden_run_pair.py -- `git archive <commit> bayesreg jd_curl tests/test_bayesreg` into a
new temporary directory; the digest module and this script of THIS tree are copied in; the helper files the run
helpers import must be byte-identical between the commit and this tree, and the run helpers of
solve/test_jd_cell_estimate.py (whose test functions JDTI touched) must have the same source text (checked by ast;
otherwise refused); a child Python with cwd = PYTHONPATH = that directory computes the digests.  Nothing in the
repository is written except --out, which must not exist (hard constraint 4).
"""
import argparse
import ast
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
DIGEST_MODULE = "tests/test_bayesreg/solve/_jdti_digest.py"
THIS = "tests/test_bayesreg/_golden/make_golden_jdti.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/solve/__init__.py",
           "tests/test_bayesreg/solve/_edge_dip.py", "tests/test_bayesreg/solve/_synthetic.py",
           "tests/test_bayesreg/solve/test_jd_scaling.py")
RUN_HELPERS_FILE = "tests/test_bayesreg/solve/test_jd_cell_estimate.py"
RUN_HELPERS = ("_hex", "Point", "Reduction", "Identity", "Quadratic", "dip_problem", "run_quadratic", "run_outer")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def helper_sources(path):
    """{name: source text} of the run helpers (top-level defs / classes) and the module's import lines."""
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in RUN_HELPERS:
            out[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.setdefault("imports", []).append(ast.get_source_segment(text, node))
    return out


def compute():
    import numpy as np
    import bayesreg
    from bayesreg.solve import JDScalingConfig, PGConfig
    from bayesreg.solve import jd_scaling as jds
    from tests.test_bayesreg.solve import _jdti_digest as D
    from tests.test_bayesreg.solve import test_jd_cell_estimate as T
    tick = time.perf_counter()
    digests = D.golden_cases(T, jds, PGConfig, JDScalingConfig)
    return dict(digests=digests, seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _child(args, cwd, env):
    done = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"child exit {done.returncode}:\n{done.stderr[-4000:]}")
    return done.stdout


def generate(ref):
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    env = dict(os.environ, **SINGLE_THREAD)
    if ref == "worktree":
        commit = None
        env["PYTHONPATH"] = str(REPO)
        out = _child([sys.executable, "-B", str(REPO/THIS), "--compute"], REPO, env)
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_jdti_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl",
                                      "tests/test_bayesreg"], cwd=REPO, capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            differ = [rel for rel in HELPERS if not (root/rel).is_file() or _sha256(root/rel) != _sha256(REPO/rel)]
            if differ:
                raise SystemExit(f"helpers differ between {commit} and this tree: {differ}")
            if helper_sources(root/RUN_HELPERS_FILE) != helper_sources(REPO/RUN_HELPERS_FILE):
                raise SystemExit(f"the run helpers of {RUN_HELPERS_FILE} differ between {commit} and this tree")
            for rel in (DIGEST_MODULE, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = _child([sys.executable, "-B", str(root/THIS), "--compute"], root, env)
    child = json.loads(out.strip().splitlines()[-1])
    return dict(purpose="golden digests of the JDTI gate 1: the existing jd_cell_estimate options on the code before "
                        "JDTI",
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty, digest_module=DIGEST_MODULE,
                digest_module_sha256=_sha256(REPO/DIGEST_MODULE), helpers=list(HELPERS),
                run_helpers=dict(file=RUN_HELPERS_FILE, names=list(RUN_HELPERS)),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="96c8d35", help="commit to extract (default 96c8d35) or 'worktree'")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--compute", action="store_true", help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.compute:
        print(json.dumps(compute()))
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
