"""Generate the golden digests of the DEF gate (work package DEF, 2026-10-31): whole bayesreg.cli.run_pair runs of
tests/test_bayesreg/solve/_run_pair_digest.py (arms pre_def, pre_def_off, corners, def_explicit) on the code before DEF
(commit 8a4d060: the contract text of DEF, code defaults gauss27 / gauss27+corners), numpy 2.0.2, one BLAS / OpenMP /
numba thread, CUDA_VISIBLE_DEVICES empty.

Usage (from the repository root; TMPDIR should be a scratch directory):
    python tests/test_bayesreg/_golden/make_golden_run_pair.py --ref 8a4d060 \\
        --out tests/test_bayesreg/_golden/golden_8a4d060_run_pair.json
    python tests/test_bayesreg/_golden/make_golden_run_pair.py --ref worktree        # the current files instead

How (--ref <commit>): as make_golden.py -- `git archive <commit> bayesreg jd_curl tests/test_bayesreg` is extracted
into a new temporary directory; the digest module of THIS tree and this script are copied in; the helpers the digest
module imports (solve/test_run_pair.py, solve/_synthetic.py, io_seg/_helpers.py and the package __init__ files) must
be byte-identical between the commit and this tree (checked; otherwise refused); a child Python with cwd = PYTHONPATH =
that directory runs this script with --compute, which runs the four arms into a temporary directory of its own; the
directories are removed.  Nothing in the repository is written except --out, which must not exist (hard constraint 4).
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
DIGEST_MODULE = "tests/test_bayesreg/solve/_run_pair_digest.py"
THIS = "tests/test_bayesreg/_golden/make_golden_run_pair.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/solve/__init__.py",
           "tests/test_bayesreg/solve/test_run_pair.py", "tests/test_bayesreg/solve/_synthetic.py",
           "tests/test_bayesreg/io_seg/__init__.py", "tests/test_bayesreg/io_seg/_helpers.py")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def compute():
    import numpy as np
    from tests.test_bayesreg.solve import _run_pair_digest as D
    import bayesreg
    tick = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="golden_run_pair_runs_") as tmp:
        digests = D.all_digests(Path(tmp)/"runs")
    return dict(digests=digests, arm_args={k: list(v) for k, v in D.ARM_ARGS.items()}, common_args=list(D.COMMON),
                seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _child(args, cwd, env):
    """Run the --compute child; its stdout, or SystemExit with the end of its stderr (the runs log there)."""
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
        with tempfile.TemporaryDirectory(prefix="golden_run_pair_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl",
                                      "tests/test_bayesreg"], cwd=REPO, capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            differ = [rel for rel in HELPERS if not (root/rel).is_file() or _sha256(root/rel) != _sha256(REPO/rel)]
            if differ:
                raise SystemExit(f"helpers differ between {commit} and this tree: {differ}")
            for rel in (DIGEST_MODULE, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = _child([sys.executable, "-B", str(root/THIS), "--compute"], root, env)
    child = json.loads(out.strip().splitlines()[-1])
    return dict(purpose="golden digests of the DEF gate: run_pair with the old options explicit and with the new "
                        "defaults explicit, on the code before DEF",
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty, digest_module=DIGEST_MODULE,
                digest_module_sha256=_sha256(REPO/DIGEST_MODULE), helpers=list(HELPERS),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], arm_args=child["arm_args"],
                common_args=child["common_args"], digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="8a4d060", help="commit to extract (default 8a4d060) or 'worktree'")
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
