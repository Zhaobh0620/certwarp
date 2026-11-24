"""Generate the golden digests of the C15-9 "dim bitwise unchanged" gate (EBC2): the functions of
tests/test_bayesreg/learn/_eb_digest.py evaluated on the code of the commit before C15-9 was implemented (7579645),
numpy 2.0.2, one BLAS / OpenMP / numba thread, CUDA_VISIBLE_DEVICES empty.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_eb.py --ref 7579645 \\
        --out tests/test_bayesreg/_golden/golden_7579645_eb.json
    python tests/test_bayesreg/_golden/make_golden_eb.py --ref worktree        # the current files instead

How (--ref <commit>): as make_golden.py -- `git archive <commit> bayesreg jd_curl tests/test_bayesreg` is extracted
into a new temporary directory; the digest module of THIS tree and this script are copied in; the helpers the digest
module imports (model/_helpers.py, infer/_cases.py and the package __init__ files) must be byte-identical between the
commit and this tree (checked; otherwise refused); a child Python with cwd = PYTHONPATH = that directory runs this
script with --compute; the directory is removed.  Nothing in the repository is written except --out, which must not
exist (hard constraint 4).
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
DIGEST_MODULE = "tests/test_bayesreg/learn/_eb_digest.py"
THIS = "tests/test_bayesreg/_golden/make_golden_eb.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/learn/__init__.py",
           "tests/test_bayesreg/model/__init__.py", "tests/test_bayesreg/model/_helpers.py",
           "tests/test_bayesreg/infer/__init__.py", "tests/test_bayesreg/infer/_cases.py")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def compute():
    import numpy as np
    from tests.test_bayesreg.learn import _eb_digest as D
    import bayesreg
    tick = time.perf_counter()
    digests = D.all_digests()
    return dict(digests=digests, seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__,
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
        out = subprocess.run([sys.executable, "-B", str(REPO/THIS), "--compute"], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True).stdout
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_eb_") as tmp:
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
            out = subprocess.run([sys.executable, "-B", str(root/THIS), "--compute"], cwd=root, env=env,
                                 capture_output=True, text=True, check=True).stdout
    child = json.loads(out.strip().splitlines()[-1])
    return dict(purpose="golden digests of the C15-9 dim bitwise gate (EBC2)",
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty, digest_module=DIGEST_MODULE,
                digest_module_sha256=_sha256(REPO/DIGEST_MODULE), helpers=list(HELPERS),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="7579645", help="commit to extract (default 7579645) or 'worktree'")
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
