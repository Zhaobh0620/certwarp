"""Generate the golden digests of the SWLI gate (work package SWLI, 2026-10-31; contract C04-6 gates 1 and 10): the brain
default of run_pair (log rows, numeric fields of hyper.json and qc.json, disp / final_native arrays) and the brain Laplace
samples (tests/test_bayesreg/solve/_swli_digest.py) on the code before SWLI (commit ddcbfce: the SWL contract draft, no
sigma_domain code), numpy 2.0.2, one BLAS / OpenMP / numba thread, CUDA_VISIBLE_DEVICES empty.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_swli.py --ref ddcbfce \\
        --out tests/test_bayesreg/_golden/golden_ddcbfce_swli.json
    python tests/test_bayesreg/_golden/make_golden_swli.py --ref worktree        # the current files instead
    python tests/test_bayesreg/_golden/make_golden_swli.py --ref ddcbfce --keys \\
        --out tests/test_bayesreg/_golden/golden_ddcbfce_swli_keys.json      # the record key sets (review minor 4)

How (--ref <commit>): as make_golden_run_pair.py -- `git archive <commit> bayesreg jd_curl tests/test_bayesreg` is
extracted into a new temporary directory; the digest modules of THIS tree (_swli_digest.py, _run_pair_digest.py) and
this script are copied in; the helpers they import must be byte-identical between the commit and this tree (checked;
otherwise refused); a child Python with cwd = PYTHONPATH = that directory runs this script with --compute; the
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
DIGEST_MODULES = ("tests/test_bayesreg/solve/_swli_digest.py", "tests/test_bayesreg/solve/_run_pair_digest.py")
THIS = "tests/test_bayesreg/_golden/make_golden_swli.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/solve/__init__.py",
           "tests/test_bayesreg/solve/test_run_pair.py", "tests/test_bayesreg/solve/_synthetic.py",
           "tests/test_bayesreg/io_seg/__init__.py", "tests/test_bayesreg/io_seg/_helpers.py",
           "tests/test_bayesreg/infer/__init__.py", "tests/test_bayesreg/infer/_cases.py",
           "tests/test_bayesreg/model/__init__.py", "tests/test_bayesreg/model/_helpers.py")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


KEYS_MODULE = "tests/test_bayesreg/solve/_swli_keys.py"     # --keys (review SWLI minor 4): the record key sets


def compute_keys():
    import numpy as np
    from tests.test_bayesreg.solve import _swli_keys as K
    import bayesreg
    tick = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="golden_swli_keys_") as tmp:
        keys = K.all_key_sets(Path(tmp)/"runs")
    return dict(digests=keys, arms=dict(default=[]), seconds=time.perf_counter()-tick,
                bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


def compute():
    import numpy as np
    from tests.test_bayesreg.solve import _swli_digest as D
    import bayesreg
    tick = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="golden_swli_runs_") as tmp:
        digests = D.all_digests(Path(tmp)/"runs")
    return dict(digests=digests, arms={k: list(v) for k, v in D.ARMS.items() if k in D.GOLDEN_ARMS},
                seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                environment=dict(python=sys.version, numpy=np.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
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


def generate(ref, keys=False):
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    env = dict(os.environ, **SINGLE_THREAD)
    mode = "--compute-keys" if keys else "--compute"
    modules = (*DIGEST_MODULES, KEYS_MODULE) if keys else DIGEST_MODULES
    if ref == "worktree":
        commit = None
        env["PYTHONPATH"] = str(REPO)
        out = _child([sys.executable, "-B", str(REPO/THIS), mode], REPO, env)
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_swli_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl",
                                      "tests/test_bayesreg"], cwd=REPO, capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            differ = [rel for rel in HELPERS if not (root/rel).is_file() or _sha256(root/rel) != _sha256(REPO/rel)]
            if differ:
                raise SystemExit(f"helpers differ between {commit} and this tree: {differ}")
            for rel in (*modules, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = _child([sys.executable, "-B", str(root/THIS), mode], root, env)
    child = json.loads(out.strip().splitlines()[-1])
    purpose = ("key sets of the run_pair records of the brain default (gate 1 measure, review SWLI minor 4: config.json, "
               "provenance.json, hyper.json, qc.json, status.json, timing.json), on the code before SWLI" if keys else
               "golden digests of the SWLI gate (C04-6 gates 1, 10): run_pair brain default (log rows, numeric fields of "
               "hyper.json / qc.json, disp and final_native arrays) and brain Laplace samples, on the code before SWLI")
    return dict(purpose=purpose,
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty,
                digest_modules=list(modules),
                digest_module_sha256={rel: _sha256(REPO/rel) for rel in modules}, helpers=list(HELPERS),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], arms=child["arms"],
                digests=child["digests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="ddcbfce", help="commit to extract (default ddcbfce) or 'worktree'")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--keys", action="store_true", help="the key sets of the records (review SWLI minor 4) instead "
                        "of the digests")
    parser.add_argument("--compute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compute-keys", action="store_true", help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.compute:
        print(json.dumps(compute()))
        return 0
    if ns.compute_keys:
        print(json.dumps(compute_keys()))
        return 0
    if ns.out is not None and Path(ns.out).exists():
        raise SystemExit(f"{ns.out}: exists (nothing is overwritten)")
    record = generate(ns.ref, keys=ns.keys)
    text = json.dumps(record, indent=1, sort_keys=True)+"\n"
    if ns.out is not None:
        with open(ns.out, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
