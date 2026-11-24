"""Regenerate the bitwise golden digests of the C05-2a default-path gate (FB1; review LVC_CPUPAR important item 1).

The digests GOLDEN_481E33B of
    tests/test_bayesreg/model/test_logvol_corners.py     model:  small, box12   (numpy model: evaluate, gradient, HVP,
                                                                                stable_reduction)
    tests/test_bayesreg/torch64/test_torch64_corners.py  torch:  small, box12   (torch64 on the cpu device, with the
                                                                                C08-8(a)(b) device comparison)
    tests/test_bayesreg/solve/test_logvol_points.py      solve:  numpy_a, numpy_b, torch64_a   (6 accepted PG steps of
                                                                                the mild 9^3 problem, options (a)/(b))
are the functions of tests/test_bayesreg/model/_logvol_digest.py evaluated on the code of the commit before C05-2a
(481e33b; C05-2a is d0e4ae2), numpy 2.0.2, torch 2.5.1, one BLAS / OpenMP / numba thread, torch device cpu.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --out tests/test_bayesreg/_golden/golden_481e33b.json
    python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --check        # compare with the test constants
    python tests/test_bayesreg/_golden/make_golden.py --ref worktree --check       # the current files instead
Options: --skip-solve (no trajectory digests: they call the CPU certificate, i.e. the numba compilation of
certify_map, ~10-20 s on first use in the extracted tree).

Paths (work package DEF, 2026-10-31): --path pre_def (default) computes the digests with the explicit options of the
default path before DEF (_logvol_digest.path_options: logvol_points "gauss27", jd_cell_estimate "gauss27+corners";
empty on 481e33b, whose only path that is) -- the GOLDEN_481E33B constants; the record of the digest module after the
DEF edit is golden_481e33b_def.json (same digests as golden_481e33b.json, which is kept):
    python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --check \\
        --out tests/test_bayesreg/_golden/golden_481e33b_def.json
--path def_explicit computes them with the defaults since DEF given explicitly (gauss27+closed27, beta_S 1, Simpson;
jd_cell_estimate certificate) -- options the code before DEF already had -- the GOLDEN_DEF constants, generated once on
that code (8a4d060, the commit that changed the contract text, not the code):
    python tests/test_bayesreg/_golden/make_golden.py --path def_explicit --ref 8a4d060 \\
        --out tests/test_bayesreg/_golden/golden_8a4d060_def_explicit.json
--path default computes them with no options, i.e. the code's defaults; since DEF they must equal GOLDEN_DEF (the
tests compute this path in process: test_def_default_*; `--path default --ref worktree --check` does the same here).
--check compares with the constants of the path (GOLDEN_481E33B; GOLDEN_DEF for def_explicit and default; a missing
constant is not compared).

How (--ref <commit>): `git archive <commit> bayesreg jd_curl tests/test_bayesreg` is extracted into a new temporary
directory (TMPDIR); the digest module of THIS tree (_logvol_digest.py; it did not exist at 481e33b) and this script
are copied into it; the helpers the digest module imports (model/_helpers.py, solve/_synthetic.py,
torch64/_cases.py and the package __init__ files) must be byte-identical between the commit and this tree (checked;
otherwise refused); a child Python with cwd = PYTHONPATH = that directory, OPENBLAS/OMP/MKL/NUMBA threads 1 and
CUDA_VISIBLE_DEVICES empty runs this script with --compute and prints the digests; the directory is removed.
Nothing in the repository is written except --out, which must not exist (hard constraint 4).
The record holds the command, the resolved commit, the generator's commit (HEAD of this tree and whether it was
dirty), the environment, the sha256 of the digest module and the seven digests.
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
DIGEST_MODULE = "tests/test_bayesreg/model/_logvol_digest.py"
THIS = "tests/test_bayesreg/_golden/make_golden.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/model/__init__.py",
           "tests/test_bayesreg/model/_helpers.py", "tests/test_bayesreg/solve/__init__.py",
           "tests/test_bayesreg/solve/_synthetic.py", "tests/test_bayesreg/torch64/__init__.py",
           "tests/test_bayesreg/torch64/_cases.py")
TEST_FILES = dict(model="tests/test_bayesreg/model/test_logvol_corners.py",
                  torch="tests/test_bayesreg/torch64/test_torch64_corners.py",
                  solve="tests/test_bayesreg/solve/test_logvol_points.py")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


# ----------------------------------------------------------------------------------------------
# Child: the digests on the code found on sys.path (the extracted commit, or this tree)
# ----------------------------------------------------------------------------------------------

def compute(skip_solve=False, path="pre_def"):
    import numpy as np
    import torch
    from tests.test_bayesreg.model import _logvol_digest as D
    tick = time.perf_counter()
    digests = D.all_digests(path, skip_solve)
    model_options, pg = D.path_options(path)
    import bayesreg
    return dict(digests=digests, seconds=time.perf_counter()-tick, bayesreg_path=str(Path(bayesreg.__file__).parent),
                path_options=dict(model=model_options, pg=None if pg is None else pg.record()),
                environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                 longdouble_nmant=int(np.finfo(np.longdouble).nmant), platform=platform.platform(),
                                 thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))


# ----------------------------------------------------------------------------------------------
# Parent
# ----------------------------------------------------------------------------------------------

def _run(args, **kwargs):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True, **kwargs).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CONSTANTS = dict(pre_def="GOLDEN_481E33B", default="GOLDEN_DEF", def_explicit="GOLDEN_DEF")
PURPOSE = dict(pre_def="golden digests of the C05-2a default-path bitwise gate (FB1; review LVC_CPUPAR item 1); "
                       "since DEF the explicit options of the pre-DEF default path",
               def_explicit="golden digests of the defaults of work package DEF (2026-10-31; contract 8a4d060: "
                            "gauss27+closed27 with beta_S 1 and the Simpson rule, jd_cell_estimate certificate) given "
                            "explicitly, on the code before DEF",
               default="digests of the code's defaults (no options); since DEF equal to GOLDEN_DEF")


def _golden_constants(path="pre_def"):
    import ast
    out = {}
    for kind, rel in TEST_FILES.items():
        for node in ast.parse((REPO/rel).read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == \
                    [CONSTANTS[path]]:
                out[kind] = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    return out


def generate(ref, skip_solve=False, path="pre_def"):
    """The record of one generation (see the module docstring)."""
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    env = dict(os.environ, **SINGLE_THREAD)
    child_args = ["--compute", "--path", path]+(["--skip-solve"] if skip_solve else [])
    if ref == "worktree":
        commit, identical = None, True
        root = REPO
        env["PYTHONPATH"] = str(REPO)
        out = subprocess.run([sys.executable, "-B", str(REPO/THIS), *child_args], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True).stdout
    else:
        commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
        with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
            root = Path(tmp)
            archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl",
                                      "tests/test_bayesreg"], cwd=REPO, capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            identical = all((root/rel).is_file() and _sha256(root/rel) == _sha256(REPO/rel) for rel in HELPERS)
            if not identical:
                differ = [rel for rel in HELPERS if not (root/rel).is_file() or _sha256(root/rel) != _sha256(REPO/rel)]
                raise SystemExit(f"helpers differ between {commit} and this tree: {differ}")
            for rel in (DIGEST_MODULE, THIS):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/rel, root/rel)
            env["PYTHONPATH"] = str(root)
            out = subprocess.run([sys.executable, "-B", str(root/THIS), *child_args], cwd=root, env=env,
                                 capture_output=True, text=True, check=True).stdout
    child = json.loads(out.strip().splitlines()[-1])
    return dict(purpose=PURPOSE[path], path=path, path_options=child["path_options"],
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty,
                digest_module=DIGEST_MODULE, digest_module_sha256=_sha256(REPO/DIGEST_MODULE),
                helpers=list(HELPERS), helpers_identical_to_generator=identical,
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg (temporary directory)"
                if commit else "this working tree", child_bayesreg_path=child["bayesreg_path"],
                environment=child["environment"], seconds=child["seconds"], skip_solve=skip_solve,
                digests=child["digests"], test_files=TEST_FILES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="481e33b", help="commit to extract (default 481e33b) or 'worktree'")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--check", action="store_true", help="compare with the GOLDEN_481E33B constants of the tests")
    parser.add_argument("--skip-solve", action="store_true", help="no trajectory digests (no certificate)")
    parser.add_argument("--path", choices=("pre_def", "def_explicit", "default"), default="pre_def",
                        help="pre_def: the explicit options of the default path before DEF (GOLDEN_481E33B); "
                        "def_explicit: the DEF defaults given explicitly (GOLDEN_DEF); default: the code's defaults")
    parser.add_argument("--compute", action="store_true", help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.compute:
        print(json.dumps(compute(ns.skip_solve, ns.path)))
        return 0
    if ns.out is not None and Path(ns.out).exists():
        raise SystemExit(f"{ns.out}: exists (nothing is overwritten)")
    record = generate(ns.ref, ns.skip_solve, ns.path)
    status = 0
    if ns.check:
        constants = _golden_constants(ns.path)
        mismatch = {kind: {k: dict(test=v, generated=record["digests"][kind].get(k)) for k, v in table.items()
                           if record["digests"][kind].get(k) != v}
                    for kind, table in constants.items() if record["digests"][kind]}
        mismatch = {k: v for k, v in mismatch.items() if v}
        record["check"] = dict(constants=constants, mismatches=mismatch, passed=not mismatch)
        status = 0 if not mismatch else 1
    text = json.dumps(record, indent=1, sort_keys=True)+"\n"
    if ns.out is not None:
        with open(ns.out, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
