"""Golden record of gate 1 of work package SCHI (contract C13 gate of C13-4a / 5a, item 1; 2026-10-31): the measured mask
S and the masked sha256 of the seven run_pair record files, on the code before SCH (commit 11409eb: the SCH contract, no
SCH code), numpy 2.0.2, one BLAS / OpenMP / numba thread, CUDA_VISIBLE_DEVICES empty.

Usage (from the repository root):
    python tests/test_bayesreg/_golden/make_golden_sch.py --ref 11409eb \\
        --out tests/test_bayesreg/_golden/golden_11409eb_sch_records.json
    python tests/test_bayesreg/_golden/make_golden_sch.py --texts-worktree   # the test's half: this tree, one root

How (--ref <commit>): for each of two NEW temporary roots, `git archive <commit> bayesreg jd_curl tests/test_bayesreg`
is extracted into <root>/code (the record module of THIS tree, tests/test_bayesreg/solve/_sch_records.py, and this script
are copied in; the helpers they import must be byte-identical between the commit and this tree, else refused); a child
Python with cwd = PYTHONPATH = <root>/code and the single-thread environment runs every arm on inputs written under
<root>/runs and prints the seven record files.  The two runs of each arm are compared leaf by leaf
(_sch_records.measure): every differing leaf must fall in one of the five contract classes (else it is listed under
"unclassified" and the gate fails), S = the classified patterns + the contract's fixed items; the masked digests of
both runs must agree.  Nothing in the repository is written except --out, which must not exist (hard constraint 4).

Arms (run_pair on the 26^3 synthetic pair of solve/test_run_pair.py, channel source tissue, numpy, option (a)):
  default   --rounds 1,1,1,1 --max-accepted 2                      (the contract's default arm: _run_pair_digest's)
  wall      --rounds 1,1,1,1 --max-accepted 2 --wall-seconds 1e6   (beyond the contract: --wall-seconds without shares)
  nocap_l0  --rounds 1,1,1,1 --max-levels 1                        (beyond the contract: no --max-accepted, the 500 cap)
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
RECORD_MODULE = "tests/test_bayesreg/solve/_sch_records.py"
THIS = "tests/test_bayesreg/_golden/make_golden_sch.py"
HELPERS = ("tests/test_bayesreg/__init__.py", "tests/test_bayesreg/solve/__init__.py",
           "tests/test_bayesreg/solve/test_run_pair.py", "tests/test_bayesreg/solve/_synthetic.py",
           "tests/test_bayesreg/solve/_run_pair_digest.py", "tests/test_bayesreg/solve/_swli_digest.py",
           "tests/test_bayesreg/solve/_swli_keys.py",
           "tests/test_bayesreg/io_seg/__init__.py", "tests/test_bayesreg/io_seg/_helpers.py",
           "tests/test_bayesreg/infer/__init__.py", "tests/test_bayesreg/infer/_cases.py",
           "tests/test_bayesreg/model/__init__.py", "tests/test_bayesreg/model/_helpers.py")
SINGLE_THREAD = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
COMMON = ("--rounds", "1,1,1,1", "--max-accepted", "2")
ARMS = dict(default=COMMON, wall=(*COMMON, "--wall-seconds", "1000000"),
            nocap_l0=("--rounds", "1,1,1,1", "--max-levels", "1"))


def compute_texts(runs_root, arms=tuple(ARMS)):
    """Child side: run the arms under runs_root (new) and return {arm: {file: text}} with the exit codes."""
    from bayesreg.cli import run_pair
    from bayesreg.solve import jd_scaling as jds
    from tests.test_bayesreg.solve import _run_pair_digest as RD
    from tests.test_bayesreg.solve import _sch_records as R
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True)
    base = RD.write_inputs(runs_root)
    texts, codes = {}, {}
    for arm in arms:
        if hasattr(jds, "clear_cell_bounds_cache"):
            jds.clear_cell_bounds_cache()
        out = runs_root/f"run_{arm}"
        codes[arm] = run_pair.main([*base, "--out", str(out), *ARMS[arm]])
        texts[arm] = R.read_texts(out)
    return texts, codes


def _child_main(root, arms):
    import numpy as np
    import bayesreg
    tick = time.perf_counter()
    texts, codes = compute_texts(Path(root)/"runs", arms)
    print(json.dumps(dict(texts=texts, codes=codes, root=str(root), seconds=time.perf_counter()-tick,
                          bayesreg_path=str(Path(bayesreg.__file__).parent),
                          environment=dict(python=sys.version, numpy=np.__version__,
                                           longdouble_nmant=int(np.finfo(np.longdouble).nmant),
                                           platform=platform.platform(),
                                           thread_env={k: os.environ.get(k) for k in SINGLE_THREAD}))))


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _child(args, cwd, env):
    done = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"child exit {done.returncode}:\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def run_commit(commit, root, arms=tuple(ARMS)):
    """One run of the arms on `commit`, everything under the new directory `root` (code in root/code)."""
    code = Path(root)/"code"
    code.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "--format=tar", commit, "bayesreg", "jd_curl", "tests/test_bayesreg"],
                             cwd=REPO, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(code)], input=archive, check=True)
    differ = [rel for rel in HELPERS if not (code/rel).is_file() or _sha256(code/rel) != _sha256(REPO/rel)]
    if differ:
        raise SystemExit(f"helpers differ between {commit} and this tree: {differ}")
    for rel in (RECORD_MODULE, THIS):
        (code/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/rel, code/rel)
    env = dict(os.environ, **SINGLE_THREAD, PYTHONPATH=str(code))
    return _child([sys.executable, "-B", str(code/THIS), "--child", str(root), "--arms", ",".join(arms)], code, env)


def run_worktree(root, arms=tuple(ARMS)):
    """One run of the arms on this working tree (the test's half), inputs under root/runs."""
    env = dict(os.environ, **SINGLE_THREAD, PYTHONPATH=str(REPO))
    return _child([sys.executable, "-B", str(REPO/THIS), "--child", str(root), "--arms", ",".join(arms)], REPO, env)


def generate(ref):
    sys.path.insert(0, str(REPO))
    from tests.test_bayesreg.solve import _sch_records as R
    head = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
    children = []
    for tag in ("a", "b"):
        with tempfile.TemporaryDirectory(prefix=f"golden_sch_{tag}_") as tmp:
            children.append(run_commit(commit, Path(tmp)/"root"))
    roots = tuple(child["root"] for child in children)
    arms = {}
    for arm in ARMS:
        texts_a, texts_b = children[0]["texts"][arm], children[1]["texts"][arm]
        S, measured, unclassified, dig_a, dig_b = R.measure(texts_a, texts_b, roots)
        arms[arm] = dict(args=list(ARMS[arm]), exit_codes=[c["codes"][arm] for c in children], S=S,
                         measured=measured, unclassified=unclassified, digests=dig_a, digests_b=dig_b,
                         digests_agree=dig_a == dig_b, roundtrip_exact=R.roundtrip_exact(texts_a),
                         key_sets=R.key_sets(texts_a), solve_quantities_in_S=R.solve_quantities_in(S))
    return dict(purpose="SCHI gate 1 (C13 SCH gate item 1, second and third items): measured mask S and masked "
                        "sha256 of the seven record files, on the code before SCH",
                command=" ".join(["python", THIS, *sys.argv[1:]]),
                generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ref=ref, ref_commit=commit, generator_head=head, generator_dirty=dirty,
                record_module=RECORD_MODULE, record_module_sha256=_sha256(REPO/RECORD_MODULE), helpers=list(HELPERS),
                code_extracted="git archive <ref_commit> bayesreg jd_curl tests/test_bayesreg, once per root",
                roots=list(roots), child_bayesreg_paths=[c["bayesreg_path"] for c in children],
                environment=children[0]["environment"], seconds=[c["seconds"] for c in children],
                gate_failed=any(a["unclassified"] or not a["digests_agree"] for a in arms.values()),
                arms=arms)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="11409eb", help="commit before SCH (default 11409eb)")
    parser.add_argument("--out", default=None, help="write the record here (must not exist)")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arms", default=",".join(ARMS), help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    if ns.child is not None:
        _child_main(ns.child, tuple(ns.arms.split(",")))
        return 0
    if ns.out is not None and Path(ns.out).exists():
        raise SystemExit(f"{ns.out}: exists (nothing is overwritten)")
    record = generate(ns.ref)
    text = json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False)+"\n"
    if ns.out is not None:
        with open(ns.out, "x", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps(dict(gate_failed=record["gate_failed"],
                          arms={k: dict(S=v["S"], unclassified=v["unclassified"], digests_agree=v["digests_agree"],
                                        solve_quantities_in_S=v["solve_quantities_in_S"])
                                for k, v in record["arms"].items()}), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
