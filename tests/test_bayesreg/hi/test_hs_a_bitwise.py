"""C23.8-1 (SG): the hs_a path changes only its admission, not its arithmetic.  Way (i) of the contract (revision 6b, minor 9): c027447's bayesreg, jd_curl, experiments
and configs are exported with git archive into tmp_path and run in a child process on the CPU, single-threaded, on the same synthetic members and images; the new code
runs the same in this process through the hold-out path (the subjects on the synthetic holdout_B_v1, the allowlist and the OasisFiles bound to the record).  probs (float32
before storage), probs36 and structure_groups as stored (float16, hs_seg (a)'s float16_keeping_argmax and the grouping), the members in their order and the normalization
must be the same bit for bit; so must the files the whole hold-out entry writes.  No real subject's file is read."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

from bayesreg.seg.unet.groups import group_probabilities

from ._world import ROOT, SUBJECTS, hi, holdout_kwargs, load_p, world

OLD = 'c0274476b4a61ef82952eff5eed459baabb15ced'          # main before SG, the contract's c027447 (bayesreg/, jd_curl/, experiments/hs have no diff from it to 854c708)

CHILD = r'''
import hashlib, json, sys
from pathlib import Path
args = json.loads(sys.argv[1])
root = Path(args["root"]).resolve()
sys.path.insert(0, str(root/"experiments"/"hs"))
import numpy as np
import torch
torch.set_num_threads(1)
import hs_seg
import bayesreg
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import load_segmenter_registry
from bayesreg.seg.unet.data import OasisFiles
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, group_probabilities, load_grouping
from bayesreg.seg.unet.registry import find_member_files
for module in (bayesreg, hs_seg, hs_seg.nx_seg):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
assert Path(DEFAULT_GROUPS_PATH).resolve().is_relative_to(root), DEFAULT_GROUPS_PATH
import bayesreg.io.guard as guard
assert not hasattr(guard, "check_subject"), "this is not the code before SG"

def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

registry = load_segmenter_registry(args["registry"])
allowlist = load_dev_allowlist(args["allowlist"])
files = OasisFiles(root=Path(args["data"]))
grouping = load_grouping(DEFAULT_GROUPS_PATH, "structure13_v1")
available = find_member_files(args["models_root"])
out = {}
for subject in args["subjects"]:
    plan, models, policy = hs_seg.uf_plan(registry, "UF4", subject, allowlist, available)
    probs, meta, norm, members = hs_seg.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=torch.device("cpu"))
    labels = probs.astype(np.float16).argmax(axis=0)
    p16, ties = hs_seg.nx_seg.float16_keeping_argmax(probs, labels)
    g16 = group_probabilities(probs, grouping, dtype=np.float32).astype(np.float16)
    out[subject] = dict(probs=sha(probs), probs_dtype=str(probs.dtype), p16=sha(p16), g16=sha(g16), shapes=[list(p16.shape), list(g16.shape)], ties=int(ties),
                        members=[m["member_sha256"] for m in members], policy=policy, norm=norm, image=meta["sha256"], threads=torch.get_num_threads())
print("RESULT " + json.dumps(out))
'''


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def export_old(target):
    """c027447's bayesreg, jd_curl, experiments and configs into `target` (git archive; nothing of data/, results or runs/)."""
    done = subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', OLD, 'bayesreg', 'jd_curl', 'experiments', 'configs'], capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as archive:
        archive.extractall(target)
    return target


@pytest.fixture
def w(tmp_path, decided, monkeypatch):
    return world(tmp_path, monkeypatch=monkeypatch)


def test_hs_a_through_the_hold_out_path_is_bitwise_the_code_before_sg(w, tmp_path):
    old = export_old(tmp_path/'old')
    assert (old/'bayesreg/io/guard.py').is_file() and not (old/'bayesreg/io/unseal.py').exists()
    args = dict(root=str(old), registry=str(w.registry.path), allowlist=str(w.allowlist_unbound.path), data=str(w.data), models_root=str(w.models_root), subjects=list(SUBJECTS))
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(PYTHONPATH=str(old), CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([sys.executable, '-B', '-c', CHILD, json.dumps(args)], cwd=old, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    before = json.loads(next(line for line in done.stdout.splitlines() if line.startswith('RESULT '))[len('RESULT '):])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        available = hi.member_files(w.registry, w.models_root)
        after = {}
        for subject in SUBJECTS:
            assert w.allowlist.unseal is w.hi_unseal.base and hi.bayes_unseal.files_problem(w.files) is None      # the hold-out path: bound allowlist, bound files
            plan, models, policy = hi.plan_of(w.registry, subject, w.allowlist, available)
            probs, meta, norm, members = hi.hs_a_probabilities(subject, files=w.files, registry=w.registry, models=models, plan=plan, device=torch.device('cpu'))
            p16, g16, ties = hi.stored_arrays(probs, w.grouping)
            assert np.array_equal(g16.view(np.uint16), group_probabilities(probs, w.grouping, dtype=np.float32).astype(np.float16).view(np.uint16))
            after[subject] = dict(probs=sha(probs), probs_dtype=str(probs.dtype), p16=sha(p16), g16=sha(g16), shapes=[list(p16.shape), list(g16.shape)], ties=int(ties),
                                  members=[m['member_sha256'] for m in members], policy=policy, norm=json.loads(json.dumps(norm)), image=meta['sha256'], threads=torch.get_num_threads())
        record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out'))
    finally:
        torch.set_num_threads(threads)
    assert after == before, json.dumps(dict(before=before, after=after), indent=1)
    assert record['passed'] is True and sorted(record['rows']) == sorted(SUBJECTS)
    for subject in SUBJECTS:                     # the files of the whole entry hold the same bits
        assert sha(load_p(w.tmp/'out'/'pred'/subject/'probs36.npz')) == before[subject]['p16']
        assert sha(load_p(w.tmp/'out'/'pred'/subject/'structure_groups.npz')) == before[subject]['g16']
    assert before[SUBJECTS[0]]['threads'] == 1 and len(before[SUBJECTS[0]]['members']) == 4
    assert before[SUBJECTS[0]]['p16'] != before[SUBJECTS[1]]['p16'] and before[SUBJECTS[0]]['image'] != before[SUBJECTS[1]]['image']     # two different inputs, not a constant
