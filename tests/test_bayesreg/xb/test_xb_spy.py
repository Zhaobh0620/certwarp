"""XB's spies (prereg §2.3 (e)): the policy's classes, the driver's audit-hook spy (positive controls: it does see the original, a label file, an unlisted path), the method process's spyrun (a real
subprocess under sci310: a forbidden open stops it even inside `except Exception`), the /proc fd sampler.  Synthetic files in tmp_path only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import _world as W

hc, xc, xb_spy = W.hc, W.xc, W.xb_spy
import xb_policy  # noqa: E402


def method_policy(work):
    return xb_spy.method_policy(name='test', work=work, code_root=xc.REPO, env_prefix=None, baselines=[], results=hc.RESULTS, main_repo=hc.MAIN_REPO, holdout_root=hc.HOLDOUT_ROOT)


def test_the_policy_classes(tmp_path):
    work = hc.RESULTS/'work'/'a'/'p'
    pol = xb_policy.Policy(method_policy(work))
    assert pol.classify(str(work/'inputs.npz'))[0] == 'allowed'
    assert pol.classify(str(work/'disp.npz'), write=True)[0] == 'allowed'
    assert pol.classify(str(hc.RESULTS/'other'/'disp.npz'))[0] == 'forbidden'                            # the results root but its own work directory
    assert pol.classify(str(hc.STAGE_ROOT/'x'/'manifest.json'))[0] == 'forbidden'
    assert pol.classify(str(work/'pair.npz'))[0] == 'forbidden'                                           # by name, even in its work directory
    assert pol.classify(str(work/'aligned_seg35.nii.gz'))[0] == 'forbidden'
    assert pol.classify(str(work/'evaluation.json'))[0] == 'forbidden'
    assert pol.classify('/mnt/d/data/OASIS_OAS1_0001_MR1/aligned_norm.nii.gz')[0] == 'forbidden'
    assert pol.classify('/home/rmcfz/code/python_projects/seq_vp_3d/results/x.json')[0] == 'forbidden'
    assert pol.classify(str(xc.REPO/'bayesreg'/'eval'/'truth.py'))[0] == 'allowed'                      # code: no name rule
    assert pol.classify('/home/rmcfz/somewhere_else.txt')[0] == 'unlisted'
    assert pol.classify(str(xc.REPO/'x.txt'), write=True)[0] == 'unlisted'                               # the code tree is read only


def test_the_driver_spy_sees_the_original_a_label_file_and_an_unlisted_path(tmp_path):
    original = W.write_stage_pair(tmp_path/'stage', W.DEV_PAIR)
    label = tmp_path/'stage'/'aligned_seg35.nii.gz'
    label.write_bytes(b'x')
    other = tmp_path/'other.txt'
    other.write_text('x')
    policy = xb_spy.driver_policy(work=tmp_path/'work', record_dir=tmp_path/'rec', lock_dir=tmp_path/'locks', code_root=xc.REPO, results=hc.RESULTS, main_repo=hc.MAIN_REPO,
                                  holdout_root=hc.HOLDOUT_ROOT, extra_exact=[original.parent/'manifest.json'])
    with xb_spy.DriverSpy(policy, original=original) as spy:
        with pytest.raises(xb_spy.Breach):
            open(original, 'rb').close()
        with spy.reading():
            xc.read_image_members(original)
        with pytest.raises(xb_spy.Breach):
            label.read_bytes()
        with pytest.raises(xb_spy.Breach):
            open(other).close()
        (original.parent/'manifest.json').read_text()
    record = spy.record(expect_original=True)
    assert not record['passed'] and record['original_opens_inside_reader'] == 1 and len(record['breaches']) == 3
    with xb_spy.DriverSpy(policy, original=original) as quiet:
        with quiet.reading():
            xc.read_image_members(original)
    assert quiet.record(expect_original=True)['passed'] and not quiet.record(expect_original=False)['passed']
    assert xb_spy.active() is None


def run_spyrun(tmp_path, body, policy):
    work = tmp_path/'work'
    work.mkdir(exist_ok=True)
    script = tmp_path/'method.py'
    script.write_text(body)
    (work/'policy.json').write_text(json.dumps(policy))
    argv = xb_spy.spyrun_argv(xc.PY, work/'policy.json', work/'spy.jsonl', script, [str(work/'in.txt')])
    env = dict(os.environ, BL2_T_LAUNCH=repr(time.time()))
    done = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=work, timeout=120)
    return done, xb_spy.method_spy_record(work/'spy.jsonl', expect=[work/'in.txt'])


def test_spyrun_lets_a_clean_method_run_and_stops_a_forbidden_open(tmp_path):
    work = tmp_path/'work'
    work.mkdir()
    (work/'in.txt').write_text('fixed and moving')
    (work/'aligned_seg35.nii.gz').write_bytes(b'labels')
    policy = method_policy(work)
    done, record = run_spyrun(tmp_path, 'import sys\nopen(sys.argv[1]).read()\nopen("out.txt", "w").write("ok")\n', policy)
    assert done.returncode == 0 and record['passed'] and record['positive_condition'] and (work/'out.txt').exists()
    (work/'spy.jsonl').unlink()
    body = 'import sys\nopen(sys.argv[1]).read()\ntry:\n    open("aligned_seg35.nii.gz", "rb").read()\nexcept Exception:\n    pass\nopen("after.txt", "w").write("not reached")\n'
    done, record = run_spyrun(tmp_path, body, policy)
    assert done.returncode == xb_spy.SPYRUN_BREACH_EXIT and record['breaches'] and not record['passed'] and not (work/'after.txt').exists()
    (work/'spy.jsonl').unlink()
    done, record = run_spyrun(tmp_path, 'open("/home/rmcfz/xb_spy_test_no_such_file")\n', policy)       # unlisted: stopped before the open is tried
    assert done.returncode == xb_spy.SPYRUN_BREACH_EXIT and record['breaches'][0]['cls'] == 'unlisted'


def test_a_spy_that_saw_nothing_does_not_pass(tmp_path):
    record = xb_spy.method_spy_record(tmp_path/'none.jsonl', expect=[tmp_path/'in.txt'])
    assert not record['passed'] and not record['positive_condition']


def test_the_fd_sampler_sees_an_open_file_of_the_session(tmp_path):
    held = tmp_path/'aligned_seg4.nii.gz'
    held.write_bytes(b'x')
    child = subprocess.Popen([str(xc.PY), '-I', '-B', '-c', f'import time\nf = open({str(held)!r}, "rb")\ntime.sleep(20)'], start_new_session=True)
    try:
        policy = method_policy(tmp_path/'work')
        sampler = xb_spy.FdSampler(policy, child.pid)
        for _ in range(50):
            new = sampler.sample()
            if new:
                break
            time.sleep(.1)
        assert new and new[0]['path'] == os.path.realpath(held) and new[0]['cls'] == 'forbidden'
        sampler.kill()
        child.wait(timeout=10)
        assert child.returncode == -9 and not sampler.record()['passed']
    finally:
        if child.poll() is None:
            child.kill()
