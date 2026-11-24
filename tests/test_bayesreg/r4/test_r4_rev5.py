"""R4 修订 5（修订 4 的小差异复核 0/0/4；总负责人裁决）的测试：第 1 条逻辑读绑定的两个反例（只开别人的 aligned_seg35；只开本人的 aligned_seg4）；
第 2 条开发集逻辑读绑定本对的 pair.npz；第 3 条 UnitSpy.__enter__ 失败时全部还原；第 4 条撕裂的锁行。合成数据，不读真值，不用 GPU，不起单元。"""
import builtins
import io
import sys
import types
import zipfile

import numpy as np
import pytest

from .test_r4_rev3 import failing_shim, judge_tmp  # noqa: F401
from .test_r4_rev4 import reader  # noqa: F401
from .test_r4_rev4 import lk, rc, re_, run_r4, spy, r4_steps  # noqa: F401


# ---- 第 1 条：一次逻辑读开错了文件 ------------------------------------------------------------------------------------------------------------------
def test_a_logical_read_that_opens_only_another_subjects_truth_is_caught(reader):
    module, rules, readers = reader
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=readers) as s:
        module.read('S1', only=('S2', 'seg35'))                                                   # 键 (S1, seg35)，只开了 S2/aligned_seg35
    problems = spy.truth_reads(s.record(), expected_keys=[('S1', 'seg35')])
    assert problems and all('not exactly S1/aligned_seg35' in p for p in problems)


def test_a_logical_read_that_opens_only_its_own_other_kind_is_caught(reader):
    module, rules, readers = reader
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=readers) as s:
        module.read('S1', 'seg35', only=('S1', 'seg4'))                                           # 键 (S1, seg35)，只开了 S1/aligned_seg4
    problems = spy.truth_reads(s.record(), expected_keys=[('S1', 'seg35')])
    assert problems and all('not exactly S1/aligned_seg35' in p for p in problems)


# ---- 第 2 条：开发集的逻辑读绑定本对 -------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def dev_reader(tmp_path):
    for pair in ('A__B', 'C__D'):
        (tmp_path/pair).mkdir()
        np.savez(tmp_path/pair/'pair.npz', fixed=np.zeros(2), fixed_seg35=np.zeros(2), moving_seg35=np.zeros(2))
    fake = types.ModuleType('_mc0_evaluate')

    def truth_labels(pair, other=None):
        with np.load(tmp_path/pair/'pair.npz') as z:
            labs = {k: np.array(z[k]) for k in ('fixed_seg35', 'moving_seg35')}
        if other:
            with np.load(tmp_path/other/'pair.npz') as z:
                labs['moving_seg35'] = np.array(z['moving_seg35'])                                 # 在 A__B 的逻辑读里读了 C__D 的成员
        return labs, {}, 'sha'
    fake.truth_labels = truth_labels
    saved = sys.modules.get('_mc0_evaluate')
    sys.modules['_mc0_evaluate'] = fake
    yield fake, judge_tmp(spy.dev_rules(tmp_path/'r4', unit='eval-dev-1'), tmp_path)
    if saved is None:
        del sys.modules['_mc0_evaluate']
    else:
        sys.modules['_mc0_evaluate'] = saved


def test_a_development_logical_read_reads_members_of_its_own_pair_only(dev_reader):
    fake, rules = dev_reader
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
        sys.modules['_mc0_evaluate'].truth_labels('A__B')
        sys.modules['_mc0_evaluate'].truth_labels('C__D')
    record = s.record()
    assert spy.truth_reads(record, expected_keys=[('A__B',), ('C__D',)]) == []
    ids = {tuple(row['key']): row['id'] for row in record['truth_logical']}
    assert [m['logical'] for m in record['truth_members']] == [ids[('A__B',)]]*2+[ids[('C__D',)]]*2      # 成员记录记着它属于哪一次逻辑读
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
        sys.modules['_mc0_evaluate'].truth_labels('A__B', other='C__D')
    problems = spy.truth_reads(s.record(), expected_keys=[('A__B',)])
    assert len(problems) == 1 and 'C__D' in problems[0] and 'not a member of <…>/A__B/pair.npz' in problems[0]


# ---- 第 3 条：__enter__ 失败时全部还原 ------------------------------------------------------------------------------------------------------------------
def test_a_failed_enter_restores_open_and_the_npz_member_access(tmp_path):
    before = (builtins.open, io.open, np.load, np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open)
    rules = judge_tmp(spy.dev_rules(tmp_path/'r4', unit='eval-dev-1'), tmp_path)
    s = spy.UnitSpy(rules, expect_reads=False, truth_readers=(('_r4_rev5_no_such_reader', 'read', lambda a, k: (a[0],)),))
    with pytest.raises(ModuleNotFoundError):
        with s:
            pass                                                                                    # 不会到这里：读取器模块不存在，__enter__ 在装 NpzFile 的包装之后失败
    assert (builtins.open, io.open, np.load, np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open) == before
    assert s.patched == []


# ---- 第 4 条：撕裂的锁行 -------------------------------------------------------------------------------------------------------------------------------
def test_a_torn_second_lock_line_counts_as_after_the_lock(tmp_path, failing_shim):
    from ._batch import batch_here
    root = batch_here(tmp_path/'r4')
    lock = lk.lock_path(root, lk.BENCH)
    lk.take_first(lock, arms=rc.ARM_ORDER, code_commit='c', plan_sha256='p', reason='第一读')
    extra = ['--read', 'second', '--combo', 'COMBO-B-C1']
    with open(lock, 'ab') as f:
        f.write(b'\n  \n')                                                                          # 空行不算
    assert run_r4.lock_taken(root, 'eval-bench', extra) is False
    with open(lock, 'ab') as f:
        f.write('{"read": "second", "reason": "第二读'.encode('utf-8')[:-2])                       # 第二行写到一半，断在多字节字符中间
    with pytest.raises(Exception):
        lk.lines(lock)                                                                              # 解析会失败；lock_taken 不解析
    assert run_r4.lock_taken(root, 'eval-bench', extra) is True
    kw = dict(root=root, gate=lambda own: (True, {}), gap=0, repo=root, require_committed=False)
    assert run_r4.run_step('eval-bench', extra, **kw) == 9
    stop = rc.read_json(re_.stop_files(root)[0])
    assert stop['after_lock'] is True and '建锁之后' in stop['reason'] and stop['arm'] == 'COMBO-B-C1'
