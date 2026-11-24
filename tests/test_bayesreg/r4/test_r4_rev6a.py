"""R4 修订 6a（修订 6 的小差异复核 0/1/6；总负责人裁决）的测试：I1 续跑仓库只能是检出的同一个仓库的主工作树、真实运行写死；M1 子进程核代码根；M2 没有 batch 文件也拒、
queue --dry-run 打印；M3 data 链接；M4 续跑仓库的 HEAD、分支、记录的 blob；M5 make-checkout 的记录；M6 batch_init 的成功路径、两份 batch 的续跑仓库、每个单元入口都走 batch_identity。
全部在临时 git 仓库里做，不碰 /home/rmcfz/seq_vp_tag_results，不读真值，不用 GPU，不起单元。"""
import ast
import json
import os
from pathlib import Path
import types

import numpy as np
import pytest

from .test_r4_rev4 import ch, rc, re_, run_r4  # noqa: F401
from .test_r4_rev6 import git, quiet, resume_doc, world  # noqa: F401

HERE = Path(__file__).resolve()
R4_DIR = HERE.parents[3]/'experiments'/'r4'


# ---- I1：续跑仓库 -------------------------------------------------------------------------------------------------------------------------------------
def test_an_unrelated_repository_or_a_linked_worktree_is_not_the_resume_repository(world, tmp_path):
    checkout, main, data = world['checkout'], world['main'], world['data']
    other = tmp_path/'other'
    other.mkdir()
    (other/'f.txt').write_text('x', encoding='utf-8')
    git(other, 'init', '-q', '-b', 'main')
    git(other, 'add', '-A')
    git(other, 'commit', '-q', '-m', 'unrelated')
    assert any('不是同一个仓库' in p for p in run_r4.checkout_setup_problems(checkout, resume_repo=other, data_target=data))
    linked = tmp_path/'linked'
    git(main, 'worktree', 'add', '-q', '-b', 'side', str(linked))
    problems = run_r4.checkout_setup_problems(checkout, resume_repo=linked, data_target=data)
    assert any('不是主工作树' in p for p in problems) and not any('不是同一个仓库' in p for p in problems)
    assert run_r4.checkout_setup_problems(checkout, resume_repo=main, data_target=data) == []


def test_a_real_run_takes_only_the_fixed_resume_repository(world, tmp_path, monkeypatch):
    root = world['root']
    monkeypatch.setattr(rc, 'R4_ROOT', root)                                                                # 这个测试的根当作真实的根（不碰真的结果根）
    with pytest.raises(rc.Refused, match='真实运行的续跑仓库只能是'):
        run_r4.batch_init(root, 'dev', repo=world['checkout'], resume_repo=world['main'])
    rc.write_json(root/'batch.json', dict(resume_repo=str(world['main'])))
    with pytest.raises(rc.Refused, match='真实运行的续跑仓库只能是'):
        re_.resume_repo_of(root)
    with pytest.raises(SystemExit):
        run_r4.build_parser().parse_args(['batch-init', '--role', 'bench', '--resume-repo', str(world['main'])])     # 命令行没有 --resume-repo


# ---- M1：子进程核代码根 ---------------------------------------------------------------------------------------------------------------------------------
def test_the_context_names_its_code_root_and_a_child_from_elsewhere_is_refused(tmp_path, monkeypatch):
    assert ch.make_context('bench', 'CTL', 'P__Q', root=tmp_path, stage_dir=tmp_path, check_identity=False, test=True)['code_root'] == str(rc.ROOT.resolve())
    import run_hx
    monkeypatch.setattr(run_hx, 'check_identity', lambda ctx, out, name: ({}, []))
    ran = []
    monkeypatch.setattr(ch, 'run_child_spied', lambda ctx, stage: ran.append(stage) or 0)
    o = types.SimpleNamespace(context='context.json', command='t88')
    for ctx, code, started in ((dict(check_identity=True, out=str(tmp_path/'a'), code_root='/elsewhere'), 8, []),
                               (dict(check_identity=True, out=str(tmp_path/'b'), code_root=str(ch.core.ROOT)), 0, ['t88']),
                               (dict(check_identity=False, test_context=run_hx.TEST_CONTEXT, out=str(tmp_path/'c'), code_root='/elsewhere'), 0, ['t88'])):    # 测试上下文照旧放过
        ran.clear()
        monkeypatch.setattr(ch, 'read_context', lambda path, _ctx=ctx: _ctx)
        assert run_r4.cmd_child(o) == code and ran == started
    assert rc.read_json(tmp_path/'a'/'code_root_t88.json')['code_root'] == '/elsewhere'


# ---- M2：queue --dry-run 打印检出的核对 ------------------------------------------------------------------------------------------------------------------
def test_the_queue_dry_run_prints_the_checkout_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rc, 'R4_ROOT', tmp_path/'r4')
    assert run_r4.main(['queue', '--role', 'bench', '--arm', 'CTL', '--dry-run']) == 2
    assert '运行检出：没有' in capsys.readouterr().out
    rc.write_json(tmp_path/'r4'/'batch.json', dict(checkout=str(rc.ROOT.resolve()), resume_repo=str(tmp_path), pairs=['a__b']))
    monkeypatch.setattr(run_r4, 'machine_gate', lambda own: ([], {}))
    assert run_r4.main(['queue', '--role', 'bench', '--arm', 'CTL', '--dry-run']) == 0
    assert '运行检出：ok' in capsys.readouterr().out


# ---- M3：data 链接 --------------------------------------------------------------------------------------------------------------------------------------
def test_the_unit_identity_checks_the_data_link(world, monkeypatch, tmp_path):
    checkout, head = world['checkout'], world['head']
    monkeypatch.setattr(ch.he, 'identity', lambda repo, **kw: ({}, []))
    b = dict(code_commit=head, checkout=str(checkout), data_link=str(world['data']))
    assert run_r4.batch_identity(b, repo=checkout)[0] == []
    assert any('data_link' in p for p in run_r4.batch_identity(dict(code_commit=head, checkout=str(checkout)), repo=checkout)[0])
    (checkout/'data').unlink()
    (checkout/'data').symlink_to(tmp_path)
    assert any('不是指向 batch.json 记的' in p for p in run_r4.batch_identity(b, repo=checkout)[0])


# ---- M4：续跑仓库的 HEAD、分支与记录的 blob -------------------------------------------------------------------------------------------------------------
def test_the_resume_repository_head_branch_and_blobs_are_recorded(world):
    root, main = world['root'], world['main']
    rc.write_json(root/'batch.json', dict(checkout=str(world['checkout']), code_commit=world['head'], resume_repo=str(main)))
    stop = re_.write_stop(root, dict(step='pair', arm='A1', scope='candidate', reason='test'))
    (main/re_.RESUME_DIR_REL).mkdir(parents=True)
    (main/re_.RESUME_DIR_REL/'resume_0.json').write_text(json.dumps(resume_doc(stop)), encoding='utf-8')
    git(main, 'add', '-A')
    git(main, 'commit', '-q', '-m', 'resume record')
    record = {}
    problems, candidates, _ = re_.resume_state(root, before_lock=False, record=record)
    assert problems == [] and ('candidate', 'A1') in candidates
    assert record['head'] == git(main, 'rev-parse', 'HEAD') and record['symbolic_ref'] == 'refs/heads/main' and record['stops'] == 1
    assert record['records_used'] == {stop.name: dict(record='resume_0.json', blob=git(main, 'rev-parse', f'HEAD:{re_.RESUME_DIR_REL}/resume_0.json'))}
    logged = []
    assert run_r4.unresolved(root, None, True, log=logged.append)[0] is False and any(record['head'] in line for line in logged)
    git(main, 'checkout', '-q', '-b', 'elsewhere')
    problems, _, _ = re_.resume_state(root, before_lock=False)
    assert problems == [f'续跑仓库 {main.resolve()} 的当前分支是 refs/heads/elsewhere，不是 refs/heads/main（修订 6a M4）']


def test_the_development_reading_records_the_resume_repository(tmp_path, monkeypatch):
    pairs = [f'D{k}' for k in range(19)]

    def prelock(*a, notes=None, resume_log=None, **k):
        resume_log.update(head='h', symbolic_ref='refs/heads/main')
        return {p: {} for p in pairs}, {p: np.zeros((2, 2, 2, 3)) for p in pairs}, {}
    monkeypatch.setattr(re_, 'dev_prelock', prelock)
    monkeypatch.setattr(re_, 'run_hx_admit', lambda stage, pair: (None, None))
    monkeypatch.setattr(re_, 'evaluator', lambda: (_ for _ in ()).throw(RuntimeError('stop after the lock')))
    doc = re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p')
    assert doc['resume_repo'] == dict(head='h', symbolic_ref='refs/heads/main')


# ---- M5：make-checkout 的记录 ----------------------------------------------------------------------------------------------------------------------------
def test_make_checkout_writes_its_record_outside_the_checkout(world):
    head, results = world['head'], world['results']
    record = rc.read_json(results/f'r4_code_{head[:8]}.make_checkout.json')
    assert record['commit'] == head and record['checkout'] == str(world['checkout']) and record['problems'] == [] and world['doc']['record'] == str(results/f'r4_code_{head[:8]}.make_checkout.json')
    assert not (world['checkout']/f'r4_code_{head[:8]}.make_checkout.json').exists()
    (world['main']/'docs'/'c.md').write_text('c\n', encoding='utf-8')
    git(world['main'], 'add', '-A')
    git(world['main'], 'commit', '-q', '-m', 'third')
    second = git(world['main'], 'rev-parse', 'HEAD')
    (results/f'r4_code_{second[:8]}.make_checkout.json').write_text('{}', encoding='utf-8')               # 记录已在：拒绝，不建检出
    with pytest.raises(rc.Refused, match='已在'):
        run_r4.make_checkout(source=world['main'], parent=results, data_target=world['data'], identity=lambda d: [], log=quiet)
    assert not (results/f'r4_code_{second[:8]}').exists()
    (results/f'r4_code_{second[:8]}.make_checkout.json').unlink()
    doc = run_r4.make_checkout(source=world['main'], parent=results, data_target=world['data'], identity=lambda d: ['the frozen tree differs'], log=quiet)
    assert rc.read_json(Path(doc['record']))['problems'] == ['the frozen tree differs']                    # 核对不过也写


# ---- M6：batch_init 的成功路径与两份 batch；每个单元入口都走 batch_identity ---------------------------------------------------------------------------------
@pytest.fixture()
def real_link_checkout(world):
    """data 链接指向 rc.DATA_LINK_TARGET 的检出（batch_init 照默认目标核；只建链接，不读目标）。"""
    (world['main']/'docs'/'d.md').write_text('d\n', encoding='utf-8')
    git(world['main'], 'add', '-A')
    git(world['main'], 'commit', '-q', '-m', 'for batch_init')
    doc = run_r4.make_checkout(source=world['main'], parent=world['results'], data_target=rc.DATA_LINK_TARGET, identity=lambda d: [], log=quiet)
    return dict(world, checkout=Path(doc['checkout']), head=doc['commit'])


def test_batch_init_records_the_checkout_and_the_resume_repository(real_link_checkout, monkeypatch):
    w = real_link_checkout
    monkeypatch.setattr(run_r4, 'identity_problems', lambda repo, **kw: ([], run_r4.code_record(repo)))      # HE 的 identity 与 plan.md 另有测试
    monkeypatch.setattr(rc, 'plan_problem', lambda repo=None: None)
    path = run_r4.batch_init(w['root'], 'dev', repo=w['checkout'], resume_repo=w['main'])
    doc = rc.read_json(path)
    main = w['main'].resolve()
    assert doc['checkout'] == str(w['checkout'].resolve()) and doc['code_commit'] == w['head'] and doc['data_link'] == str(rc.DATA_LINK_TARGET)
    assert doc['resume_repo'] == str(main) and doc['resume_dir'] == str(main/re_.RESUME_DIR_REL)
    assert re_.resume_repo_of(w['root']) == main


def test_batch_init_refuses_a_resume_repository_other_than_the_other_batchs(real_link_checkout, monkeypatch, tmp_path):
    w = real_link_checkout
    monkeypatch.setattr(run_r4, 'identity_problems', lambda repo, **kw: ([], run_r4.code_record(repo)))
    monkeypatch.setattr(rc, 'plan_problem', lambda repo=None: None)
    rc.write_json(w['root']/'batch.json', dict(resume_repo=str(tmp_path/'another')))
    with pytest.raises(rc.Refused, match='另一份 batch'):
        run_r4.batch_init(w['root'], 'dev', repo=w['checkout'], resume_repo=w['main'])
    assert not (w['root']/'dev_batch.json').exists()


def calls_of(tree, name):
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, 'id', None) for c in ast.walk(fn) if isinstance(c, ast.Call)}, fn


def first_dict(tree, name):
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    d = next(n for n in ast.walk(fn) if isinstance(n, ast.Dict))
    return {k.value: v.id for k, v in zip(d.keys, d.values)}


def test_every_unit_entry_checks_its_batch_identity():
    run_tree = ast.parse((R4_DIR/'run_r4.py').read_text(encoding='utf-8'))
    steps_tree = ast.parse((R4_DIR/'r4_steps.py').read_text(encoding='utf-8'))
    handlers, table = first_dict(run_tree, 'main'), first_dict(steps_tree, 'dispatch')
    assert 'batch_identity' in calls_of(steps_tree, '_identity')[0]
    for command in ('pair', 'finalize', 'sdlogj'):                                                         # 队列起的单元
        assert 'batch_identity' in calls_of(run_tree, handlers[command])[0], command
    for step in run_r4.STEPS:                                                                               # 单步的单元
        assert {'_identity', 'batch_identity'} & calls_of(steps_tree, table[step])[0], step
    calls, fn = calls_of(run_tree, handlers['t88'])                                                         # T88 与修复子进程：上下文的身份加代码根（M1）
    assert handlers['repair'] == handlers['t88'] and 'check_identity' in calls and any(isinstance(n, ast.Constant) and n.value == 'code_root' for n in ast.walk(fn))
    callers = {f.name for tree in (run_tree, steps_tree) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               and any(isinstance(c, ast.Call) and getattr(c.func, 'attr', getattr(c.func, 'id', None)) == 'identity_problems' for c in ast.walk(f))}
    assert callers == {'batch_identity', 'batch_init', 'cmd_dry_run'}                                     # 没有别的入口绕过 batch_identity
