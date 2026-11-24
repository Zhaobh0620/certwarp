"""R4 修订 6（总负责人裁决，2026-11-13 约 8:30 PM CST）的测试：运行从专用的干净检出跑（make-checkout；batch.json 记检出与续跑仓库；单元的 HEAD = 钉的提交、代码根 = 检出），
续跑记录在续跑仓库（main）里核，续跑仓库的新提交不影响运行检出。全部在临时 git 仓库里做，不碰 /home/rmcfz/seq_vp_tag_results，不读真值，不用 GPU，不起单元。"""
import json
import os
from pathlib import Path
import subprocess

import pytest

from .test_r4_rev4 import ch, rc, re_, run_r4, spy  # noqa: F401

GIT = ('-c', 'user.name=r4-test', '-c', 'user.email=r4-test@example.invalid', '-c', 'commit.gpgsign=false')


def git(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *GIT, *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def quiet(message):
    pass


@pytest.fixture()
def world(tmp_path):
    """一个临时的"main"仓库（续跑仓库），一个临时的数据目录，一个临时的结果根；在结果根下建运行检出。"""
    main = tmp_path/'main'
    (main/'experiments').mkdir(parents=True)
    (main/'experiments'/'x.py').write_text('X = 1\n', encoding='utf-8')
    (main/'docs').mkdir()
    (main/'docs'/'a.md').write_text('a\n', encoding='utf-8')
    git(main, 'init', '-q', '-b', 'main')
    git(main, 'add', '-A')
    git(main, 'commit', '-q', '-m', 'first')
    data, results = tmp_path/'data_target', tmp_path/'results'
    data.mkdir()
    results.mkdir()
    assert rc.RESULTS not in results.resolve().parents
    head = git(main, 'rev-parse', 'HEAD')
    doc = run_r4.make_checkout(source=main, parent=results, data_target=data, identity=lambda d: [], log=quiet)
    return dict(main=main, data=data, results=results, head=head, checkout=Path(doc['checkout']), doc=doc, root=tmp_path/'r4')


# ---- make-checkout ------------------------------------------------------------------------------------------------------------------------------
def test_make_checkout_builds_a_detached_clean_checkout_with_the_data_link(world):
    checkout, head = world['checkout'], world['head']
    assert checkout == world['results']/f'r4_code_{head[:8]}' and world['doc']['commit'] == head and world['doc']['problems'] == []
    assert git(checkout, 'rev-parse', 'HEAD') == head and git(checkout, 'status', '--porcelain', '--untracked-files=no') == ''
    assert subprocess.run(['git', '-C', str(checkout), 'symbolic-ref', '-q', 'HEAD'], capture_output=True).returncode != 0          # 游离
    assert (checkout/'data').is_symlink() and Path(os.readlink(checkout/'data')) == world['data']
    assert git(checkout, 'ls-files', 'data') == ''                                                                                      # 不进 git
    assert run_r4.checkout_setup_problems(checkout, resume_repo=world['main'], data_target=world['data']) == []
    with pytest.raises(rc.Refused, match='已在'):
        run_r4.make_checkout(source=world['main'], parent=world['results'], data_target=world['data'], identity=lambda d: [], log=quiet)


def test_make_checkout_records_the_problems_of_a_failed_check_and_keeps_the_checkout(world):
    (world['main']/'docs'/'b.md').write_text('b\n', encoding='utf-8')
    git(world['main'], 'add', '-A')
    git(world['main'], 'commit', '-q', '-m', 'second')
    doc = run_r4.make_checkout(source=world['main'], parent=world['results'], data_target=world['data'], identity=lambda d: ['the frozen tree differs'], log=quiet)
    assert doc['problems'] == ['the frozen tree differs'] and Path(doc['checkout']).is_dir()                                          # 失败证据留着
    with pytest.raises(rc.Refused, match='找不到'):
        run_r4.make_checkout('0'*40, source=world['main'], parent=world['results'], data_target=world['data'], identity=lambda d: [], log=quiet)


def test_the_setup_check_refuses_the_resume_repository_itself_and_a_branch_checkout(world):
    main = world['main']
    (main/'data').symlink_to(world['data'])
    problems = run_r4.checkout_setup_problems(main, resume_repo=main, data_target=world['data'])
    assert any('续跑仓库本身' in p for p in problems) and any('不是游离的' in p for p in problems)
    with pytest.raises(rc.Refused, match='运行检出'):
        run_r4.batch_init(world['root'], 'bench', repo=main, resume_repo=main)
    assert not (world['root']/'batch.json').exists()


# ---- 单元的身份：HEAD = 钉的提交，代码根 = 运行检出 ----------------------------------------------------------------------------------------------------
def test_a_unit_refuses_a_checkout_whose_head_is_not_the_pinned_commit(world, monkeypatch):
    checkout, head, main = world['checkout'], world['head'], world['main']
    monkeypatch.setattr(ch.he, 'identity', lambda repo, **kw: ({}, []))                                     # HE 的 identity 另有测试；这里只看检出与提交
    b = dict(code_commit=head, checkout=str(checkout), data_link=str(world['data']))
    assert run_r4.batch_identity(b, repo=checkout)[0] == []
    assert any('运行检出' in p for p in run_r4.batch_identity(b, repo=main)[0])                              # 同一个提交、但不是检出：拒
    assert any('没有记运行检出' in p for p in run_r4.batch_identity(dict(code_commit=head, data_link=str(world['data'])), repo=checkout)[0])
    (checkout/'experiments'/'x.py').write_text('X = 2\n', encoding='utf-8')
    git(checkout, 'commit', '-q', '-am', 'moved')                                                           # 检出的 HEAD 动了
    problems, record = run_r4.batch_identity(b, repo=checkout)
    assert any('不是 batch.json 钉的' in p for p in problems) and record['commit'] != head


def test_queue_and_single_steps_refuse_to_start_outside_the_checkout(world):
    root = world['root']
    rc.write_json(root/'batch.json', dict(checkout=str(world['checkout']), code_commit=world['head'], resume_repo=str(world['main'])))
    fail = lambda *a, **k: pytest.fail('a unit was started')                                               # noqa: E731
    kw = dict(root=root, launch_unit=fail, gate=lambda own: (True, {}), gap=0, log=quiet)
    assert run_r4.queue_arm('bench', 'CTL', pairs=['a__b'], **kw) == 2                                     # rc.ROOT 不是这份 batch.json 的检出
    assert run_r4.run_step('prepare-bench', [], **kw) == 2 and not (root/run_r4.STEP_DIR).exists()
    assert run_r4.launch_problem(root, 'bench', repo=world['checkout']) is None
    assert '没有' in run_r4.launch_problem(root, 'dev', repo=world['checkout'])                               # 修订 6a M2：没有 dev_batch.json 也拒
    assert run_r4.run_step('devcheck', [], **kw) == 2 and run_r4.queue_arm('dev', 'B', pairs=[], **kw) == 2 and not re_.stop_files(root)


# ---- 续跑记录在续跑仓库里核 ---------------------------------------------------------------------------------------------------------------------------
def resume_doc(stop):
    return dict(schema=re_.RESUME_SCHEMA, handling='jia', signed_by='the lead', scope='candidate', arm='A1', stop_sha256=rc.sha256_file(stop))


def test_resume_records_are_checked_in_the_resume_repository_not_in_the_checkout(world):
    root, main, checkout = world['root'], world['main'], world['checkout']
    rc.write_json(root/'batch.json', dict(checkout=str(checkout), code_commit=world['head'], resume_repo=str(main)))
    assert re_.resume_repo_of(root) == main.resolve()
    stop = re_.write_stop(root, dict(step='pair', arm='A1', scope='candidate', reason='test'))
    assert run_r4.unresolved(root, None, True, log=quiet)[0] is True
    text = json.dumps(resume_doc(stop), ensure_ascii=False)
    (checkout/re_.RESUME_DIR_REL).mkdir(parents=True)
    (checkout/re_.RESUME_DIR_REL/'resume_0.json').write_text(text, encoding='utf-8')                       # 放在运行检出里：不算
    problems, _, _ = re_.resume_state(root, before_lock=False)
    assert problems == [f'{stop.name}：没有续跑记录']
    (main/re_.RESUME_DIR_REL).mkdir(parents=True)
    (main/re_.RESUME_DIR_REL/'resume_0.json').write_text(text, encoding='utf-8')                           # 在续跑仓库里、没提交：拒
    problems, _, _ = re_.resume_state(root, before_lock=False)
    assert problems == ['resume_0.json：续跑记录没有提交或工作树里有改动']
    git(main, 'add', re_.RESUME_DIR_REL)
    problems, _, _ = re_.resume_state(root, before_lock=False)
    assert problems == ['resume_0.json：续跑记录没有提交或工作树里有改动']                                        # 只 add、没 commit：拒
    git(main, 'commit', '-q', '-m', 'resume record')
    problems, candidates, whole = re_.resume_state(root, before_lock=False)
    assert problems == [] and ('candidate', 'A1') in candidates and whole is None
    assert run_r4.unresolved(root, None, True, log=quiet) == (False, candidates)
    (main/re_.RESUME_DIR_REL/'resume_0.json').write_text(text.replace('the lead', 'someone'), encoding='utf-8')   # 工作树里这一份与提交的不一致：拒
    assert re_.resume_state(root, before_lock=False)[0] == ['resume_0.json：续跑记录没有提交或工作树里有改动']


def test_new_commits_in_the_resume_repository_do_not_touch_the_checkout(world, monkeypatch):
    root, main, checkout, head = world['root'], world['main'], world['checkout'], world['head']
    monkeypatch.setattr(ch.he, 'identity', lambda repo, **kw: ({}, []))
    b = dict(code_commit=head, checkout=str(checkout), data_link=str(world['data']))
    (main/'docs'/'a.md').write_text('a, edited\n', encoding='utf-8')
    (main/'runs'/'20261113-r4'/'resume').mkdir(parents=True)
    (main/'runs'/'20261113-r4'/'resume'/'note.json').write_text('{}', encoding='utf-8')
    git(main, 'add', '-A')
    git(main, 'commit', '-q', '-m', 'docs and runs only')
    assert git(main, 'rev-parse', 'HEAD') != head
    assert run_r4.batch_identity(b, repo=checkout)[0] == [] and git(checkout, 'rev-parse', 'HEAD') == head
    assert run_r4.checkout_setup_problems(checkout, resume_repo=main, data_target=world['data']) == []


def test_resume_repositories_of_the_two_batches_must_agree(world, tmp_path):
    root = world['root']
    rc.write_json(root/'batch.json', dict(resume_repo=str(world['main'])))
    rc.write_json(root/'dev_batch.json', dict(resume_repo=str(tmp_path)))
    with pytest.raises(rc.Refused, match='resume_repo 不同'):
        re_.resume_repo_of(root)
    assert re_.resume_repo_of(tmp_path/'nothing') == rc.RESUME_REPO


# ---- spy：评测单元的续跑目录在续跑仓库下；续跑仓库是判定的根 -----------------------------------------------------------------------------------------
def test_the_evaluation_whitelist_names_the_resume_directory_of_the_resume_repository(world, tmp_path):
    main, root = world['main'], world['root']
    rules = spy.bench_rules(root, unit='eval-bench-first', stage_dir=root/'s', pred_root=root/'s', atropos_root=root/'s', extra=(main/re_.RESUME_DIR_REL,),
                            allow_images=False, repo=world['checkout'], resume_repo=main)
    (main/re_.RESUME_DIR_REL).mkdir(parents=True)
    assert rules.judge(main/re_.RESUME_DIR_REL/'resume_0.json')[:2] == (True, True)
    assert rules.judge(main/'docs'/'a.md')[:2] == (True, False)                                              # main 里别的东西：判、不放行
    assert rules.judge(world['checkout']/re_.RESUME_DIR_REL/'resume_0.json')[:2] == (True, False)            # 检出里的续跑目录不再放行
