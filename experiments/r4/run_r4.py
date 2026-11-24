"""R4（runs/20261113-r4/plan.md；合同 C25）的入口与队列。PYTHONPATH=<frozen>:<code>，从干净检出跑（C25.9-6）；单元名一律 tag-r4-*。

    run_r4.py dry-run [--role bench|dev]           代码身份、plan.md 的 sha256、钉住的文件、台的 16 人与 16 对（只读 id）、台的登记；开发集：19 对的准入（manifest）。不读体素、不读真值、不用 GPU、不写文件。
    run_r4.py make-checkout [--commit SHA]         【修订 6】在 main 仓库里跑：git worktree add --detach <RESULTS>/r4_code_<短 sha> <提交>，补 data 符号链接，核冻结树与代码路径干净
    run_r4.py batch-init --role bench|dev
                                                   【修订 6】在运行检出里跑。写 <R4>/batch.json（或 dev_batch.json）：提交、运行检出、续跑仓库、plan.md、folds、台的登记与附属文件、配对表的 sha256（只写一次）；
                                                   之后所有单元都从这个检出起（HEAD = 钉的提交）；续跑记录在续跑仓库（默认 main）里核，main 可以继续提交
    run_r4.py prepare-bench                        台的 stage（C25.6-4、-5）                                        单元 tag-r4-prep
    run_r4.py atropos-bench                        16 人的 Atropos（C25.6-9）                                        单元 tag-r4-atropos
    run_r4.py seg-bench [--device cuda]            16 人、本折 UF2 的推断（C25.6-8；GPU）                           单元 tag-r4-seg
    run_r4.py smoke [--device cuda]                GPU 冒烟（C25.10-7）：台 A 折第 0 对，CTL 与六个候选各一次 T88，不修复；CTL、A1、A2 的 locate_cells 与 certify_map
    run_r4.py devcheck [--device cuda]             0438-0439 选项全关的 T88，field_0600.npz 的 sha256 对 hs_v1 SHx24（C25.10-2）
    run_r4.py queue --role bench|dev --arm ARM [--gap-seconds 10] [--wait-minutes 30] [--dry-run]
                                                   一臂：每对一个 tag-r4- 单元（supervisor），再定稿单元；开发集再加 SDlogJ 单元。失败即停、不重跑；<R4>/hold 拦下一步
    run_r4.py pair --role --arm --pair [--device cuda]      一对一臂（supervisor）；t88 / repair --context FILE 是它的子进程
    run_r4.py finalize --role --arm               定稿单元（复核证书；第三次认证 → 退路）
    run_r4.py sdlogj --arm ARM                     开发集可行性：官方 SDlogJ（不读标签），可行性的判定
    run_r4.py eval-bench --read first|second [--combo NAME] [--dry-run]     台的评测（锁前核对；--dry-run 只做锁前核对）
    run_r4.py decide-bench                         bench_decision.json（筛门、排序、组合、顺序 O）
    run_r4.py next-dev                             开发集的下一步（顺序 O、可行性、读数）
    run_r4.py precheck-hs                          HS 的 19 个输出重建，sha256 对 repaired_u_sha256（不读真值；总负责人在干跑时跑）
    run_r4.py eval-dev --read 1|2 --arm ARM [--dry-run]                     开发集读数（--dry-run 只做建锁之前的清单）
    run_r4.py decide-dev --read 1|2
    run_r4.py unit STEP [OPTIONS]                   单步命令（prepare-bench、atropos-bench、seg-bench、smoke、devcheck、eval-bench、eval-dev）在它自己的单元 tag-r4-<步> 里：
                                                   与一臂的队列共用机器门（等满 30 分钟就停）、hold、单元间隔 ≥ 10 s；同一步起过一次就不再起；失败写 stop
退出码：0 完成；9 读真值之前的失败（队列停下，交总负责人与 BZ）；8 代码身份或核对不符；3 机器门关满 30 分钟或 hold；2 用法或拒绝。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r4_common as rc  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SCRIPT = HERE/'run_r4.py'
UNIT_MEMORY = ('--property=MemoryMax=12G', '--property=MemorySwapMax=4G', '--property=OOMPolicy=continue')
THREAD_ENV = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
HOLD = 'hold'


# ---- 代码身份（C25.9-6；C24 的做法） -----------------------------------------------------------------------------------------------------------
def code_record(repo=rc.ROOT):
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    head, status = git('rev-parse', 'HEAD'), git('status', '--porcelain', '--untracked-files=no')
    return dict(commit=head.stdout.strip() or None, clean=head.returncode == 0 and status.returncode == 0 and not status.stdout.strip(), root=str(repo))


def identity_problems(repo=rc.ROOT, *, want_commit=None, want_checkout=None, want_data_link=None):
    """HE 的 identity（PYTHONPATH = <frozen>:<code>、代码路径已提交且干净、冻结树是 eb808be 的）加：HEAD 是 batch.json 钉的提交；给了 want_checkout 时代码根就是它（修订 6）。"""
    sys.path.insert(0, str(HERE.parent/'he'))
    import he_core
    _, problems = he_core.identity(repo)
    extra, record = checkout_problems(repo, want_commit=want_commit, want_checkout=want_checkout, want_data_link=want_data_link)
    return list(problems)+extra, record


def checkout_problems(repo=rc.ROOT, *, want_commit=None, want_checkout=None, want_data_link=None):
    """(problems, code_record)：HEAD 是钉的提交；代码根是 batch.json 记的运行检出（修订 6）；data 是指向 batch.json 记的目标的符号链接（修订 6a M3）。
    不跑 HE 的 identity，测试在临时仓库里用。"""
    record = code_record(repo)
    problems = []
    link = Path(repo)/'data'
    if want_data_link is not None and (not link.is_symlink() or os.readlink(link) != want_data_link):
        problems.append(f'{link} 不是指向 batch.json 记的 {want_data_link} 的符号链接')
    if want_commit is not None and record['commit'] != want_commit:
        problems.append(f"HEAD {record['commit']} 不是 batch.json 钉的 {want_commit}")
    if want_checkout is not None and Path(repo).resolve() != Path(want_checkout).resolve():
        problems.append(f'代码根 {repo} 不是 batch.json 记的运行检出 {want_checkout}（修订 6：单元从运行检出起）')
    return problems, record


def batch_identity(b, repo=rc.ROOT):
    """一个单元对它的 batch.json 的身份核对（修订 6）：identity_problems，HEAD = code_commit，代码根 = checkout。"""
    return identity_problems(repo, want_commit=b['code_commit'], want_checkout=b.get('checkout') or '<batch.json 没有记运行检出>',
                             want_data_link=b.get('data_link') or '<batch.json 没有记 data_link>')


# ---- 运行检出（修订 6；C25.9-6"从同一个提交的干净检出跑"，R2 的 source_repo 先例） -----------------------------------------------------------------
def git_out(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def checkout_setup_problems(repo, *, resume_repo, data_target=rc.DATA_LINK_TARGET):
    """运行检出本身的样子：不是续跑仓库；HEAD 游离（git worktree add --detach）；data 是指向 data_target 的符号链接、不在 git 里；已跟踪的文件干净；续跑仓库是一个 git 仓库的顶层。"""
    repo, problems = Path(repo), []
    if repo.resolve() == Path(resume_repo).resolve():
        problems.append(f'{repo} 是续跑仓库本身：要在 make-checkout 建的运行检出里跑（修订 6）')
    if git_out(repo, 'symbolic-ref', '-q', 'HEAD')[0] == 0:
        problems.append(f'{repo} 的 HEAD 在分支上，不是游离的（make-checkout 用 git worktree add --detach）')
    link = repo/'data'
    if not link.is_symlink() or Path(os.readlink(link)) != Path(data_target):
        problems.append(f'{link} 不是指向 {data_target} 的符号链接')
    if git_out(repo, 'ls-files', '--error-unmatch', '--', 'data')[0] == 0:
        problems.append('data 在 git 里')
    code = code_record(repo)
    if not code['clean']:
        problems.append(f'{repo} 的已跟踪文件不干净')
    rc_t, top, _ = git_out(resume_repo, 'rev-parse', '--show-toplevel')
    if rc_t or Path(top).resolve() != Path(resume_repo).resolve():
        problems.append(f'续跑仓库 {resume_repo} 不是一个 git 仓库的顶层')
    common = {name: git_out(where, 'rev-parse', '--path-format=absolute', '--git-common-dir') for name, where in (('checkout', repo), ('resume', resume_repo))}
    git_dir = git_out(resume_repo, 'rev-parse', '--path-format=absolute', '--git-dir')
    if any(c[0] for c in common.values()) or Path(common['checkout'][1]).resolve() != Path(common['resume'][1]).resolve():          # 修订 6a I1
        problems.append(f"续跑仓库 {resume_repo} 与检出 {repo} 不是同一个仓库（--git-common-dir {common['resume'][1]} 对 {common['checkout'][1]}）")
    if git_dir[0] or common['resume'][0] or Path(git_dir[1]).resolve() != Path(common['resume'][1]).resolve():
        problems.append(f'续跑仓库 {resume_repo} 不是主工作树（--git-dir {git_dir[1]} 不是 --git-common-dir {common["resume"][1]}）：链接的 worktree 不行')
    return problems


def make_checkout(commit=None, *, source=rc.ROOT, parent=rc.RESULTS, data_target=rc.DATA_LINK_TARGET, identity=None, log=print):
    """建运行检出：git -C <source> worktree add --detach <parent>/r4_code_<短 sha> <提交>，补 data 符号链接（不进 git），核 HEAD、已跟踪文件干净、冻结树与代码路径
    （he_core.identity，不核 PYTHONPATH；identity 给定时用它，测试用）。目录或记录已在：拒绝，不覆盖。核对不过：检出留着（失败证据），记录里写明，命令退出 8。
    记录以 O_EXCL 写在检出之外：<parent>/r4_code_<短 sha>.make_checkout.json（修订 6a M5），核对不过也写。返回记录。"""
    code, full, err = git_out(source, 'rev-parse', '--verify', f'{commit or "HEAD"}^{{commit}}')
    if code:
        raise rc.Refused(f'提交 {commit or "HEAD"} 在 {source} 里找不到：{err}')
    dest = Path(parent)/f'{rc.CHECKOUT_PREFIX}{full[:8]}'
    record_path = Path(parent)/f'{rc.CHECKOUT_PREFIX}{full[:8]}.make_checkout.json'          # 修订 6a M5：在检出之外，O_EXCL
    if dest.exists() or dest.is_symlink() or record_path.exists():
        raise rc.Refused(f'{dest} 或 {record_path} 已在：不覆盖')
    if git_out(source, 'cat-file', '-e', f'{full}:data')[0] == 0:
        raise rc.Refused(f'提交 {full} 里有 data：data 应是不进 git 的符号链接')
    code, out, err = git_out(source, 'worktree', 'add', '--detach', str(dest), full)
    if code:
        raise rc.Refused(f'git worktree add 失败：{err or out}')
    os.symlink(data_target, dest/'data')
    if identity is None:
        sys.path.insert(0, str(HERE.parent/'he'))
        import he_core
        _, identity_problems_ = he_core.identity(dest, check_pythonpath=False)
    else:
        identity_problems_ = list(identity(dest))
    record = code_record(dest)
    problems = checkout_setup_problems(dest, resume_repo=source, data_target=data_target)+list(identity_problems_)
    if record['commit'] != full:
        problems.append(f"检出的 HEAD {record['commit']} 不是 {full}")
    doc = dict(checkout=str(dest), commit=full, source_repo=str(Path(source).resolve()), data_link=str(data_target), clean=record['clean'], problems=problems, utc=rc.utc(),
               local_time=rc.local_time(), next=f'cd {dest} && PYTHONPATH=<frozen>:{dest} $PY -B experiments/r4/run_r4.py batch-init --role bench（之后所有步骤都在这个检出里跑）')
    rc.write_json(record_path, dict(doc, schema=rc.SCHEMA.format('make_checkout'), record=str(record_path)))       # 核对不过（退出 8）时也写
    log(json.dumps(doc, ensure_ascii=False, indent=1))
    return dict(doc, record=str(record_path))


# ---- 机器门（plan.md 第 1 节；R2、R3 的写法） ---------------------------------------------------------------------------------------------------
def machine_gate(own_unit=None, *, run=subprocess.run, meminfo='/proc/meminfo', loadavg='/proc/loadavg', disk=None):
    """没有别的活动 tag-* / seq-vp-* 单元，1 分钟负载 < 1.5，没有 GPU 计算进程，MemAvailable ≥ 9 GiB，/home ≥ 100 GB，C: ≥ 40 GiB。返回 (problems, record)。"""
    import shutil
    disk = shutil.disk_usage if disk is None else disk
    problems, record = [], dict(utc=rc.utc())
    try:
        listing = run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', 'tag-*', 'seq-vp-*'], capture_output=True, text=True, timeout=60)
        units = sorted(line.split()[0] for line in listing.stdout.splitlines() if line.strip())
        others = [u for u in units if own_unit is None or u.removesuffix('.service') != own_unit]
    except (OSError, subprocess.SubprocessError) as exc:
        others = [f'systemctl: {exc}']
    try:
        gpu = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
        gpu_pids = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        gpu_pids = []
    load = float(Path(loadavg).read_text().split()[0])
    available = next((int(line.split()[1])/2**20 for line in Path(meminfo).read_text().splitlines() if line.startswith('MemAvailable:')), 0.)
    home, c_drive = disk('/home').free/1e9, disk('/mnt/c').free/2**30
    record.update(other_units=others, gpu_compute_pids=gpu_pids, load1=load, mem_available_gib=available, home_free_gb=home, c_free_gib=c_drive)
    problems += [f'别的单元在跑：{others}'] if others else []
    problems += [f'GPU 上有计算进程：{gpu_pids}'] if gpu_pids else []
    problems += [f'负载 {load} ≥ {rc.LOAD_MAX}'] if load >= rc.LOAD_MAX else []
    problems += [f'MemAvailable {available:.1f} GiB < {rc.MEM_AVAILABLE_GIB}'] if available < rc.MEM_AVAILABLE_GIB else []
    problems += [f'/home {home:.0f} GB < {rc.HOME_FREE_GB:.0f}'] if home < rc.HOME_FREE_GB else []
    problems += [f'C: {c_drive:.1f} GiB < {rc.C_FLOOR_GIB:.0f}'] if c_drive < rc.C_FLOOR_GIB else []
    return problems, record


def wait_gate(own_unit, *, minutes=rc.GATE_WAIT_MINUTES, gate=machine_gate, sleep=time.sleep, log=print):
    deadline = time.monotonic()+60*minutes
    while True:
        problems, record = gate(own_unit)
        if not problems:
            return True, record
        if time.monotonic() >= deadline:
            return False, dict(record, problems=problems)
        log(f'门关着：{problems}')
        sleep(60)


# ---- 单元 --------------------------------------------------------------------------------------------------------------------------------------
def unit_name(*parts):
    name = rc.UNIT_PREFIX+'-'.join(''.join(c if c.isalnum() or c == '-' else '-' for c in p) for p in parts)
    if not name.startswith('tag-r4-') or 'seq-vp' in name:
        raise ValueError(name)
    return name[:200]


def unit_argv(name, args, *, repo=rc.ROOT, frozen=None):
    sys.path.insert(0, str(HERE.parent/'mc0'))
    import mc0_core
    frozen = mc0_core.FROZEN if frozen is None else frozen
    env = dict(THREAD_ENV, PYTHONPATH=f'{frozen}:{repo}')
    argv = ['systemd-run', '--user', f'--unit={name}', '--wait', '--collect', f'--working-directory={repo}', *UNIT_MEMORY, *[f'--setenv={k}={v}' for k, v in env.items()], PY, '-B', str(SCRIPT), *args]
    if 'pytest' in ' '.join(argv):
        raise ValueError('启动串里不得有测试运行器的名字')
    return argv


def launch(name, args, *, runner=subprocess.run, log=print):
    argv = unit_argv(name, args)
    log('$ '+shlex.join(argv))
    return runner(argv, capture_output=True, text=True).returncode


# ---- 队列与单步：共用的门、间隔、hold（plan.md 第 1 节；R2、R3 的写法） --------------------------------------------------------------------------
CLOCK = 'unit_clock.json'


def held(root):
    return (Path(root)/HOLD).exists()


def pace(root, gap=rc.GAP_SECONDS, *, sleep=time.sleep, now=time.time):
    """单元之间隔 ≥ gap 秒（A5a 的规则）：上一个 tag-r4- 单元结束的时刻记在 <R4>/unit_clock.json，队列与单步共用。"""
    last = (rc.read_json(Path(root)/CLOCK) or {}).get('ended_epoch')
    if last is not None and now()-last < gap:
        sleep(gap-(now()-last))


def mark_unit_end(root, name, code, *, now=time.time):
    path = Path(root)/CLOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(ended_epoch=now(), unit=name, exit_code=code, utc=rc.utc())), encoding='utf-8')


def gated_launch(root, name, args, *, launch_unit=launch, gate=None, gap=rc.GAP_SECONDS, sleep=time.sleep, log=print):
    """hold → 机器门（关满 30 分钟就停）→ 间隔 → 起单元 → 记结束时刻。返回 (退出码, 原因)；hold 或门关满是 (3, 原因)，没有起单元。"""
    gate = wait_gate if gate is None else gate
    if held(root):
        log('hold：停在下一个单元之前')
        return 3, 'hold'
    os.environ.setdefault('R4_QUEUE_UNIT', own_unit() or '')                                 # 修订 3 次要 6：run_step / 队列自己的单元（若在单元里）不算"别的单元"
    ok, record = gate(os.environ.get('R4_QUEUE_UNIT') or None)
    if not ok:
        log(f'机器门关满：{record}')
        return 3, 'gate'
    pace(root, gap, sleep=sleep)
    code = launch_unit(name, args)
    mark_unit_end(root, name, code)
    return code, None


def launch_problem(root, role, repo=rc.ROOT):
    """None，或这一处不许起单元的原因（修订 6）：队列与单步必须从这一角色的 batch.json（或 dev_batch.json）记的运行检出起（systemd-run 的工作目录、PYTHONPATH、脚本都是这个检出的）。
    修订 6a M2：batch 文件不在也拒（退出 2，不起单元，不写 stop）。"""
    name = 'batch.json' if role == 'bench' else 'dev_batch.json'
    doc = rc.read_json(Path(root)/name)
    if doc is None:
        return f'没有 {Path(root)/name}：先 make-checkout，再在检出里 batch-init --role {role}'
    if not doc.get('checkout') or Path(repo).resolve() != Path(doc['checkout']).resolve():
        return f"代码根 {repo} 不是 {role} 的 batch.json 记的运行检出 {doc.get('checkout')}：从运行检出起队列与单步"
    return None


def unresolved(root, repo, require_committed, log=print):
    """有没处理的读真值之前的失败（没有合格的、已提交的续跑记录），或整台已按 (甲) 不可判：任何单元都不起。返回 (停, {(scope, 臂): 原因})。"""
    import r4_eval as re_
    record = {}
    problems, candidates, whole = re_.resume_state(root, repo, before_lock=False, require_committed=require_committed, record=record)
    if record.get('stops'):
        log(f'续跑仓库（修订 6a M4）：{json.dumps(record, ensure_ascii=False)}')
    if problems:
        log(f'有没处理的读真值之前的失败：{problems}')
    if whole:
        log(f'整台已按 (甲) 不可判：{whole}')
    return bool(problems or whole), candidates


def queue_arm(role, arm, *, root=rc.R4_ROOT, pairs, launch_unit=launch, gate=None, gap=rc.GAP_SECONDS, sleep=time.sleep, log=print, repo=None, require_committed=True):
    """一臂：每对一个单元（跑过的对不再跑），之后定稿单元；开发集再加 SDlogJ 单元。退出码与第一处失败。读真值之前的失败：写 <R4>/stops/stop_<k>.json，返回 9（交总负责人与 BZ）。
    repo：续跑仓库，None 即 batch.json 记的（修订 6）。"""
    import r4_chain as ch
    import r4_eval as re_
    problem = launch_problem(root, role)
    if problem:
        log(f'拒绝：{problem}')
        return 2
    stop, candidates = unresolved(root, repo, require_committed, log)
    if stop:
        return 9
    scope = 'feasibility' if role == 'dev' else ('combo' if rc.is_combo(arm) else 'candidate')
    if (scope, arm) in candidates:
        log(f'{arm} 已按 (甲) 处理（续跑不是重跑，不碰失败的那一臂）：{candidates[(scope, arm)]}')
        return 0
    runs = Path(root)/('runs' if role == 'bench' else 'dev_runs')/arm
    for pair in pairs:
        if ch.line_path(runs/pair).is_file():
            continue
        if (runs/pair).exists():
            re_.write_stop(root, dict(role=role, arm=arm, pair=pair, step='pair', scope=stop_scope(role, arm), reason=f'{runs/pair} 在却没有 line.json（上一次中途停了）：不重跑'))
            return 9
        code, why = gated_launch(root, unit_name(role, arm, pair.split('__')[0][-8:-4], pair.split('__')[-1][-8:-4]), ['pair', '--role', role, '--arm', arm, '--pair', pair],
                                 launch_unit=launch_unit, gate=gate, gap=gap, sleep=sleep, log=log)
        if why:
            return code
        line = ch.line_record(runs/pair) or {}
        if code != 0 or line.get('line_status') not in rc.OUTPUT_STATUSES:
            re_.write_stop(root, dict(role=role, arm=arm, pair=pair, step='pair', exit_code=code, line_status=line.get('line_status'),
                                      scope='bench' if line.get('line_status') == rc.REJECTED and role == 'bench' else stop_scope(role, arm),
                                      reason='读真值之前的失败（C25.7-3）：hx_provenance_rejected 按共用步骤，hx_implementation_error 停下整个队列；交总负责人与 BZ'))
            return 9 if code in (0, 9) else code
    steps = [('finalize', ['finalize', '--role', role, '--arm', arm], stop_scope(role, arm), '定稿单元失败（读真值之前）')]
    if role == 'dev':
        steps.append(('sdlogj', ['sdlogj', '--arm', arm], 'feasibility', 'SDlogJ 单元失败（读真值之前）'))
    for step, args, scope_, reason in steps:
        code, why = gated_launch(root, unit_name(role, arm, step), args, launch_unit=launch_unit, gate=gate, gap=gap, sleep=sleep, log=log)
        if why:
            return code
        if code != 0:
            re_.write_stop(root, dict(role=role, arm=arm, step=step, exit_code=code, scope=scope_, reason=reason))
            return 9
    return 0


# 单步命令的单元（C25.7-3 的归属：数据准备、分割、Atropos 与 0438-0439 的对照归整台；冒烟自己写归属；评测单元建锁之后的结局由评测记录说）
STEPS = {'prepare-bench': 'bench', 'atropos-bench': 'bench', 'seg-bench': 'bench', 'smoke': None, 'devcheck': 'bench', 'eval-bench': 'bench', 'eval-dev': 'feasibility'}
STEP_DIR = 'steps'


def option(extra, name):
    extra = list(extra)
    return extra[extra.index(name)+1] if name in extra and extra.index(name)+1 < len(extra) else None


def step_lock(root, step, extra):
    """这一步的读一次锁（评测单元）；别的步骤没有。"""
    import r4_lock as lk
    if step == 'eval-bench':
        return lk.lock_path(root, lk.BENCH)
    if step == 'eval-dev':
        return lk.lock_path(root, lk.DEV1 if option(extra, '--read') == '1' else lk.DEV2)
    return None


def lock_taken(root, step, extra):
    """这一步的读有没有建锁（修订 4 次要 5）：台的第二读看锁里有没有第二行，第一读与开发集的读数看锁在不在。
    修订 5 第 4 条：只数非空行，不解析（按字节分行，撕裂的行可能断在多字节字符中间）；撕裂的第二行说明第二读已经开始写锁，算建锁之后。"""
    path = step_lock(root, step, extra)
    if path is None or not path.exists():
        return False
    if step == 'eval-bench' and option(extra, '--read') == 'second':
        return sum(1 for line in path.read_bytes().split(b'\n') if line.strip()) >= 2
    return True


def own_unit(cgroup='/proc/self/cgroup'):
    """这个进程所在的 systemd 单元名（tag-*.service），没有就 None（修订 3 次要 6：队列与单步不把自己当成"别的单元"）。"""
    try:
        text = Path(cgroup).read_text()
    except OSError:
        return None
    for part in text.replace('\n', '/').split('/'):
        if part.startswith('tag-') and part.endswith('.service'):
            return part.removesuffix('.service')
    return None


STEP_ROLE = {'devcheck': 'dev', 'eval-dev': 'dev'}                                          # 单步读哪一份 batch（r4_steps._identity）；其余是台的


def run_step(step, extra=(), *, root=rc.R4_ROOT, launch_unit=launch, gate=None, gap=rc.GAP_SECONDS, sleep=time.sleep, log=print, repo=None, require_committed=True):
    """一个单步命令在它自己的单元 tag-r4-<步> 里：没处理的 stop → 不起；同一步（含参数）起过一次就不再起（失败即停、不重跑：标记 <R4>/steps/<名>.json 以 O_EXCL 写）；
    hold、机器门、间隔与一臂的队列共用（gated_launch）；退出码非 0 → 写 stop（冒烟自己写了的不重写），返回 9。hold 或门关满：没起单元，标记改名保留，返回 3。"""
    import r4_eval as re_
    if step not in STEPS:
        raise ValueError(f'没有单步 {step!r}')
    problem = launch_problem(root, STEP_ROLE.get(step, 'bench'))                                 # 修订 6
    if problem:
        log(f'拒绝：{problem}')
        return 2
    stop, _ = unresolved(root, repo, require_committed, log)
    if stop:
        return 9
    label = '-'.join([step]+[str(x).lstrip('-') for x in extra])
    name = unit_name(*label.split('-'))
    marker = Path(root)/STEP_DIR/f'{label}.json'
    if marker.exists():
        log(f'{label} 起过一次（{marker}）：不重跑')
        return 2
    stops_before = len(re_.stop_files(root))
    try:
        rc.write_json(marker, dict(step=step, args=list(extra), unit=name, utc=rc.utc(), local_time=rc.local_time()))
    except FileExistsError:
        return 2
    code, why = gated_launch(root, name, [step, *extra], launch_unit=launch_unit, gate=gate, gap=gap, sleep=sleep, log=log)
    if why:
        k = len(list(marker.parent.glob(f'{label}.not_started_*.json')))
        marker.rename(marker.parent/f'{label}.not_started_{why}_{k}.json')                    # 没有起单元：标记改名保留，可以再起
        return code
    if code != 0:
        if len(re_.stop_files(root)) == stops_before:
            combo = option(extra, '--combo') if step == 'eval-bench' else None
            arm = combo or option(extra, '--arm')
            after = lock_taken(root, step, extra)
            reason = (f'单步 {step} 失败，在建锁之后（{step_lock(root, step, extra)}）：读真值之后的结局，见评测记录；不重跑' if after else
                      f'单步 {step} 失败，在读真值之前（C25.7-3）')
            re_.write_stop(root, dict(step=step, args=list(extra), unit=name, exit_code=code, scope='combo' if combo else (STEPS[step] or 'bench'), arm=arm, reason=reason,
                                      after_lock=after))
        return 9
    return 0


def queue_arm_problem(root, role, arm, *, repo=None):
    """None，或这一臂不许起队列的原因（复核 I4：打错臂名不得用掉读数或跑错的场）：台上只有 ARM_ORDER 的七臂与第一读判定里计划的那个组合；开发集只有 next-dev 给的那一项（可行性一步）。"""
    import r4_decide as rd
    import r4_eval as re_
    if role == 'bench':
        if arm in rc.ARM_ORDER:
            return None
        first = rc.read_json(Path(root)/'analysis'/'bench_decision_first.json') or {}
        if rc.is_combo(arm) and first.get('combo') == arm and first.get('combo_needs_second_read'):
            return None
        return f'{arm} 不是七臂之一，也不是第一读判定里计划的组合（{first.get("combo")}）'
    order, feasibility, readings = re_.dev_state(root, repo)
    step = rd.next_dev_step(order, feasibility, readings)
    if step.get('action') == 'feasibility' and step.get('item') == arm:
        return None
    return f'开发集的下一步是 {step}，不是 {arm} 的可行性一步'


def stop_scope(role, arm):
    """失败的归属（C25.7-3）：开发集是可行性队列；台上 CTL 归整台，组合臂归组合，其余归该候选。"""
    if role == 'dev':
        return 'feasibility'
    return 'bench' if arm == rc.CONTROL else ('combo' if rc.is_combo(arm) else 'candidate')


# ---- 单元的主体 ------------------------------------------------------------------------------------------------------------------------------------
def batch(root, role):
    doc = rc.read_json(Path(root)/('batch.json' if role == 'bench' else 'dev_batch.json'))
    if doc is None:
        raise rc.Refused(f'没有 {role} 的 batch.json：先跑 batch-init')
    return doc


def batch_init(root, role, *, repo=rc.ROOT, resume_repo=rc.RESUME_REPO):
    """修订 6：在 make-checkout 建的运行检出里跑（不是续跑仓库、HEAD 游离、data 链接）；batch.json 记运行检出与续跑仓库。另一份 batch 在时，续跑仓库要相同。"""
    import r4_bench as rb
    if Path(root).resolve() == rc.R4_ROOT.resolve() and Path(resume_repo).resolve() != rc.RESUME_REPO.resolve():
        raise rc.Refused(f'真实运行的续跑仓库只能是 {rc.RESUME_REPO}，不是 {resume_repo}（修订 6a I1）')
    setup = checkout_setup_problems(repo, resume_repo=resume_repo)
    if setup:
        raise rc.Refused(f'运行检出：{setup}')
    other = rc.read_json(Path(root)/('dev_batch.json' if role == 'bench' else 'batch.json'))
    if other is not None and Path(other.get('resume_repo') or '/nonexistent').resolve() != Path(resume_repo).resolve():
        raise rc.Refused(f"另一份 batch 记的续跑仓库 {other.get('resume_repo')} 不是 {resume_repo}")
    problems, code = identity_problems(repo)
    plan = rc.plan_problem(repo=repo)
    if problems or plan or not code['clean']:
        raise rc.Refused(f'代码身份或 plan.md：{problems} {plan}')
    doc = dict(schema=rc.SCHEMA.format('batch'), role=role, code_commit=code['commit'], clean=code['clean'], plan_sha256=rc.PLAN_SHA256, arms=list(rc.ARM_ORDER),
               specs={a: dict(spec=rc.arm_spec(a), sha256=rc.spec_sha256(rc.arm_spec(a))) for a in rc.ARM_ORDER}, utc=rc.utc(), local_time=rc.local_time(),
               checkout=str(Path(repo).resolve()), data_link=os.readlink(Path(repo)/'data'), resume_repo=str(Path(resume_repo).resolve()),
               resume_dir=str(Path(resume_repo).resolve()/__import__('r4_eval').RESUME_DIR_REL),
               commits_by_unit={}, note='the units of this batch run from the checkout at `checkout` (HEAD = code_commit; C25.9-6); the resume records are checked in `resume_repo` '
                                        '(committed there, the working copy equal to the commit; its HEAD is not pinned); a (乙) rerun adds its commit by unit (C25.9-6)')
    if role == 'bench':
        bench = rb.load_bench()
        doc.update(folds_sha256=rc.FOLDS_SHA256, registry_sha256=rc.sha256_file(repo/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(repo/rc.BENCH_CONTACT_REL),
                   pairs=[p[0] for p in bench.pairs], pairs_table_sha256=rb.pairs_table_sha256(bench), subjects=bench.subjects)
        rb.load_bench_registry(bench, dict(registry_sha256=doc['registry_sha256'], contact_sha256=doc['contact_sha256']))
    return rc.write_json(Path(root)/('batch.json' if role == 'bench' else 'dev_batch.json'), doc)


def pins_of(root):
    b = batch(root, 'bench')
    return dict(registry_sha256=b['registry_sha256'], contact_sha256=b['contact_sha256'])


def bench_pairs(root):
    return list(batch(root, 'bench')['pairs'])


def stage_dir(root, role):
    return Path(root)/'bench_stage' if role == 'bench' else rc.ROOT/rc.DEV_STAGE_REL


def rules_for(ctx, unit):
    """一对一臂的单元的白名单：本对两人的子目录与本对的 pair 目录（r4_spy.pair_rules，复核次要 7）。"""
    import r4_spy as spy
    return spy.pair_rules(ctx, unit)


def supervise(ctx, *, rules=None, run=None):
    """supervisor 在 spy 之下（C25.10-5）：原 pair.npz 的成员守卫与正条件（准入读到 fixed、moving），白名单。spy 不过：退出码 8（队列写 stop）。"""
    import r4_chain as ch
    import r4_spy as spy
    rules = rules_for(ctx, 'supervisor') if rules is None else rules
    with spy.UnitSpy(rules, Path(ctx['stage_dir'])/ctx['pair']/'pair.npz') as s:
        code = (ch.run_pair if run is None else run)(ctx)
    record = s.record()
    if Path(ctx['out']).is_dir():
        rc.write_json(Path(ctx['out'])/'opened_files_supervisor.json', record)
    return code if record['passed'] else 8


def cmd_pair(o):
    import r4_chain as ch
    root = rc.R4_ROOT
    b = batch(root, o.role)
    problems, _ = batch_identity(b)
    if problems:
        print(problems, file=sys.stderr)
        return 8
    ctx = ch.make_context(o.role, o.arm, o.pair, root=root, stage_dir=stage_dir(root, o.role), device=o.device, bench_pins=pins_of(root) if o.role == 'bench' else None)
    return supervise(ctx)


def cmd_child(o):
    import r4_chain as ch
    ctx = ch.read_context(o.context)
    import run_hx
    _, problems = run_hx.check_identity(ctx, ctx['out'], f'identity_{o.command}.json')
    if problems:
        return 8
    if ctx.get('check_identity') is True and Path(ch.core.ROOT).resolve() != Path(ctx.get('code_root') or '/nonexistent/no_code_root').resolve():     # 修订 6a M1（测试上下文照旧放过）
        rc.write_json(Path(ctx['out'])/f'code_root_{o.command}.json', dict(code_root=ctx.get('code_root'), core_root=str(Path(ch.core.ROOT).resolve()), utc=rc.utc(),
                                                                           problem='the child runs from another code root than the one its context was made in'))
        return 8
    return ch.run_child_spied(ctx, o.command)                                      # 复核 B2：Breach 被捕获、记录照写、退出 8


def finalize_arm(root, role, arm, pairs, *, rules=None, finalize=None, code=None):
    """定稿单元：每对在自己的 spy 之下（原 pair.npz 的正条件：有输出的对，准入要读到 fixed、moving；没有输出的对什么都不读）。记录 opened_files_finalize.json；spy 不过：8。"""
    import r4_chain as ch
    import r4_spy as spy
    import r4_chain as ch
    finalize = ch.finalize_pair if finalize is None else finalize
    records, passed = {}, True
    for pair in pairs:
        ctx = rc.read_json(Path(root)/('runs' if role == 'bench' else 'dev_runs')/arm/pair/'context.json')
        if ctx is None:
            print(f'{pair}: no context.json', file=sys.stderr)
            return 9
        line = ch.line_record(ctx['out']) or {}
        expect = line.get('line_status') in rc.OUTPUT_STATUSES
        with spy.UnitSpy(rules_for(ctx, 'finalize') if rules is None else rules, Path(ctx['stage_dir'])/pair/'pair.npz', expect_reads=expect) as s:
            finalize(ctx)
        records[pair] = s.record()
        passed = passed and records[pair]['passed']
    target = Path(root)/('final' if role == 'bench' else 'dev_final')/arm/'opened_files_finalize.json'
    stopped = [p for p in pairs if (ch.read_final(root, role, arm, p) or {}).get('line_status') == rc.IMPLEMENTATION]
    rc.write_json(target, dict(unit='finalize', arm=arm, role=role, pairs=records, passed=passed, code=code, implementation_errors=stopped))
    if not passed:
        return 8
    return ch.STOP_EXIT if stopped else 0                                                      # 复核次要 5：恒等的第三次认证也不过：hx_implementation_error，停队列


def cmd_finalize(o):
    root = rc.R4_ROOT
    b = batch(root, o.role)
    problems, _ = batch_identity(b)
    if problems:
        return 8
    pairs = bench_pairs(root) if o.role == 'bench' else __import__('r4_eval').dev_pairs()
    return finalize_arm(root, o.role, o.arm, pairs, code=code_record())


def sdlogj_arm(root, arm, pairs, *, rules=None, code=None):
    """开发集可行性：官方 SDlogJ（不读标签）。spy 之下：不打开任何 pair.npz；正条件：每个有输出的对的 final.npz 都被打开过；白名单是开发集的类别。spy 不过：8。"""
    import r4_chain as ch
    import r4_decide as rd
    import r4_spy as spy
    rules = spy.dev_rules(root, unit='sdlogj', extra=(Path(root)/'dev_final'/arm, Path(root)/'dev_runs'/arm)) if rules is None else rules
    rows, outputs = {}, []
    with spy.UnitSpy(rules, expect_reads=False) as s:
        for pair in pairs:
            final = ch.read_final(root, 'dev', arm, pair)
            if final is None:
                rows[pair] = dict(line_status=None)
                continue
            row = dict(line_status=final.get('line_status'))
            if row['line_status'] in rc.OUTPUT_STATUSES:
                outputs.append(str(Path(final['output']['directory'])/'final.npz'))
                u, problem = ch.output_array(final)
                grid = (rc.read_json(Path(final['output']['directory'])/'line.json') or {}).get('grid') or {}
                spacing = tuple(grid.get('spacing_xyz') or ())
                if problem is None and len(spacing) != 3:
                    problem = 'the output line.json names no grid spacing'
                row['sdlogj_official'] = None if problem else ch.sdlogj_of(u, spacing)
                row['problem'] = problem
            rows[pair] = row
    record = s.record()
    opened = [r['path'] for r in record['opened']]
    record['positive_final_npz'] = all(path in opened for path in outputs) and not any(Path(p).name == 'pair.npz' for p in opened)
    record['passed'] = record['passed'] and record['positive_final_npz']
    verdict = rd.feasibility_verdict(rows)
    rc.write_json(Path(root)/'dev_final'/arm/'opened_files_sdlogj.json', record)
    rc.write_json(Path(root)/'dev_final'/arm/'feasibility.json', dict(schema=rc.SCHEMA.format('feasibility'), arm=arm, rows=rows, verdict=verdict, utc=rc.utc(), spy_passed=record['passed'], code=code,
                                                                     rule='C25.8-3: 19/19 with an output and the method-level mean official SDlogJ <= .10; no truth read'))
    if not record['passed']:
        return 8
    return 9 if verdict['verdict'] == rd.FEAS_STOP else 0


def cmd_sdlogj(o):
    b = batch(rc.R4_ROOT, 'dev')
    problems, code = batch_identity(b)                                                         # 复核次要 11；修订 6
    if problems:
        return 8
    return sdlogj_arm(rc.R4_ROOT, o.arm, __import__('r4_eval').dev_pairs(), code=code)


def cmd_dry_run(o):
    import r4_bench as rb
    problems, code = identity_problems()
    print(f'代码身份：{"ok" if not problems else problems}；HEAD {code["commit"]}，clean {code["clean"]}')
    print(f'plan.md：{rc.plan_problem() or "sha256 与钉住的相同"}')
    if o.role in (None, 'bench'):
        try:
            bench = rb.load_bench()
            print(f'台：16 人 {sum(len(v) for v in bench.subjects.values())}，16 对 {len(bench.pairs)}，配对表 sha256 {rb.pairs_table_sha256(bench)}')
            pins = dict(registry_sha256=rc.sha256_file(rc.ROOT/rc.BENCH_REGISTRY_REL), contact_sha256=rc.sha256_file(rc.ROOT/rc.BENCH_CONTACT_REL))
            rb.load_bench_registry(bench, pins)
            print(f'台的登记：ok（{pins}）')
            for name in bench.pair_names:
                rb.bench_admission(bench, pair=name)
            print('台守卫：16 对都过')
        except rc.Refused as exc:
            print(f'台：拒绝：{exc}')
            return 8
    if o.role in (None, 'dev'):
        import r4_eval as re_
        pairs = re_.dev_pairs()
        print(f'开发集：{len(pairs)} 对')
    return 0 if not problems else 8


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dry-run')
    d.add_argument('--role', choices=('bench', 'dev'), default=None)
    b = sub.add_parser('batch-init')
    b.add_argument('--role', choices=('bench', 'dev'), required=True)                         # 修订 6a I1：没有 --resume-repo，续跑仓库写死 rc.RESUME_REPO
    m = sub.add_parser('make-checkout')
    m.add_argument('--commit', default=None, help='default: HEAD of the repository this script is in')
    for name in ('prepare-bench', 'atropos-bench', 'seg-bench', 'smoke', 'devcheck', 'decide-bench', 'next-dev', 'precheck-hs'):
        q = sub.add_parser(name)
        q.add_argument('--device', default='cuda')
        q.add_argument('--dry-run', action='store_true')
    q = sub.add_parser('queue')
    q.add_argument('--role', choices=('bench', 'dev'), required=True)
    q.add_argument('--arm', required=True)
    q.add_argument('--gap-seconds', type=float, default=rc.GAP_SECONDS)
    q.add_argument('--wait-minutes', type=float, default=rc.GATE_WAIT_MINUTES)
    q.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('pair')
    p.add_argument('--role', choices=('bench', 'dev'), required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--pair', required=True)
    p.add_argument('--device', default='cuda')
    for name in ('t88', 'repair'):
        c = sub.add_parser(name)
        c.add_argument('--context', required=True)
    f = sub.add_parser('finalize')
    f.add_argument('--role', choices=('bench', 'dev'), required=True)
    f.add_argument('--arm', required=True)
    s = sub.add_parser('sdlogj')
    s.add_argument('--arm', required=True)
    e = sub.add_parser('eval-bench')
    e.add_argument('--read', choices=('first', 'second'), required=True)
    e.add_argument('--combo', default=None)
    e.add_argument('--dry-run', action='store_true')
    v = sub.add_parser('eval-dev')
    v.add_argument('--read', type=int, choices=(1, 2), required=True)
    v.add_argument('--arm', required=True)
    v.add_argument('--dry-run', action='store_true')
    w = sub.add_parser('decide-dev')
    w.add_argument('--read', type=int, choices=(1, 2), required=True)
    u = sub.add_parser('unit', help='a single step in its own unit tag-r4-<step> behind the shared gate, hold and spacing; once only')
    u.add_argument('step', choices=sorted(STEPS))
    u.add_argument('rest', nargs=argparse.REMAINDER, help="the step's own options (e.g. --read first, --arm B)")
    return ap


def main(argv=None):
    o = build_parser().parse_args(argv)
    import r4_steps as steps
    handlers = {'dry-run': cmd_dry_run, 'pair': cmd_pair, 't88': cmd_child, 'repair': cmd_child, 'finalize': cmd_finalize, 'sdlogj': cmd_sdlogj}
    try:
        if o.command in handlers:
            return handlers[o.command](o)
        if o.command == 'unit':
            return run_step(o.step, [a for a in o.rest if a != '--'])
        if o.command == 'batch-init':
            batch_init(rc.R4_ROOT, o.role)
            return 0
        if o.command == 'make-checkout':
            try:
                return 0 if not make_checkout(o.commit)['problems'] else 8
            except rc.Refused as exc:                                                       # 用法或拒绝（目录已在、提交找不到）：没有建检出
                print(f'拒绝：{exc}', file=sys.stderr)
                return 2
        if o.command == 'queue':
            if o.arm not in rc.ARM_ORDER and not rc.is_combo(o.arm):
                return 2
            where = launch_problem(rc.R4_ROOT, o.role)                                           # 修订 6a M2：干跑也核、也打印
            if o.dry_run:
                print('运行检出：'+(where or f'ok（代码根 {rc.ROOT} 是 batch 记的检出）'))
            if where:
                print(f'拒绝：{where}', file=sys.stderr)
                return 2
            problem = queue_arm_problem(rc.R4_ROOT, o.role, o.arm)
            if problem:
                print(f'拒绝：{problem}', file=sys.stderr)
                return 2
            pairs = bench_pairs(rc.R4_ROOT) if o.role == 'bench' else __import__('r4_eval').dev_pairs()
            if o.dry_run:
                problems, record = machine_gate(os.environ.get('R4_QUEUE_UNIT'))
                print(f'门：{problems or "开"}；{len(pairs)} 对；单元名例：{unit_name(o.role, o.arm, "0000", "0001")}')
                return 0
            return queue_arm(o.role, o.arm, pairs=pairs, gap=o.gap_seconds, gate=lambda own: wait_gate(own, minutes=o.wait_minutes))
        return steps.dispatch(o)
    except rc.Refused as exc:
        print(f'拒绝：{exc}', file=sys.stderr)
        return 9


if __name__ == '__main__':
    sys.exit(main())
