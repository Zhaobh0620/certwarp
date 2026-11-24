"""R4 的读一次锁（合同 C25.6-11、C25.8-4、C25.9-4）。

  bench_truth_read.lock   台的真值：第一行"第一读"（全部臂），第二行只能是组合臂计划内的第二读；再无第三行。
  dev_read_1.lock、dev_read_2.lock   开发集第 1、2 次读数，各一行。
锁在全部锁前核对之后以 O_EXCL 建，建好之后才打开真值；锁之前失败不留锁。建锁之后的任何结局都是终局，不再读（修订 10e：没有"BZ 批准再读"这一条）。
每行一个 JSON：{read: first | second, utc, local_time, arms, code_commit, plan_sha256, reason, ...}。锁文件不删、不改。
"""
import json
import os
from pathlib import Path

import r4_common as rc

BENCH, DEV1, DEV2 = 'bench_truth_read.lock', 'dev_read_1.lock', 'dev_read_2.lock'
FIRST, SECOND = 'first', 'second'


class LockRefused(Exception):
    """这一读不许做：锁已在（第一读、开发集读数），或第二行不合规则。"""
    status = 'lock_refused'


def lock_path(root, name):
    return Path(root)/'locks'/name


def lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def _line(read, *, arms, code_commit, plan_sha256, reason, extra=None):
    return dict(read=read, utc=rc.utc(), local_time=rc.local_time(), arms=list(arms), code_commit=code_commit, plan_sha256=plan_sha256, reason=reason, pid=os.getpid(), **(extra or {}))


def check_absent(path):
    """锁前核对的第一条：锁不在（第一读、开发集读数）。"""
    if Path(path).exists():
        first = (lines(path) or [{}])[0]
        raise LockRefused(f'{path} 已在（{first.get("read")}，{first.get("utc")}）：真值已读过一次，建锁之后的任何结局都是终局，不再读')


def take_first(path, *, arms, code_commit, plan_sha256, reason, extra=None):
    """第一读（台）或开发集的一次读数：O_EXCL 建锁，写第一行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _line(FIRST, arms=arms, code_commit=code_commit, plan_sha256=plan_sha256, reason=reason, extra=extra)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise LockRefused(f'{path} 在建锁的那一刻已在：不读') from None
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(line, ensure_ascii=False)+'\n')
    return line


def check_second(path, combo):
    """第二行的规则（C25.6-11、C25.9-4）：锁里恰有第一读一行；组合臂没在第一读里；只许一个组合臂。"""
    rows = lines(path)
    if len(rows) != 1 or rows[0].get('read') != FIRST:
        raise LockRefused(f'{path}：第二行只能追加在恰好一行"第一读"之后（现有 {len(rows)} 行）')
    if not rc.is_combo(combo):
        raise LockRefused(f'第二读只许组合臂，不是 {combo!r}')
    rc.combo_parts(combo)
    if combo in rows[0].get('arms', []):
        raise LockRefused(f'{combo} 已在第一读里')
    return rows[0]


def take_second(path, combo, *, code_commit, plan_sha256, truth_sha256, reason='第二读（计划内，组合臂）'):
    """组合臂计划内的第二读：同一把锁追加第二行；真值文件的 sha256 由评测单元在读完之后对照第一读（C25.6-11）。"""
    first = check_second(path, combo)
    line = _line(SECOND, arms=[combo], code_commit=code_commit, plan_sha256=plan_sha256, reason=reason, extra=dict(first_read_truth_sha256=truth_sha256))
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(line, ensure_ascii=False)+'\n')
    return first, line
