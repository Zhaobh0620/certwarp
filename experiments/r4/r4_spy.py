"""R4 的 spy（合同 C25.10-5、-6，修订 10a、10b、10c、10e）：逐单元记录打开的每个文件（含 npz 里读了哪些成员），按白名单判。

- 复用 experiments/he/spies.py（未改），带它的正条件：每对的单元必须看到预期次数的 fixed、moving 读取（读到 0 次的 spy 不算通过）。
- 白名单式断言：路径按 resolve() 之后判；只在数据与结果根之内判（/mnt/d/data、仓库 data/ 链接的目标、仓库本身、/home/rmcfz/seq_vp_tag_results），根之外只记录不判。
  仓库里只放行代码目录（bayesreg/、experiments/、jd_curl/、scripts/）与明列的配置文件；台上与可行性一步各有自己的类别。
- 真值文件（aligned_seg35、aligned_seg4）与评测记录（spies.EVALUATION_RECORDS 的名字，加 R4 自己的评测与判定文件）不在任何配准一侧单元的白名单里；评测单元另判（每读一次每份只打开一次）。
- 打开记录写成 opened_files.json，随单元的记录存档。子进程（Atropos、配置 C 的冻结 worker）看不到：只核 argv 与交给它的路径（firewall_c.firewall_check）。
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_he = str(HERE.parent/'he')
if _he not in sys.path:
    sys.path.insert(0, _he)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import spies  # noqa: E402

import r4_common as rc  # noqa: E402

TRUTH_NAMES = ('aligned_seg35', 'aligned_seg4')
R4_EVAL_RECORDS = ('bench_eval', 'dev_eval', 'bench_decision', 'dev_decision', 'decision_hs', 'composition', 'calibration', 'eval_')
CODE_DIRS = ('bayesreg', 'experiments', 'jd_curl', 'scripts')


def resolved(path):
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def judged_roots(repo=rc.ROOT, resume_repo=None):
    """修订 6：运行检出在 RESULTS 之下；续跑仓库（main）另算一个判定的根，从检出跑的单元读 main 里的任何东西都要过白名单（只放行续跑目录）。"""
    return [resolved(p) for p in (rc.DATA_ROOT, rc.DATA_LINK_TARGET, repo, rc.RESULTS, rc.RESUME_REPO if resume_repo is None else resume_repo)]


def _under(p, root):
    return root is not None and p is not None and (p == root or root in p.parents)


def repo_config_files(repo=rc.ROOT):
    """仓库里明列的配置文件（代码目录之外）：白名单与它的来源清单、封存名单、登记、分组、折、训练集名单、plan.md 与续跑记录。"""
    import json
    from bayesreg.io.sealed import SEALED_LISTS
    repo = Path(repo)
    files = {repo/rc.ALLOWLIST_REL, repo/rc.BENCH_REGISTRY_REL, repo/rc.BENCH_CONTACT_REL, repo/rc.HS_REGISTRY_REL, repo/'configs/bayesreg/structure_groups_v1.json', repo/rc.FOLDS_REL,
             repo/rc.TRAIN_LIST_REL, repo/rc.PLAN_REL, repo/'configs/bayesreg/colour_table_a5_v1.json', repo/'configs/bayesreg/seg4_of_seg35_v1.json',
             repo/'splits/bayesreg_v1/folds.json'}
    files |= {repo/s['path'] for s in SEALED_LISTS}
    try:
        for source in json.loads((repo/rc.ALLOWLIST_REL).read_text(encoding='utf-8')).get('sources', []):
            files.add(repo/source['path'])
    except (OSError, ValueError):
        pass
    return {resolved(f) for f in files}


def is_truth(path):
    name = Path(path).name
    return any(t in name for t in TRUTH_NAMES)


def is_evaluation_record(path):
    p = Path(path)
    return p.suffix == '.json' and (any(w in p.name for w in spies.EVALUATION_RECORDS) or any(w in p.name for w in R4_EVAL_RECORDS))


def bench_rules(root, *, unit, stage_dir, pred_root, atropos_root, models_root=None, extra=(), record_dirs=(), allow_images=True, repo=rc.ROOT, resume_repo=None):
    """台上多受试者单元（prepare、Atropos、推断、评测）的白名单（C25.10-5）：给定的 stage、预测、Atropos 根、成员文件、aligned_norm 图像，单元自己的输出（extra），代码、冻结树、明列的配置。
    评测记录只在 record_dirs 之下放行（评测单元自己的输出、第一读的记录）。"""
    root = resolved(root)
    allowed_dirs = [resolved(stage_dir), resolved(pred_root), resolved(atropos_root), resolved(core_frozen()), root/'cache' if root else None]+[resolved(e) for e in extra]
    if models_root is not None:
        allowed_dirs.append(resolved(models_root))
    return Rules(unit=unit, allowed_dirs=[d for d in allowed_dirs if d is not None], image_root=resolved(rc.DATA_ROOT), allow_images=allow_images, repo=repo,
                 record_dirs=[resolved(d) for d in record_dirs], resume_repo=resume_repo)


def dev_rules(root, *, unit, repo=rc.ROOT, extra=(), record_dirs=(), resume_repo=None):
    """开发集多对单元（SDlogJ、读数）的白名单：冻结树、cache 与给定的目录（extra）；不放行整个 seg/atropos_v1（里面有 dice_vs_truth_seg4.json，复核次要 7）。"""
    root = resolved(root)
    allowed_dirs = [resolved(core_frozen()), root/'cache']+[resolved(e) for e in extra]
    return Rules(unit=unit, allowed_dirs=[d for d in allowed_dirs if d is not None], image_root=resolved(rc.DATA_ROOT), allow_images=False, repo=repo,
                 record_dirs=[resolved(d) for d in record_dirs], resume_repo=resume_repo)


def pair_rules(ctx, unit, *, repo=rc.ROOT):
    """一对一臂的单元（supervisor、T88、修复、定稿、冒烟、0438-0439 的对照）：收窄到本对的 stage 目录、本对两人的预测与 Atropos 子目录、本单元的输出目录、本对的配置 C 目录、
    本对的定稿目录、cache、冻结树（复核次要 7）。台与开发集同一规则，只是根不同。不放行任何图像（配准一侧只读 stage）。"""
    root = resolved(ctx['root'])
    pair = ctx['pair']
    fixed, moving = pair.split('__')
    pred, atropos = Path(ctx['paths']['pred_root']), Path(ctx['paths']['atropos_root'])
    bench = ctx['role'] == 'bench'
    dirs = [Path(ctx['stage_dir'])/pair, pred/fixed, pred/moving, atropos/fixed, atropos/moving, Path(ctx['out']), root/('configC' if bench else 'dev_configC')/pair,
            root/('final' if bench else 'dev_final')/ctx['arm']/pair, Path(ctx['paths'].get('cache') or root/'cache'), core_frozen()]
    if bench and ctx['paths'].get('models_root'):
        dirs += sorted(Path(ctx['paths']['models_root']).glob('fold*'))                       # the member directories (member.json: the temperatures, 复核次要 9)
    writes = [Path(ctx['out']), root/('configC' if bench else 'dev_configC')/pair, root/('final' if bench else 'dev_final')/ctx['arm']/pair, Path(ctx['paths'].get('cache') or root/'cache')]
    return Rules(unit=unit, allowed_dirs=[resolved(d) for d in dirs], image_root=resolved(rc.DATA_ROOT), allow_images=False, repo=repo, write_dirs=writes)


def core_frozen():
    sys.path.insert(0, str(HERE.parent/'mc0')) if str(HERE.parent/'mc0') not in sys.path else None
    import mc0_core
    return mc0_core.FROZEN


class Rules:
    def __init__(self, *, unit, allowed_dirs, image_root, allow_images, repo=rc.ROOT, record_dirs=(), write_dirs=None, resume_repo=None):
        self.unit, self.allowed_dirs, self.image_root, self.allow_images = unit, allowed_dirs, image_root, allow_images
        self.record_dirs = [d for d in record_dirs if d is not None]
        self.write_dirs = None if write_dirs is None else [resolved(d) for d in write_dirs if d is not None]
        self.repo = resolved(repo)
        self.code_dirs = [self.repo/d for d in CODE_DIRS]
        self.config_files = repo_config_files(repo)
        self.roots = judged_roots(repo, resume_repo)
        self.allow_truth = False

    def judge_write(self, path):
        """写操作（修订 3，N-I1）：只许写在本单元自己的输出目录里（write_dirs；None 时退回读的规则）；根之外只记录。"""
        p = resolved(path)
        if p is None or not any(_under(p, r) for r in self.roots):
            return False, True, 'outside the judged roots (recorded only)'
        if self.write_dirs is None:
            return self.judge(path)
        if is_truth(p):
            return True, False, 'a truth label file (written)'
        if any(_under(p, d) for d in self.write_dirs):
            return True, True, 'written into an output directory of this unit'
        return True, False, 'written outside the output directories of this unit'

    def judge(self, path):
        """(judged, allowed, why)。"""
        p = resolved(path)
        if p is None:
            return False, True, 'not resolvable'
        if not any(_under(p, r) for r in self.roots):
            return False, True, 'outside the judged roots (recorded only)'
        if is_truth(p):
            if self.allow_truth and (_under(p, self.image_root) or _under(p, resolved(rc.DATA_LINK_TARGET))):
                return True, True, 'a truth label file (the evaluation unit; LabelGuard)'
            return True, False, 'a truth label file'
        if is_evaluation_record(p):
            if any(_under(p, d) for d in self.record_dirs):
                return True, True, 'an evaluation record of this evaluation unit (its own outputs, the first read)'
            return True, False, 'an evaluation record'
        if any(_under(p, d) for d in self.code_dirs) or p in self.config_files:
            return True, True, 'code or a listed configuration file'
        if _under(p, self.image_root) or _under(p, resolved(rc.DATA_LINK_TARGET)):
            if p.name == 'aligned_norm.nii.gz' and self.allow_images:
                return True, True, 'an aligned_norm image'
        if any(_under(p, d) for d in self.allowed_dirs):
            return True, True, 'an allowed directory of this unit'
        return True, False, 'not on this unit\'s whitelist'


class UnitSpy:
    """一个单元的 spy：HE 的 Spies（原 pair.npz 的成员守卫与正条件；没有原 pair 时 original 指向不存在的路径），加白名单判定。
    truth_readers（修订 3，N-B1）：[(模块名, 函数名, key_of(args, kwargs))]，评测单元读真值的函数；每次调用记一条"逻辑读"，连同调用期间打开的文件的序号区间。"""

    def __init__(self, rules, original=None, *, expect_reads=None, truth_readers=()):
        self.rules = rules
        self.inner = spies.Spies(original if original is not None else Path('/nonexistent/r4_no_original_pair.npz'))
        self.expect_reads = (original is not None) if expect_reads is None else expect_reads
        self.truth_readers, self.logical, self.patched = tuple(truth_readers), [], []

    def __enter__(self):
        self.stack, self.serial, self.members = [], 0, []
        try:                                                                                     # 修订 5 第 3 条：任何一步失败都先还原（builtins.open、NpzFile.__getitem__、读取器），再抛出
            self._install()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _install(self):
        self.inner.install()
        import importlib
        if self.truth_readers:                                                                   # 修订 4 次要 2：npz 里名字含 _seg 的成员，读的时候在哪一次逻辑读之内（修订 5 第 2 条：记逻辑读的编号）
            import numpy as np
            current = np.lib.npyio.NpzFile.__getitem__
            spy_ = self

            def getitem(npz, key, _current=current):
                if isinstance(key, str) and '_seg' in key:
                    spy_.members.append(dict(member=key, file=str(getattr(getattr(npz, 'zip', None), 'filename', None)), logical=spy_.stack[-1] if spy_.stack else None,
                                             inside=bool(spy_.stack)))
                return _current(npz, key)
            np.lib.npyio.NpzFile.__getitem__ = getitem
            self.patched.append((np.lib.npyio.NpzFile, '__getitem__', current))
        for module_name, attr, key_of in self.truth_readers:
            module = importlib.import_module(module_name) if module_name not in sys.modules else sys.modules[module_name]
            real = getattr(module, attr)
            spy = self

            def counted(*args, _real=real, _name=f'{module_name}.{attr}', _key=key_of, **kwargs):
                start, ident = len(spy.inner.opened), spy.serial
                spy.serial += 1
                spy.stack.append(ident)
                try:
                    return _real(*args, **kwargs)
                finally:
                    spy.stack.pop()
                    spy.logical.append(dict(id=ident, reader=_name, key=list(_key(args, kwargs)), start=start, end=len(spy.inner.opened)))
            setattr(module, attr, counted)
            self.patched.append((module, attr, real))

    def __exit__(self, *exc):
        for module, attr, real in reversed(self.patched):
            setattr(module, attr, real)
        self.patched = []
        self.inner.remove()
        return False

    def record(self, *, expect_reads=None):
        base = self.inner.record(expect_reads=self.expect_reads if expect_reads is None else expect_reads)
        rows, violations = [], []
        reads = {}
        for path in self.inner.read_opened:
            reads[path] = reads.get(path, 0)+1
        for index, path in enumerate(self.inner.opened):
            is_read = reads.get(path, 0) > 0
            if is_read:
                reads[path] -= 1
            judged, allowed, why = self.rules.judge(path) if is_read else self.rules.judge_write(path)
            rows.append(dict(index=index, path=str(path), resolved=str(resolved(path)), mode='read' if is_read else 'write', judged=judged, allowed=allowed, why=why))
            if judged and not allowed:
                violations.append(dict(path=str(path), mode='read' if is_read else 'write', why=why))
        truth = [r['path'] for r in rows if is_truth(r['path'])]
        base_passed, base_rule = self.base_passed(base)
        passed = base_passed and not violations and (self.rules.allow_truth or not truth)
        return dict(unit=self.rules.unit, he_spy=base, base_passed=base_passed, base_rule=base_rule, opened=rows, violations=violations, truth_opened=truth, truth_logical=list(self.logical),
                    truth_members=list(getattr(self, 'members', [])), passed=passed,
                    rule='passed = the base spy passes (base_rule), no truth file or member opened, every judged read on the unit\'s whitelist and every write in its output directories (C25.10-5)')

    def base_passed(self, base):
        """底层（HE 的 spies）是否通过。修订 8（复核阻断）：评测单元（rules.record_dirs 非空）不用 HE 的 passed——它对名字含 metrics 等的 .json 一律算评测记录、不看位置，
        开发集读数要读 HS 的 repair_metrics.json 与 decision_hs.json（在 record_dirs 之内）——改用 R4 自己的规则重算：正条件成立、没有 forbidden_access、读过的评测记录全部落在
        rules.record_dirs 之内。其余单元照旧用 HE 的 passed。"""
        if not self.rules.record_dirs:
            return base['passed'], 'the HE spy\'s passed (positive condition, no forbidden access, no evaluation record opened)'
        outside = [p for p in base['evaluation_records_opened'] if not any(_under(resolved(p), d) for d in self.rules.record_dirs)]
        ok = base['positive_condition'] is True and not base['forbidden_access'] and not outside
        return ok, (f'an evaluation unit (修订 8): the HE spy\'s positive condition, no forbidden access, every evaluation record opened inside rules.record_dirs'
                    + (f'; outside: {outside}' if outside else ''))


def truth_reads(record, per_read=1, *, expected_keys=None):
    """评测单元（修订 3，N-B1；合同 C25.10-5 的"只打开一次"由总负责人记为"只读取一次（一次逻辑读）"）：每份真值（键：受试者与种类，或开发集的对）每读一次恰好一次逻辑读；
    每次打开真值文件都落在某一次逻辑读的调用之内（jd_curl.nifti.load_scalar 一次读要打开同一文件几次：嗅探、读头、ArrayProxy、file_sha256）。返回问题列表。"""
    problems, counts = [], {}
    for row in record.get('truth_logical') or []:
        key = tuple(row['key'])
        counts[key] = counts.get(key, 0)+1
    problems += [f'{list(k)}: {n} logical reads' for k, n in counts.items() if n != per_read]
    if expected_keys is not None:
        expected = set(map(tuple, expected_keys))
        problems += [f'{list(k)}: not read' for k in sorted(expected-set(counts))]
        problems += [f'{list(k)}: read but not expected' for k in sorted(set(counts)-expected)]              # 修订 4 次要 1
    spans = [(row['start'], row['end']) for row in record.get('truth_logical') or []]
    for row in record['opened']:
        if is_truth(row['path']) and not any(a <= row['index'] < b for a, b in spans):
            problems.append(f"{row['path']}: opened outside a logical read")
    members = record.get('truth_members') or []
    for row in record.get('truth_logical') or []:
        key = row['key']
        if len(key) == 2:                                                                      # 修订 4 次要 1：一次逻辑读只开一份真值，就是这个受试者目录下的 aligned_<种类>
            subject, kind = key
            paths = sorted({r['resolved'] for r in record['opened'] if is_truth(r['path']) and row['start'] <= r['index'] < row['end']})
            if len(paths) != 1 or Path(paths[0]).name != f'aligned_{kind}.nii.gz' or Path(paths[0]).parent.name != subject:
                problems.append(f'{key}: the logical read opened {paths}, not exactly {subject}/aligned_{kind}.nii.gz')
        elif len(key) == 1:                                                                    # 修订 5 第 2 条：开发集的键是对名，逻辑读之内读到的 *_seg* 成员都来自 <…>/<该对>/pair.npz
            pair = key[0]
            for m in members:
                if m.get('logical') == row.get('id') and (m['file'] in (None, 'None') or Path(m['file']).name != 'pair.npz' or Path(m['file']).parent.name != pair):
                    problems.append(f"{key}: the logical read read {m['file']}:{m['member']}, not a member of <…>/{pair}/pair.npz")
    # 修订 4 次要 1 的"不同的逻辑读不得打开同一路径"（修订 5 第 1 条删掉 seen 一项）：两元素键由"恰好这一个路径"与"每个键恰好一次逻辑读"一起蕴含（路径由键决定，键不同路径就不同）；
    # 一元素键（开发集）由上面的"成员都来自 <该对>/pair.npz"蕴含。
    problems += [f"{m['file']}:{m['member']}: a truth member read outside a logical read" for m in members if not m['inside']]     # 修订 4 次要 2
    return problems
