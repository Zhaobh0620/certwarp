"""R4 的防火墙（合同 C25.6-10、C25.10-5、-6）：不得调用的 HS 函数（静态核对，做传递闭包）、整体不 import decide_hs（子进程里核 sys.modules）、spy 的白名单。
只读源代码与 tmp 里的文件。"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
R4 = ROOT/'experiments'/'r4'
for _p in ('r4', 'he', 'mc0', 'ef', 'fl'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_common as rc  # noqa: E402
import r4_spy as spy  # noqa: E402

SEARCH = ('r4', 'he', 'd0', 'fl', 'ef', 'mc0', 'hs', 'holdout', 'p2', 'seed', 'unv2', 'nn')      # r4_chain、r4_bench 的 sys.path 次序（r4 在前）
FORBIDDEN = {('decide_hs', '*'), ('run_hs', 'decide_fallback'), ('run_hs', 'fallback_unit'), ('run_hs', 'id_evaluate'), ('rp_core', 'fallback_status'), ('run_fl', 'evaluate_field'),
             ('run_fl', 'evaluate_repair'), ('run_fl', 'evaluate_all'), ('run_fl', 'repair_one'), ('run_fl', 'repair_allowed'), ('fl_core', 'chosen_repairs')}


def module_file(name):
    for d in SEARCH:
        path = ROOT/'experiments'/d/f'{name}.py'
        if path.is_file():
            return path
    return None


class Module:
    """一个 experiments 模块的函数（含类的方法，以 Class.method 记）与它的别名表（import x as y、from x import f）。"""
    cache = {}

    @classmethod
    def of(cls, name):
        if name not in cls.cache:
            path = module_file(name)
            cls.cache[name] = None if path is None else cls(name, path)
        return cls.cache[name]

    def __init__(self, name, path):
        self.name, self.tree = name, ast.parse(path.read_text(encoding='utf-8'))
        self.functions, self.aliases, self.names = {}, {}, {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.aliases[a.asname or a.name.split('.')[0]] = a.name.split('.')[-1] if a.asname else a.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    self.names[a.asname or a.name] = (node.module.split('.')[-1], a.name)
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.functions[f'{node.name}.{item.name}'] = item
                        self.functions.setdefault(item.name, item)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                for t in node.targets:                                         # core = he.core 之类的模块别名
                    if isinstance(t, ast.Name) and node.value.value.id in self.aliases:
                        self.aliases[t.id] = node.value.attr if Module.of(node.value.attr) else self.aliases.get(node.value.value.id)

    def calls(self, node):
        """(module, function) of every call in `node`：alias.f(...)、f(...)（本模块的或 from-import 的）、alias.f 作为值传出去（如 lambda、回调）。"""
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in self.aliases:
                out.add((self.aliases[n.value.id], n.attr))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in self.functions:
                    out.add((self.name, n.id))
                elif n.id in self.names:
                    out.add(self.names[n.id])
        return out


def closure():
    """从 experiments/r4 的每个函数（与模块体）出发，沿 experiments 模块里的函数走到底；返回走到的 (module, function)。"""
    seen, todo = set(), []
    for path in sorted(R4.glob('*.py')):
        m = Module.of(path.stem)
        todo += list(m.calls(m.tree))
    while todo:
        item = todo.pop()
        if item in seen:
            continue
        seen.add(item)
        m = Module.of(item[0])
        if m is None or item[1] not in m.functions:
            continue
        todo += list(m.calls(m.functions[item[1]]))
    return seen


def test_no_forbidden_hs_function_is_reachable():
    reached = closure()
    bad = sorted(i for i in reached if i in FORBIDDEN or (i[0], '*') in FORBIDDEN)
    assert not bad, bad
    assert ('run_hx', 'identity_fallback') in reached and ('run_fl', 'field_metrics') in reached and ('he_core', 'repair_decision') in reached     # 正条件：图走得到真用的函数


def test_the_scan_finds_a_forbidden_call(tmp_path, monkeypatch):
    sample = tmp_path/'sample_r4.py'
    sample.write_text('import run_fl\nimport decide_hs as d\ndef f():\n    return run_fl.repair_one\ndef g():\n    d.classify(1)\n', encoding='utf-8')
    m = Module('sample_r4', sample)
    assert ('run_fl', 'repair_one') in m.calls(m.functions['f']) and ('decide_hs', 'classify') in m.calls(m.functions['g'])


def test_no_r4_module_imports_decide_hs_statically():
    for path in sorted(R4.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]+[n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not [x for x in names if x and 'decide_hs' in x], path.name


IMPORT_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1]+'/experiments/r4')
import r4_common, r4_decide, r4_lock, r4_options, r4_bench, r4_chain, r4_eval, r4_spy, r4_steps, run_r4
import r4_options as o
print(json.dumps(sorted(m for m in sys.modules if 'decide_hs' in m or m in ('run_hs', 'rp_core'))))
'''


def test_no_unit_imports_decide_hs_at_run_time(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([sys.executable, '-B', '-c', IMPORT_SCRIPT, str(ROOT)], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert 'decide_hs' not in loaded and '_r2_decide_hs' not in loaded, loaded


# ---- spy 的白名单（C25.10-5，修订 10e） -----------------------------------------------------------------------------------------------------------
def bench_ctx(root, pair='x__y', arm='CTL'):
    return dict(role='bench', root=str(root), arm=arm, pair=pair, stage_dir=str(root/'bench_stage'), out=str(root/'runs'/arm/pair),
                paths=dict(pred_root=str(root/'seg'/rc.BENCH_SEGMENTER_ID/'pred'), atropos_root=str(root/'atropos'), cache=str(root/'cache')))


def test_the_bench_whitelist():
    root = rc.R4_ROOT
    rules = spy.pair_rules(bench_ctx(root), 't88')
    ok = [ROOT/'experiments/he/he_core.py', ROOT/rc.ALLOWLIST_REL, ROOT/rc.BENCH_REGISTRY_REL, ROOT/rc.FOLDS_REL, ROOT/'configs/bayesreg/structure_groups_v1.json',
          root/'bench_stage/x__y/pair.npz', root/'runs/CTL/x__y/t88/run.json', root/'seg'/rc.BENCH_SEGMENTER_ID/'pred/x/probs36.npz', root/'atropos/y/provenance.json',
          root/'configC/x__y/fallback.json', root/'final/CTL/x__y/final.json']
    bad = [rc.DATA_ROOT/'OASIS_OAS1_0001_MR1/aligned_seg35.nii.gz', rc.DATA_ROOT/'OASIS_OAS1_0001_MR1/aligned_seg4.nii.gz', ROOT/'docs/decision_log.md',
           rc.RESULTS/'hs_v1/analysis/decision_hs.json', rc.RESULTS/'r3_v1/composition.json', rc.RESULTS/'seg/unet35_v4_ens/pred/x/probs36.npz', root/'eval/bench_eval_1.json',
           rc.DATA_LINK_TARGET/'three_d/stage_L2R_val_v1/x__y/pair.npz', ROOT/'runs/20261111-r3/result.md',
           rc.DATA_ROOT/'OASIS_OAS1_0001_MR1/aligned_norm.nii.gz',                              # 配准一侧只读 stage（复核次要 7）
           root/'bench_stage/x__z/pair.npz', root/'seg'/rc.BENCH_SEGMENTER_ID/'pred/z/probs36.npz', root/'atropos/z/provenance.json', root/'runs/CTL/x__z/t88/run.json',
           root/'runs/A1/x__y/t88/run.json', root/'configC/x__z/fallback.json']
    assert all(rules.judge(p)[1] for p in ok), [(p, rules.judge(p)) for p in ok if not rules.judge(p)[1]]
    assert not any(rules.judge(p)[1] for p in bad), [(p, rules.judge(p)) for p in bad if rules.judge(p)[1]]
    assert rules.judge('/usr/lib/python3.10/os.py')[0] is False                                    # 根之外只记录不判


def test_the_development_feasibility_categories():
    root = rc.R4_ROOT
    pair = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'
    ctx = dict(role='dev', root=str(root), arm='B', pair=pair, stage_dir=str(ROOT/rc.DEV_STAGE_REL), out=str(root/'dev_runs/B'/pair),
               paths=dict(pred_root=str(rc.HS_PRED_ROOT), atropos_root=str(rc.DEV_ATROPOS_ROOT), cache=str(root/'cache')))
    rules = spy.pair_rules(ctx, 't88')
    assert rules.judge(ROOT/rc.DEV_STAGE_REL/pair/'pair.npz')[1] and rules.judge(rc.DEV_ATROPOS_ROOT/'OASIS_OAS1_0439_MR1/posteriors01.nii.gz')[1]
    assert rules.judge(rc.HS_PRED_ROOT/'OASIS_OAS1_0438_MR1/probs36.npz')[1] and rules.judge(ROOT/rc.HS_REGISTRY_REL)[1] and rules.judge(root/'dev_configC'/pair/'fallback.json')[1]
    assert not rules.judge(rc.DATA_ROOT/'OASIS_OAS1_0438_MR1/aligned_norm.nii.gz')[1]                 # 开发集的配准一侧只读 stage
    assert not rules.judge(rc.HS_ROOT/'runs/RSHx24_c600/0438-0439/repair_metrics.json')[1]           # 评测记录
    assert not rules.judge(root/'runs/CTL/x/t88/run.json')[1]                                        # 台的目录不在开发集的白名单里
    assert not rules.judge(rc.DEV_ATROPOS_ROOT/'dice_vs_truth_seg4.json')[1] and not rules.judge(rc.HS_PRED_ROOT/'OASIS_OAS1_0440_MR1/probs36.npz')[1]     # 复核次要 7
    assert not rules.judge(ROOT/rc.DEV_STAGE_REL/'OASIS_OAS1_0440_MR1__OASIS_OAS1_0441_MR1/pair.npz')[1]


def test_the_unit_spy_catches_a_truth_file_and_requires_its_positive_condition(tmp_path):
    rules = spy.bench_rules(tmp_path/'r4', unit='t', stage_dir=tmp_path/'r4/bench_stage', pred_root=tmp_path/'r4/seg', atropos_root=tmp_path/'r4/atropos')
    rules.allowed_dirs.append((tmp_path/'r4/bench_stage').resolve())
    rules.roots = [tmp_path]
    rules.image_root = tmp_path/'data'
    (tmp_path/'data/S').mkdir(parents=True)
    (tmp_path/'data/S/aligned_seg35.nii.gz').write_bytes(b'x')
    (tmp_path/'elsewhere.txt').write_text('x')
    with spy.UnitSpy(rules, expect_reads=False) as s:
        open(tmp_path/'data/S/aligned_seg35.nii.gz', 'rb').close()
    record = s.record()
    assert record['passed'] is False and record['truth_opened']
    with spy.UnitSpy(rules, expect_reads=False) as s:
        open(tmp_path/'elsewhere.txt').close()
    assert s.record()['passed'] is False                                                            # 根之内、白名单之外
    import numpy as np
    original = tmp_path/'r4/bench_stage/p/pair.npz'
    original.parent.mkdir(parents=True)
    np.savez(original, fixed=np.zeros(2), moving=np.zeros(2), fixed_seg35=np.zeros(2))
    with spy.UnitSpy(rules, original) as s:
        pass
    assert s.record()['passed'] is False                                                            # 一次也没读到 fixed、moving：正条件不成立
    with spy.UnitSpy(rules, original) as s:
        with pytest.raises(Exception):
            np.load(original)['fixed_seg35']
    assert s.record()['he_spy']['forbidden_access']
    rules.allow_truth = True
    import types
    reader = types.ModuleType('_r4_fw_reader')

    def read(subject, kind='seg35'):
        for _ in range(3):                                                                  # one logical read opens the file several times (nibabel, sha256)
            open(tmp_path/'data'/subject/f'aligned_{kind}.nii.gz', 'rb').close()
    reader.read = read
    sys.modules['_r4_fw_reader'] = reader
    try:
        with spy.UnitSpy(rules, expect_reads=False, truth_readers=(('_r4_fw_reader', 'read', lambda a, k: (a[0], k.get('kind', 'seg35'))),)) as s:
            reader.read('S')
        record = s.record()
        assert record['passed'] is True and spy.truth_reads(record, expected_keys=[('S', 'seg35')]) == []
        with spy.UnitSpy(rules, expect_reads=False, truth_readers=(('_r4_fw_reader', 'read', lambda a, k: (a[0], k.get('kind', 'seg35'))),)) as s:
            reader.read('S')
            reader.read('S')
            open(tmp_path/'data/S/aligned_seg35.nii.gz', 'rb').close()                        # outside any logical read
        problems = spy.truth_reads(s.record())
        assert any('2 logical reads' in x for x in problems) and any('outside a logical read' in x for x in problems)
    finally:
        del sys.modules['_r4_fw_reader']
