"""修订 7 第 3 条、修订 8 重要 1：experiments/r4 里每一条 import / from-import（含函数内部的）与 __import__('x')，在 PYTHONPATH=<冻结树>:<代码> 下逐个解析：模块能导入、名字存在。
另核两类真实运行才会撞上的错（测试打桩绕过的路径上也一样）：
- 模块上的属性链（`rb.write_bench_pair`、`np.lib.npyio.NpzFile`、`ch.he.settings_sha256`、`core.frozen_api`）：链首是模块别名——import 得到的，或模块级赋值得到的
  （`core = he.core`、`core, he = ch.core, ch.he`，取导入后的模块对象上的值）——且这个名字没有被任何函数的参数或局部绑定用过；链上每一段是模块时，下一段的属性必须存在；
- 调用的签名：调到模块级函数或类（经属性链、from-import 或同一文件的顶层 def）时，用 inspect.signature 绑定位置参数个数与关键字名（调用里有 ** 展开的不核，有 * 展开的只核关键字名）。
覆盖的是这三类错（导入、模块属性、签名），不是这些行的全部行为。
用法：python -B _import_check.py <代码根>（PYTHONPATH=<冻结树>:<代码根>，与 run_r4.py 的单元相同）。输出 JSON：{"checked": n, "problems": [...]}。"""
import ast
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.pop(0)                                                                         # 不让测试目录排在前面
ROOT = Path(sys.argv[1]).resolve()
R4 = ROOT/'experiments'/'r4'
sys.path.insert(0, str(R4))                                                             # 照 run_r4.py 作为脚本运行时的 sys.path[0]
FILES = sorted(R4.glob('*.py'))
for path in FILES:                                                                      # r4 代码在运行时往 sys.path 里加的目录（模块级与函数里的）
    for name in sorted(set(re.findall(r"HERE\.parent/'(\w+)'", path.read_text(encoding='utf-8')))):
        directory = str(ROOT/'experiments'/name)
        if directory not in sys.path:
            sys.path.append(directory)
modules = {path.stem: importlib.import_module(path.stem) for path in FILES}

problems, checked = [], 0


def where(path, node):
    return f'{path.name}:{node.lineno}'


def resolve_from(module_name, name):
    module = importlib.import_module(module_name)
    if hasattr(module, name):
        return getattr(module, name)
    return importlib.import_module(f'{module_name}.{name}')


def function_bindings(tree):
    """函数里绑定的名字（参数、局部赋值、for、with、except、推导式的目标、嵌套的 def / class 名）：同名的模块别名在整个文件里都不核，免得把局部变量当成模块。"""
    names = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
                names.add(node.name)
    return names


def callable_signature(obj):
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


for path in FILES:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    shadowed = function_bindings(tree)
    module_self = modules[path.stem]
    aliases = {}                                                                        # 名 → 对象（模块，或 from-import 的对象）
    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    checked += 1
                    module = importlib.import_module(alias.name)
                    if alias.asname:
                        aliases[alias.asname] = module
                    else:
                        aliases[alias.name.split('.')[0]] = importlib.import_module(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    problems.append(f'{where(path, node)}: a relative import')
                    continue
                for alias in node.names:
                    checked += 1
                    aliases[alias.asname or alias.name] = resolve_from(node.module, alias.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '__import__' and node.args and isinstance(node.args[0], ast.Constant):
                checked += 1
                importlib.import_module(node.args[0].value)
        except Exception as exc:                                                        # noqa: BLE001 - every unresolved import is a problem
            problems.append(f'{where(path, node)}: {ast.unparse(node)[:160]} -> {type(exc).__name__}: {exc}')
    for node in tree.body:                                                              # 修订 8 重要 1：模块级赋值得到的模块别名（导入后的模块对象上的值）
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in (target.elts if isinstance(target, ast.Tuple) else [target]):
                    if isinstance(name, ast.Name) and inspect.ismodule(getattr(module_self, name.id, None)):
                        aliases[name.id] = getattr(module_self, name.id)
    top_defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    def module_of(expr):
        """表达式若是模块（别名，或模块上一段段取到的模块），返回它；否则 None。"""
        if isinstance(expr, ast.Name):
            obj = aliases.get(expr.id) if expr.id not in shadowed else None
            return obj if inspect.ismodule(obj) else None
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == '__import__' and expr.args and isinstance(expr.args[0], ast.Constant):
            return importlib.import_module(expr.args[0].value)
        if isinstance(expr, ast.Attribute):
            base = module_of(expr.value)
            obj = getattr(base, expr.attr, None) if base is not None else None
            return obj if inspect.ismodule(obj) else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            target = module_of(node.value)
            if target is not None:
                checked += 1
                if not hasattr(target, node.attr):
                    problems.append(f'{where(path, node)}: {ast.unparse(node)[:120]}: {target.__name__} has no attribute {node.attr!r}')
        if isinstance(node, ast.Call):
            func, obj = node.func, None
            if isinstance(func, ast.Attribute):
                base = module_of(func.value)
                obj = getattr(base, func.attr, None) if base is not None else None
            elif isinstance(func, ast.Name) and func.id in aliases and func.id not in shadowed and not inspect.ismodule(aliases[func.id]):
                obj = aliases[func.id]
            elif isinstance(func, ast.Name) and func.id in top_defs and func.id not in shadowed:
                obj = getattr(module_self, func.id, None)
            if obj is None or inspect.ismodule(obj) or not callable(obj):
                continue
            sig = callable_signature(obj)
            if sig is None or any(k.arg is None for k in node.keywords):
                continue
            params = sig.parameters.values()
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params)
            checked += 1
            if any(isinstance(a, ast.Starred) for a in node.args):
                names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
                bad = [k.arg for k in node.keywords if k.arg not in names] if not has_var_kw else []
                if bad:
                    problems.append(f'{where(path, node)}: {ast.unparse(node.func)}(...) has no parameters {bad}')
                continue
            try:
                sig.bind(*[None]*len(node.args), **{k.arg: None for k in node.keywords})
            except TypeError as exc:
                problems.append(f'{where(path, node)}: {ast.unparse(node)[:160]} -> {exc}')
print(json.dumps(dict(checked=checked, problems=problems), ensure_ascii=False))
