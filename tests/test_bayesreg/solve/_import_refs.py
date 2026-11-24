"""Contract C23 revision 6o, its last open item (the lead's ruling 11 of 11-14; P0S): the "not referred to" check of solve/test_hs_tools.py on the AST, in place of the word list that missed
`from .. import decode`, `from ..decode import`, several names on one line and imports in parentheses over several lines (R3b difference review n1).

What a file refers to (references(path, root)), each with its line:
  - every import statement anywhere in the file (module level, functions, try / if blocks): `import a.b.c [as x]` -> a.b.c; `from X import n1, n2` -> X, X.n1, X.n2 (a name may be a
    submodule; `*` gives X only); a relative X resolved against the file's own package with importlib.util.resolve_name (the interpreter's rule; one that climbs above the top is reported as
    '<unresolvable relative import>');
  - the dynamic imports, called by name or as an attribute (P0S revision 2, the code review's I1):
      * by module name (NAME_IMPORTS: import_module, __import__, run_module, find_spec, reload): the name argument must be a plain str constant, else '<dynamic>' -- an f-string, a
        concatenation ('bayesreg.seg.' + x), a variable are all '<dynamic>'; a relative name is resolved against the package argument (which must then be a str constant, else '<dynamic>') or
        the file's package; __import__'s fromlist (a list or tuple of str constants, else '<dynamic>') adds <name>.<item>, its level (an int constant, else '<dynamic>') makes the name
        relative;
      * by path (PATH_IMPORTS: spec_from_file_location's location, run_path, SourceFileLoader's path): the path's last segment must be a str constant (a constant, the right side of / or +,
        the last part of an f-string, the last argument of a call such as Path(...) or os.path.join(...)), else '<dynamic>'; the path's constants are joined into a dotted name, and the
        last segment's stem is matched against the targets' last components ('<path>decode' reaches bayesreg.seg.decode wherever the directory comes from: experiments/hs/run_hs.py's
        EXPERIMENTS/'fl'/'run_fl.py' resolves to run_fl and reaches nothing);
        the file name must be a constant as a whole (P0 review docs/bayesreg/reviews/P0_复核_20261115.md, minor 5: a last constant that is only the suffix '.py' gave the empty name and
        reached nothing): the right side of + and the last part of an f-string count only when the name cannot start before them (they hold a separator, the left side of + ends with
        one, or the f-string has no other part), so f'{name}.py', str(D/name) + '.py' and f'{D}/{x}_decide.py' are '<dynamic>'; with_suffix(c) is its receiver's last segment with
        the suffix c ((D/name).with_suffix('.py') is '<dynamic>', (D/'decode').with_suffix('.py') reaches decode); an empty name ('.py' alone) is '<dynamic>';
    '<dynamic>' is unresolved and reaches every target: it is reported (fail closed);
  - so that nothing the word list caught is lost: every dotted attribute chain of plain names (bayesreg.seg.decode.decode_map), every dotted name or path inside a string constant that is not
    a docstring (TOKEN, normalised as above), and every identifier of `identifiers` (a function name of a target, e.g. train_small_member).
Comments and docstrings are not statements and refer to nothing.
Known blind spots (the code review's minor 8; not caught, stated here): an attribute of an alias (`import bayesreg.seg as s; s.decode`), getattr with a constant name
(`getattr(bayesreg.io, 'code_identity')`), a bare one-component import after a sys.path change (`sys.path.insert(0, 'bayesreg/seg'); import decode`), exec or eval of source text; and (the T1
review's minor 3; the module before T1 missed it too; a contrived spelling, recorded, not fixed) a by-path import whose + has a constant left side ending in a separator that pathlib drops:
`str(D/'de/') + 'code.py'` is .../decode.py, but the rule trusts 'de/' to end with a separator and reads the name code.py, which reaches nothing.  Known false
positives (reported, so the test fails closed; acceptable): an attribute chain that only looks like a target (`seg.decode('utf-8')` on a variable named seg), a string naming one.

hits(name, targets): the targets of which a dotted tail of two components or more (the whole target included: bayesreg.seg.decode, seg.decode; eval.holdout, io.code_identity, unet.small) is
a run of whole components of the name -- so the target itself and any name inside it (bayesreg.seg.decode.decode_map), a string or an attribute chain naming it, and an import found through
another sys.path entry (seg.decode).  Components are compared whole: bayesreg.seg.decoder, eval.holdout_report are not hits; an unresolved name hits every target."""
import ast
import importlib.util
from pathlib import Path, PurePosixPath
import re

TOKEN = re.compile(r'[A-Za-z_]\w*(?:[./\\][A-Za-z_]\w*)+')         # a dotted name or a path inside a string (each one matched on its own)
NAME_IMPORTS = dict(import_module=((0, 'name'), (1, 'package')), __import__=((0, 'name'), None), run_module=((0, 'mod_name'), None), find_spec=((0, 'name'), (1, 'package')),
                    reload=((0, 'module'), None))               # (the name argument, the package argument): (position, keyword)
PATH_IMPORTS = dict(spec_from_file_location=(1, 'location'), run_path=(0, 'path_name'), SourceFileLoader=(1, 'path'))      # the path argument: (position, keyword)
DYNAMIC_IMPORTS = tuple(NAME_IMPORTS)+tuple(PATH_IMPORTS)
UNRESOLVED = ('<dynamic>', '<unresolvable relative import>')
PATH_MARK = '<path>'
SEPARATORS = ('/', '\\')


def module_of(root, path):
    """(module, package) of a file of the tree under `root`: bayesreg/seg/decode.py -> (bayesreg.seg.decode, bayesreg.seg); bayesreg/seg/__init__.py -> (bayesreg.seg, bayesreg.seg); a script
    of experiments/hs -> (experiments.hs.<name>, experiments.hs)."""
    parts = list(Path(path).resolve().relative_to(Path(root).resolve()).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
        return '.'.join(parts), '.'.join(parts)
    return '.'.join(parts), '.'.join(parts[:-1])


def dotted(value):
    return value.strip().replace('\\', '/').removesuffix('.py').replace('/', '.').strip('.')


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _chain(node):
    """a.b.c of an attribute chain on a plain name, else None."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return '.'.join([node.id, *reversed(names)])


def _resolve(relative, package):
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return '<unresolvable relative import>'


def _arg(call, where):
    """The call's argument at (position, keyword), or None."""
    if where is None:
        return None
    position, keyword = where
    if position < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[:position+1]):
        return call.args[position]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _is_str(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _has_separator(value):
    return any(s in value for s in SEPARATORS)


def _last_segment(node):
    """The str constant that ends a path expression and holds its whole file name, or None (module docstring; P0 review minor 5)."""
    if _is_str(node):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _last_segment(node.right)                                   # a whole component
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        right = _last_segment(node.right)
        if right is None or _has_separator(right):
            return right
        left = _last_segment(node.left)
        return right if left is not None and left.endswith(SEPARATORS) else None      # else the name starts on the left: str(D/name) + '.py'
    if isinstance(node, ast.JoinedStr) and node.values:
        last = _last_segment(node.values[-1])
        return last if last is None or _has_separator(last) or len(node.values) == 1 else None     # f'{name}.py': the name starts in a formatted part
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'with_suffix':
        base = _last_segment(node.func.value)
        if base is None or len(node.args) != 1 or node.keywords or not _is_str(node.args[0]):
            return None
        try:
            return str(PurePosixPath(base.replace('\\', '/')).with_suffix(node.args[0].value))
        except ValueError:                                                 # an empty name, or not a suffix
            return None
    if isinstance(node, ast.Call) and node.args and not node.keywords:
        return _last_segment(node.args[-1])
    return None


def _strings_in_order(node):
    constants = [c for c in ast.walk(node) if _is_str(c)]
    return [c.value for c in sorted(constants, key=lambda c: (c.lineno, c.col_offset))]       # source order (ast.walk is breadth first: a/'b'/'c' would come reversed)


def _name_import(call, kind, package):
    """The names a by-name dynamic import reaches, or ['<dynamic>'] (module docstring)."""
    where_name, where_package = NAME_IMPORTS[kind]
    name = _arg(call, where_name)
    if not _is_str(name):
        return ['<dynamic>']
    value = name.value
    anchor = package
    given = _arg(call, where_package)
    if given is not None and not (isinstance(given, ast.Constant) and given.value is None):
        if not _is_str(given):
            return ['<dynamic>']
        anchor = given.value
    if kind == '__import__':
        level = _arg(call, (4, 'level'))
        if level is not None:
            if not (isinstance(level, ast.Constant) and isinstance(level.value, int)):
                return ['<dynamic>']
            value = '.'*level.value+value if level.value > 0 else value
    base = _resolve(value, anchor) if value.startswith('.') else value
    out = [base]
    if kind == '__import__' and base not in UNRESOLVED:
        fromlist = _arg(call, (3, 'fromlist'))
        if fromlist is not None and not (isinstance(fromlist, ast.Constant) and fromlist.value is None):
            if not (isinstance(fromlist, (ast.List, ast.Tuple)) and all(_is_str(e) for e in fromlist.elts)):
                return ['<dynamic>']
            out += [f'{base}.{e.value}' for e in fromlist.elts]
    return out


def _path_import(call, kind):
    """The names a by-path dynamic import reaches, or ['<dynamic>'] (module docstring)."""
    location = _arg(call, PATH_IMPORTS[kind])
    last = None if location is None else _last_segment(location)
    stem = None if last is None else Path(last.replace('\\', '/')).name.removesuffix('.py')
    if not stem:                                    # no constant ends the path, the name's start is not constant, or the name is empty ('.py' alone: P0 review minor 5)
        return ['<dynamic>']
    return ['.'.join(dotted(s) for s in _strings_in_order(location) if dotted(s)), PATH_MARK+stem]


def references(path, root, *, identifiers=()):
    """[(line, how, name)] of the file (module docstring above: how is import, dynamic, attribute, string or identifier)."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    _, package = module_of(root, path)
    docstrings = _docstring_nodes(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, 'import', alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve('.'*node.level+(node.module or ''), package) if node.level else node.module
            out.append((node.lineno, 'import', base))
            if base not in UNRESOLVED:
                out += [(node.lineno, 'import', f'{base}.{alias.name}') for alias in node.names if alias.name != '*']
            out += [(node.lineno, 'identifier', alias.name) for alias in node.names if alias.name in identifiers]
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
            if name in NAME_IMPORTS:
                out += [(node.lineno, 'dynamic', n) for n in _name_import(node, name, package)]
            elif name in PATH_IMPORTS:
                out += [(node.lineno, 'dynamic', n) for n in _path_import(node, name)]
        elif isinstance(node, ast.Attribute):
            chain = _chain(node)
            if chain is not None and '.' in chain:
                out.append((node.lineno, 'attribute', chain))
            if node.attr in identifiers:
                out.append((node.lineno, 'identifier', node.attr))
        elif isinstance(node, ast.Name) and node.id in identifiers:
            out.append((node.lineno, 'identifier', node.id))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out += [(node.lineno, 'string', dotted(token)) for token in TOKEN.findall(node.value)]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in identifiers:
            out.append((node.lineno, 'identifier', node.name))
    return out


def hits(name, targets):
    """The targets `name` reaches (module docstring above).  A longer tail is a run only if its last two components are, so the last two decide."""
    if name in UNRESOLVED:
        return list(targets)
    if name.startswith(PATH_MARK):                  # a by-path import: its file's stem against the targets' last components
        return [target for target in targets if target.split('.')[-1] == name[len(PATH_MARK):]]
    parts = [p for p in name.split('.') if p]
    pairs = set(zip(parts, parts[1:]))
    return [target for target in targets if tuple(target.split('.')[-2:]) in pairs]


def module_name(rel):
    """bayesreg/seg/decode.py -> bayesreg.seg.decode."""
    return '.'.join(Path(rel).with_suffix('').parts)


def added_reached(root, added, *, tops, identifiers=()):
    """{added file, or '<how>:<name>' for an identifier or an unresolved reference: sorted referrers} -- the added files (`added`, relative paths) that the other files of `tops` refer to, closed
    over the added files reached: an added file reached refers on (its references to itself aside).  {} when none is reached."""
    root = Path(root)
    by_module = {module_name(rel): rel for rel in added}
    targets = tuple(by_module)
    reached = {}
    for rel, _, how, name, target in referring(root, tops, targets, skip=set(added), identifiers=identifiers):
        key = by_module[target] if target in by_module and name not in UNRESOLVED else f'{how}:{name}'
        reached.setdefault(key, set()).add(rel)
    frontier = [k for k in reached if k in added]
    while frontier:
        source = frontier.pop()
        for _, how, name in references(root/source, root, identifiers=identifiers):
            found = [f'{how}:{name}'] if how == 'identifier' or name in UNRESOLVED else [by_module[t] for t in hits(name, targets)]
            for key in found:
                if key == source:
                    continue
                if key in added and key not in reached:
                    frontier.append(key)
                reached.setdefault(key, set()).add(source)
    return {k: sorted(v) for k, v in reached.items()}


def referring(root, tops, targets, *, skip=(), identifiers=()):
    """[(file, line, how, name, target)] over the *.py files of `tops` (directories under root), the files of `skip` (relative paths) left out: every reference that reaches a target, every
    unresolved one, every identifier of `identifiers`."""
    root = Path(root)
    rows = []
    for top in tops:
        for path in sorted((root/top).rglob('*.py')):
            rel = str(path.relative_to(root))
            if rel in skip or '__pycache__' in path.parts:
                continue
            for line, how, name in references(path, root, identifiers=identifiers):
                found = [name] if how == 'identifier' else hits(name, targets)
                rows += [(rel, line, how, name, t) for t in found]
    return sorted(set(rows))
