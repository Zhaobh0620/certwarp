"""LP-H revision 1 (the LP-H review's minor 7; decision_log 2026-11-21 ~3:25 PM CST): a static test that the code a spied LPBA40 process runs makes no directory-descriptor-relative open.

Why: the spies of the LPBA40 entries (lpba_arms.ArmSpy and hp_core.Spy through hp_core's audit hook, xb_spy.DriverSpy and spyrun through theirs, bayesreg.eval.lpba_eval's and
make_eval_manifest_lpba's) see the audit event 'open', whose arguments are (path, mode, flags) without dir_fd: a bare name opened relative to a directory descriptor is resolved against the
working directory, and a listing by descriptor (os.scandir(fd)) is no path at all.  shutil.rmtree's fd-based walk, os.fwalk and any call with dir_fd= open that way; the feasibility run3
stop (firewall_c.prepare's shutil.rmtree of its private copy, the copy's data/ read as <checkout>/data) came from it.  tempfile.TemporaryDirectory removes its directory with shutil.rmtree:
it is held to the same rule.

What is scanned: the static import closure (every import statement, top-level or inside a function; every 'experiments/...py' path a module names, e.g. core.load_module's; every module
loaded by its name, bayesreg.eval.holdout._experiment_module("<dir>", "<name>") and lpba_eval._lpba_module("<name>")) of the entry scripts of the spied LPBA40 processes, restricted to
this code tree (bayesreg/, experiments/, jd_curl/, scripts/).  Not scanned, so not guarded (experiments/lpba/实现说明_LP-B.md, the known limits): the third-party code a method process
runs (runpy: convexAdam.convex_adam_nnUNet), the libraries of the method processes' conda environments (bl_fireants, bl_convexadam), and the copytree-then-rmtree of a shutil.move across
file systems (the closure's three shutil.move, in decide_ef, decide_mc0, decide_mc0p, do not run in a spied LPBA process).  Each file is parsed (ast); a use of rmtree, fwalk or
TemporaryDirectory, or a call with a dir_fd keyword, is a finding.  The findings must be exactly EXCEPTIONS (each with its reason), so a new one fails and a removed one is noticed."""
import ast
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'lpba', ROOT/'experiments'/'lpba'/'devcheck', ROOT/'experiments'/'he', ROOT/'experiments'/'hb', ROOT/'experiments'/'hi', ROOT/'experiments'/'xb',
           ROOT/'experiments'/'xb'/'spy', ROOT/'experiments'/'holdout', ROOT/'experiments'/'d0'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The entry scripts of the spied LPBA40 processes (prereg §8.5 item 7; experiments/lpba/实现说明_LP-B.md): the method's processes (run_hx: supervisor, t88, repair, third certification),
# the configuration C driver, the review unit and its third certification's child, the external arms' driver and its method processes' spy and scripts, LP-A's entries, the admission
# dry run, the c record, the development checks, the evaluation entry and its manifest tool.
# The method scripts are the LPBA40 arms' (xb_core.LPBA_ARMS: ants_s is a CLI arm; fireants_greedy and fireants_syn run harness/m_fireants.py; convexadam_v4seg runs m_convexadam_v4seg.py,
# which imports v4seg_convention.py) -- a test checks them against xb_core.ARMS.  harness/m_convexadam.py is not listed: it is the hold-out's descriptive ConvexAdam arm, which LPBA40 does
# not run (LPBA40 prereg §4.3), and m_convexadam_v4seg.py does not import it.  experiments/lpba/lpba_arm_table.py is loaded by the evaluation entry by its name (_lpba_module).
ENTRIES = ('experiments/he/run_hx.py', 'experiments/hb/hb_configc.py', 'experiments/hb/hb_review.py', 'experiments/xb/xb_drive.py', 'experiments/xb/spy/xb_spyrun.py',
           'experiments/xb/harness/m_fireants.py', 'experiments/xb/m_convexadam_v4seg.py', 'experiments/xb/v4seg_convention.py', 'experiments/lpba/lpba_arm_table.py',
           'experiments/lpba/prep_lpba.py', 'experiments/lpba/atropos_lpba.py',
           'experiments/hi/hi_lpba.py', 'experiments/lpba/admission_dry_run.py', 'experiments/lpba/c_record.py', 'experiments/lpba/devcheck/lpdev_a.py',
           'experiments/lpba/devcheck/lpdev_b.py', 'experiments/lpba/devcheck/lpdev_c.py', 'bayesreg/eval/lpba_eval.py', 'experiments/lpba/make_eval_manifest_lpba.py')
SCOPE = ('bayesreg', 'experiments', 'jd_curl', 'scripts')
EXCEPTIONS = {
    ('bayesreg/eval/lpba_eval.py', 'private_temp_dir', 'rmtree'):
        'a fresh tempfile.mkdtemp directory refused before anything is written into it: empty, so rmtree opens only the directory itself, by its absolute path (no bare name)',
    ('bayesreg/eval/lpba_eval.py', 'read_label', 'rmtree'):
        "the private label directory holds one file (label_mm.nii.gz) and no subdirectory: rmtree opens the directory by its absolute path (the entry's spy records it as truth) and "
        'unlinks the file by dir_fd (os.unlink: no open event), so no bare name is opened',
}
BANNED_NAMES = {'rmtree': 'rmtree', 'fwalk': 'fwalk', 'TemporaryDirectory': 'TemporaryDirectory'}
PATH_IN_CODE = re.compile(r'(?:^|[\'"/])((?:experiments|bayesreg|scripts)/[A-Za-z0-9_/\-]+\.py)')
BY_NAME = (re.compile(r'_experiment_module\(\s*[\'"]([A-Za-z0-9_]+)[\'"]\s*,\s*[\'"]([A-Za-z0-9_]+)[\'"]'),        # bayesreg.eval.holdout._experiment_module("<dir>", "<name>")
           re.compile(r'_lpba_module\(\s*[\'"]()([A-Za-z0-9_]+)[\'"]'))                                     # lpba_eval._lpba_module("<name>"): experiments/lpba/<name>.py


def module_file(name, here):
    """The file of module `name` inside the code tree, or None: importlib's finder (without importing the module itself), else a unique experiments/**/<name>.py (a directory a module
    puts on sys.path at run time), else a sibling of the importing file."""
    top = name.split('.')[0]
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None and spec.origin and spec.origin not in ('built-in', 'frozen'):
        path = Path(spec.origin).resolve()
        return path if ROOT in path.parents else None
    if '.' not in name:
        sibling = here.parent/f'{name}.py'
        if sibling.is_file():
            return sibling.resolve()
        found = sorted((ROOT/'experiments').glob(f'**/{top}.py'))
        if len(found) == 1:
            return found[0].resolve()
    return None


def imports_of(path, tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = path.parent
                for _ in range(node.level-1):
                    package = package.parent
                base = '.'.join(package.relative_to(ROOT).parts)
                module = f'{base}.{node.module}' if node.module else base
            else:
                module = node.module or ''
            names.add(module)
            names |= {f'{module}.{a.name}' for a in node.names}
    return names


def closure():
    """{repository-relative path: ast tree} of the entries and everything they import inside the code tree (statically)."""
    seen, todo = {}, [ROOT/e for e in ENTRIES]
    while todo:
        path = todo.pop().resolve()
        rel = str(path.relative_to(ROOT))
        if rel in seen or not path.is_file() or rel.split('/')[0] not in SCOPE:
            continue
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text, filename=rel)
        seen[rel] = tree
        for name in imports_of(path, tree):
            found = module_file(name, path)
            if found is not None:
                todo.append(found)
        for match in PATH_IN_CODE.finditer(text):
            candidate = ROOT/match.group(1)
            if candidate.is_file():
                todo.append(candidate)
        for pattern in BY_NAME:
            for directory, name in pattern.findall(text):
                candidate = ROOT/'experiments'/(directory or 'lpba')/f'{name}.py'
                if candidate.is_file():
                    todo.append(candidate)
    return seen


def function_of(tree, lineno):
    best, span = '<module>', None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= lineno <= node.end_lineno:
            if span is None or node.end_lineno-node.lineno < span:
                best, span = node.name, node.end_lineno-node.lineno
    return best


def findings(rel, tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            out.add((rel, function_of(tree, node.lineno), BANNED_NAMES[node.attr]))
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            out.add((rel, function_of(tree, node.lineno), BANNED_NAMES[node.id]))
        elif isinstance(node, ast.ImportFrom) and any(a.name in BANNED_NAMES for a in node.names):
            out |= {(rel, function_of(tree, node.lineno), BANNED_NAMES[a.name]) for a in node.names if a.name in BANNED_NAMES}
        elif isinstance(node, ast.Call) and any(k.arg == 'dir_fd' for k in node.keywords):
            out.add((rel, function_of(tree, node.lineno), 'dir_fd'))
    return out


def test_no_spied_lpba_entry_opens_relative_to_a_directory_descriptor():
    import run_hx  # noqa: F401  (the entries put their directories on sys.path when imported, as at run time)
    import hb_configc  # noqa: F401
    import hb_review  # noqa: F401
    import xb_drive  # noqa: F401
    import lpba_arms  # noqa: F401
    files = closure()
    for must in ('experiments/he/firewall_c.py', 'experiments/holdout/hp_core.py', 'experiments/lpba/lpba_arms.py', 'experiments/lpba/lpba_common.py', 'experiments/he/he_core.py',
                 'experiments/hb/hb_core.py', 'experiments/xb/xb_core.py', 'experiments/xb/xb_spy.py', 'bayesreg/io/sealed.py', 'bayesreg/io/pair.py', 'experiments/d0/check_prepared.py',
                 *ENTRIES):
        assert must in files, f'{must} is not in the closure (the scan would not see it)'
    found = set().union(*(findings(rel, tree) for rel, tree in files.items()))
    assert found == set(EXCEPTIONS), (f'new: {sorted(found-set(EXCEPTIONS))}', f'no longer found (update EXCEPTIONS): {sorted(set(EXCEPTIONS)-found)}')
    assert all(EXCEPTIONS.values())


def test_the_entries_hold_every_lpba_arms_method_script_and_the_modules_loaded_by_name():
    """ENTRIES against xb_core.ARMS (the script of each LPBA40 arm, relative to experiments/xb) and the closure against the modules loaded by name: lpba_arm_table (lpba_eval's
    _lpba_module), holdout_common (holdout._experiment_module)."""
    import xb_core as xc
    scripts = {f'experiments/xb/{xc.ARMS[a]["script"]}' for a in xc.LPBA_ARMS if xc.ARMS[a].get('script')}
    assert scripts == {'experiments/xb/harness/m_fireants.py', 'experiments/xb/m_convexadam_v4seg.py'} and scripts <= set(ENTRIES)
    assert 'experiments/xb/harness/m_convexadam.py' not in ENTRIES and not [a for a in xc.LPBA_ARMS if xc.ARMS[a].get('script') == 'harness/m_convexadam.py']
    import run_hx  # noqa: F401
    files = closure()
    for must in ('experiments/lpba/lpba_arm_table.py', 'experiments/xb/v4seg_convention.py', 'experiments/holdout/holdout_common.py', 'experiments/lpba/lpba_common.py'):
        assert must in files, must
    text = (ROOT/'bayesreg/eval/lpba_eval.py').read_text(encoding='utf-8')
    assert ('', 'lpba_arm_table') in BY_NAME[1].findall(text) and ('', 'lpba_common') in BY_NAME[1].findall(text)


def test_the_scan_sees_a_rmtree_fwalk_and_dir_fd(tmp_path):
    """The rule itself, on synthetic sources: each banned form is a finding where it stands."""
    source = ('import shutil, os, tempfile\nfrom shutil import rmtree as r\n'
              'def a(p):\n    shutil.rmtree(p)\n'
              'def b(p):\n    for _ in os.fwalk(p):\n        pass\n'
              'def c(p, fd):\n    return os.open(p, os.O_RDONLY, dir_fd=fd)\n'
              'def d():\n    with tempfile.TemporaryDirectory() as t:\n        return t\n'
              'def e(p):\n    os.unlink(p)\n')
    got = findings('x.py', ast.parse(source))
    assert got == {('x.py', '<module>', 'rmtree'), ('x.py', 'a', 'rmtree'), ('x.py', 'b', 'fwalk'), ('x.py', 'c', 'dir_fd'), ('x.py', 'd', 'TemporaryDirectory')}
