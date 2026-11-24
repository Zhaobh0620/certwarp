"""The static census of C23.6-3 (SG, D104): the reading primitives and the constructions of a subject's path, found in bayesreg/, experiments/, scripts/ and
runs/ (*.py), per (file, function).  Comments and docstrings are left out (they read nothing); every other line is matched as it is.  This is an approximation:
it cannot prove that no other way to read a subject's file exists (a renamed helper, open() on a label file under another name); what it does is make every new
match fail the census test until it is classified and reviewed."""
import ast
import io
from pathlib import Path
import re
import tokenize

SCOPE = ('bayesreg', 'experiments', 'scripts', 'runs')
PATTERNS = dict(
    # reading primitives
    np_load=r'\bnp\.load\(', nib_load=r'\bnib\.load\(', nibabel_load=r'\bnibabel\.load\(', load_scalar=r'\bload_scalar\(', read_image=r'\bread_image\(',
    read_labels=r'\bread_labels\(', read_truth=r'\bread_truth\(', read_back_probabilities=r'\bread_back_probabilities\(', read_probabilities=r'\bread_probabilities\(',
    torch_load=r'\btorch\.load\(', open_nifti_npz=r'\bopen\([^)]*(\.nii\.gz|\.npz)', read_bytes_nifti_npz=r'(\.nii\.gz|\.npz|IMAGE_FILE|LABEL_FILES).*\.read_bytes\(',
    # the path of a subject's file
    image_path=r'\.image_path\(', label_path=r'\.label_path\(', IMAGE_FILE=r'\bIMAGE_FILE\b', LABEL_FILES=r'\bLABEL_FILES\b', aligned_norm=r'aligned_norm',
    aligned_seg=r'aligned_seg', pair_npz=r'[\'"]pair\.npz[\'"]', subject_dir=r'/subject/', DATA_ROOT=r'\bDATA_ROOT\b', data_three_d=r'data/three_d',
    mnt_d_data=r'/mnt/d/data', data_three_d_parts=r'[\'"]data[\'"]\s*/\s*[\'"]three_d[\'"]', symlink=r'\bsymlink_to\(|\bos\.symlink\(',
    subprocess=r'\bsubprocess\.', popen=r'\bPopen\(',
    # bayesreg.io.pair's archive opener, called by name from outside load_pair (XB review m6)
    open_npz=r'\b_open_npz\b')
COMPILED = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}


def files(root):
    root = Path(root)
    for part in SCOPE:
        for path in sorted((root/part).rglob('*.py')):
            if '__pycache__' not in path.parts:
                yield path


def docstring_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, 'value', None), ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno+1))
    return lines


def scopes(tree):
    """[(first line, last line, qualified name)] of every function, innermost last when sorted by first line."""
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{prefix}{child.name}'
                if not isinstance(child, ast.ClassDef):
                    out.append((child.lineno, child.end_lineno, name))
                visit(child, name+'.')
            else:
                visit(child, prefix)
    visit(tree, '')
    return out


def code_lines(text):
    """{line number: the line without its comment}; None if the file does not tokenize."""
    lines = text.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row-1] = lines[row-1][:col]
    return {n+1: line for n, line in enumerate(lines)}


def scan(root):
    """{(repository-relative path, function or '<module>'): sorted pattern names}."""
    root = Path(root)
    found = {}
    for path in files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8', errors='replace')
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        lines = code_lines(text) if tree is not None else None
        if tree is None or lines is None:
            names = sorted(n for n, p in COMPILED.items() if p.search(text))
            if names:
                found[(rel, '<unparsed>')] = names
            continue
        skip = docstring_lines(tree)
        spans = scopes(tree)
        for number, line in lines.items():
            if number in skip:
                continue
            hits = [n for n, p in COMPILED.items() if p.search(line)]
            if not hits:
                continue
            inside = [s for s in spans if s[0] <= number <= s[1]]
            name = max(inside, key=lambda s: s[0])[2] if inside else '<module>'
            found.setdefault((rel, name), set()).update(hits)
    return {key: sorted(value) for key, value in found.items()}
