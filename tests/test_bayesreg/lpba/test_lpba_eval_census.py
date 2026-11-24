"""LP-C2 (contract C26.5-12 N51-N53, C26.10 T8 "准入在读之前（MANIFEST / MANUAL_ROWS 钉次序）", T14): the order "admit first, read after" of the LPBA40 evaluation entry and of the 10a
dry run, pinned by hand (as tests/test_bayesreg/lpba/test_lpba_census.py does for LP-A); the census rows of class T are the evaluation entry's only; E is unchanged."""
import ast
from pathlib import Path

from ..sg._census_inventory import INVENTORY

ROOT = Path(__file__).resolve().parents[3]
ORDER = (
    ('bayesreg/eval/lpba_eval.py', '_open_lpba', ['check_code_identity', '_admit', 'check_prereg', 'load_arm_table', 'check_cells', 'machine_gate', 'image_originals',
                                                  'check_manifest', 'allow', 'check_stage', 'check_out_dir', 'take_lock', '_issue']),
    ('bayesreg/eval/lpba_eval.py', 'open_lpba', ['start_spy', '_open_lpba']),
    ('bayesreg/eval/lpba_eval.py', 'evaluate_pair', ['take_pair_lock', 'verify_files', '_read_manifest', 'load_pair', 'read_label', 'score_d']),
    ('bayesreg/eval/lpba_eval.py', 'read_label', ['_check_admission', '_image_header', 'private_temp_dir', 'file_sha256', '_label_volume']),
    ('bayesreg/eval/lpba_eval.py', '_label_volume', ['_check_admission', 'load', 'Nifti1Image', 'save', 'load_scalar']),      # review minor 6: its own guard first
    ('experiments/lpba/admission_dry_run.py', 'run', ['preload', 'pre_manifest_function', 'Policy', 'spying', 'check']),
)


def function(rel, name):
    for node in ast.walk(ast.parse((ROOT/rel).read_text(encoding='utf-8'))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError((rel, name))


def first_call_line(node, name):
    lines = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and (getattr(n.func, 'id', None) == name or getattr(n.func, 'attr', None) == name)]
    assert lines, name
    return min(lines)


def test_the_evaluation_entry_admits_before_it_reads():
    for rel, name, calls in ORDER:
        node = function(rel, name)
        lines = [first_call_line(node, c) for c in calls]
        assert lines == sorted(lines), (rel, name, list(zip(calls, lines)))


def test_class_t_is_the_evaluation_entrys_and_e_is_unchanged():
    t_rows = {k for k, (cls, _) in INVENTORY.items() if cls == 'T'}
    assert t_rows == {('bayesreg/eval/lpba_eval.py', n) for n in ('<module>', '_label_volume', 'read_label', 'evaluate_pair')}
    assert {k for k, (cls, _) in INVENTORY.items() if cls == 'E'} == {('bayesreg/eval/holdout.py', n) for n in ('<module>', '_load_label_volume', 'read_truth', 'evaluate_pair')}
    rows = [k for k in INVENTORY if k[0] in ('bayesreg/eval/lpba_eval.py', 'experiments/lpba/feas_chain.py', 'experiments/lpba/admission_dry_run.py', 'bayesreg/eval/lpba_decide.py')]
    assert len(rows) == 9 and all(INVENTORY[k][0] in ('T', 'G', 'D') for k in rows) and INVENTORY[('bayesreg/eval/lpba_eval.py', 'spy_roots')][0] == 'D'
    label_reads = [n for n in ast.walk(ast.parse((ROOT/'bayesreg/eval/lpba_eval.py').read_text(encoding='utf-8')))
                   if isinstance(n, ast.Name) and n.id == 'LABEL_REL']
    readers = {f.name for f in ast.walk(ast.parse((ROOT/'bayesreg/eval/lpba_eval.py').read_text(encoding='utf-8'))) if isinstance(f, ast.FunctionDef)
               and any(isinstance(n, ast.Name) and n.id == 'LABEL_REL' for n in ast.walk(f))}
    assert label_reads and readers == {'read_label', '_label_volume'}      # the label's path is built in the reader only (and checked again by its raw helper, review minor 6)
