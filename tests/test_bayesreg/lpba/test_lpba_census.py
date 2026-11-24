"""LP-A (contract C26.5-12, C26.10 T8 "准入在读之前（MANUAL_ROWS 钉次序）"): the order "admit first, read after" of the LPBA entries of this package, pinned by hand as
tests/test_bayesreg/sg/test_sg_census.py MANUAL_ROWS does (that file is not changed by this package: the old tests C26.10 and its revision 11c allow to change are
test_pair.py:198-200 and the four cases of test_sg_behaviour.py and test_hi_tools.py, and test_sg_census.py is not among them); and the census rows exist."""
import ast
from pathlib import Path

from ..sg._census_inventory import INVENTORY

ROOT = Path(__file__).resolve().parents[3]

# (file, function, the calls in the order they must first appear)
ORDER = (
    ('experiments/lpba/prep_lpba.py', 'run', ['formal_code_problem', 'admit', 'load_c_record', 'check_hold', 'machine_gate', 'take_lock', 'allow_read', 'prepare']),
    ('experiments/lpba/prep_lpba.py', 'prepare', ['guard_write', 'mm_copy', 'write_pair', 'read_back']),
    ('experiments/lpba/atropos_lpba.py', 'run', ['formal_code_problem', 'admit', 'check_hold', 'check_stage', 'LpbaEntries', 'precheck', 'machine_gate', 'take_lock', 'segment_subjects', 'wm_ratio']),
    ('experiments/hi/hi_lpba.py', 'run', ['formal_code_problem', 'admit', 'check_registry', 'check_contact', 'check_hold', 'select_models', 'check_decision', 'plan_of', 'check_stage',
                                          'machine_gate', 'take_lock', 'stage_arrays', 'predict']),
    ('experiments/lpba/lpba_common.py', 'admit', ['check_pins', 'read_lists', 'load_lpba_allowlist', 'admit_pairs', 'admit_subjects']),
    ('experiments/lpba/lpba_common.py', 'admit_subjects', ['table', 'guard_sealed_lists', 'check_subject']),
)


def function(rel, name):
    tree = ast.parse((ROOT/rel).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError((rel, name))


def first_call_line(node, name):
    lines = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and (getattr(n.func, 'id', None) == name or getattr(n.func, 'attr', None) == name)]
    assert lines, name
    return min(lines)


def test_the_entries_admit_before_they_read():
    for rel, name, calls in ORDER:
        node = function(rel, name)
        lines = [first_call_line(node, c) for c in calls]
        assert lines == sorted(lines), (rel, name, list(zip(calls, lines)))


def test_the_census_rows_of_lp_a_are_registered():
    rows = [k for k in INVENTORY if k[0] in ('experiments/lpba/lpba_common.py', 'experiments/lpba/prep_lpba.py', 'experiments/lpba/atropos_lpba.py', 'experiments/hi/hi_lpba.py')]
    assert len(rows) == 15 and all(INVENTORY[k][0] in ('G', 'D') for k in rows)
