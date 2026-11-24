"""LP-T0 item 1: splits/lpba40_test_v1/ -- the four files of the preregistration §3.1 and §7, as runs/20261119-lpba/make_lpba40_splits.py writes them, byte for byte.
No list of splits/bayesreg_v1/ is read here."""
import hashlib
import importlib.util
from pathlib import Path

from bayesreg.io.guard import CANONICAL_SUBJECT_PATTERNS, NEVER_ALLOWLISTED

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT/'splits'/'lpba40_test_v1'


def make_module():
    spec = importlib.util.spec_from_file_location('make_lpba40_splits', ROOT/'runs'/'20261119-lpba'/'make_lpba40_splits.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(name):
    lines = (SPLIT/name).read_text(encoding='utf-8').split('\n')
    assert lines[0] == 'fixed\tmoving' and lines[-1] == ''
    return [tuple(line.split('\t')) for line in lines[1:-1]]


def test_the_files_are_the_generators_byte_for_byte(tmp_path):
    module = make_module()
    assert sorted(p.name for p in SPLIT.iterdir()) == ['SHA256SUMS', 'feasibility_pairs.tsv', 'test_pairs.tsv', 'test_subjects.txt']
    assert {name: (SPLIT/name).read_bytes() for name in module.contents()} == module.contents()
    module.write(tmp_path/'again')
    assert {p.name: p.read_bytes() for p in (tmp_path/'again').iterdir()} == module.contents()


def test_sha256sums_lists_the_three_lists():
    lines = (SPLIT/'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    assert [line.split('  ')[1] for line in lines] == ['feasibility_pairs.tsv', 'test_pairs.tsv', 'test_subjects.txt']
    for line in lines:
        digest, name = line.split('  ')
        assert hashlib.sha256((SPLIT/name).read_bytes()).hexdigest() == digest


def test_the_test_set_is_s03_to_s36_in_17_disjoint_ordered_pairs():
    subjects = (SPLIT/'test_subjects.txt').read_text(encoding='utf-8').splitlines()
    assert subjects == [f'S{n:02d}' for n in range(3, 37)] and len(subjects) == 34
    assert all(CANONICAL_SUBJECT_PATTERNS['LPBA40'].fullmatch(s) for s in subjects) and not set(subjects) & NEVER_ALLOWLISTED['LPBA40']
    pairs = rows('test_pairs.tsv')
    assert pairs == [(f'S{n:02d}', f'S{n+1:02d}') for n in range(3, 36, 2)] and len(pairs) == 17
    assert sorted(s for p in pairs for s in p) == subjects                     # every subject in one pair only
    assert all(int(f[1:]) % 2 == 1 for f, _ in pairs)                           # fixed = odd (§3.1)


def test_the_feasibility_pairs_are_s01_s02_both_ways():
    assert rows('feasibility_pairs.tsv') == [('S01', 'S02'), ('S02', 'S01')]
    assert not {'S01', 'S02'} & set((SPLIT/'test_subjects.txt').read_text(encoding='utf-8').split())
