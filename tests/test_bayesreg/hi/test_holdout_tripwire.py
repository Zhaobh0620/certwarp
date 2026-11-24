"""The tripwire of the allowlist (review I5, the lead's "现在就把这项测试写进 tests 并在 main 上通过"): the subjects that are both on the repository's allowlist (configs/bayesreg/dev_subjects.json) and on any sealed list
(bayesreg.eval.truth: holdout_B_v1, H1, H3, H4, and H2 = LPBA40 S37-S40) are exactly the subjects of a committed unseal record that passes hi_core.load_unseal (the git checks and the decision log included);
without a record they are none.  On main there is no record, so the intersection must be empty; on the unseal branch (runs/20261107-hi/入口说明.md §4) it must be exactly the unsealed list.
Any other admission (one subject more, an H1 or H4 subject, an admission without a record) fails here.  Reads JSON and id lists only."""
from bayesreg.eval.truth import load_sealed_lists
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.unseal import load_unseal

from ._world import ROOT, hi


def test_the_allowlist_holds_a_sealed_subject_only_through_a_valid_unseal_record():
    """C23.9-13 (7): the record is read by bayesreg's one loader (bayesreg.io.unseal.load_unseal, C23.3-2); the assertion is the same."""
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    sealed = load_sealed_lists(ROOT)
    allowed = {s for ids in allowlist.subjects.values() for s in ids}
    on_sealed = {s for s in allowed if sealed.which(s) is not None}
    record = ROOT/hi.UNSEAL_REL
    if record.exists():
        unseal = load_unseal(record)
        assert on_sealed == set(unseal.subjects), sorted(on_sealed ^ set(unseal.subjects))[:10]
    else:
        assert on_sealed == set(), sorted(on_sealed)[:10]


def test_no_other_unseal_or_resume_file_lies_beside_the_pinned_one():
    names = sorted(p.name for p in (ROOT/'splits/bayesreg_v1').glob('unseal*'))
    assert all(n == 'unseal_holdout_B_v1.json' or hi.RESUME_NAME.fullmatch(n) for n in names), names
    if not (ROOT/hi.UNSEAL_REL).exists():
        assert names == [], names                # a resume record without the record it resumes is a mistake
