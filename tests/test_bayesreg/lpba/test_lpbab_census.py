"""LP-B review minor 6 (contract C26.5-12 N47-N50, C26.10 T8 "准入在读之前（MANUAL_ROWS 钉次序）"): the readings of LP-B's LPBA40 branches the census patterns do not see, pinned by hand as
tests/test_bayesreg/sg/test_sg_census.py MANUAL_ROWS does (that file is not changed by this package; LP-A's tests/test_bayesreg/lpba/test_lpba_census.py is the precedent): the function is there and the
admission (or the check) comes before the read in its source; and the census rows of LP-B are registered."""
from .test_lpba_census import ROOT, first_call_line, function
from ..sg._census_inventory import INVENTORY

# (file, function, the calls in the order they must first appear; [] = the function is there, no order to pin)
ORDER = (
    ('experiments/he/he_holdout.py', 'lpba40_pre_manifest', ['check_pins', 'read_lists', 'load_lpba_allowlist', 'admit_pairs', 'check_lpba40_names']),
    ('experiments/he/he_holdout.py', 'lpba_admitted_images', ['lpba40_pre_manifest', '_read_manifest', 'load_pair']),
    ('experiments/hb/hb_configc.py', 'admit', ['admit_holdout_pair', 'lpba40_pre_manifest']),                                      # N49: the LPBA branch (its manifest after it: test_hb_lpba T8)
    ('experiments/hb/hb_core.py', 'lpba_hi', ['read_json', 'sha']),                                                                # N48: HI's records, then the files' sha256 (nothing decoded)
    ('experiments/hb/hb_store.py', 'lpba_input_lock', ['formal_code_problem', 'lpba_hi', 'sha']),                                  # step 7: rule one, HI's records, then the fields' sha256
    ('experiments/hb/hb_store.py', 'lpba_delete_fields', ['formal_code_problem', 'tracked_unchanged', 'sha', 'unlink']),          # the lock committed before any field goes
    ('experiments/xb/xb_core.py', 'method_t88_lpba', ['read_json', 'work_dir']),                                                   # N50 (N19's kind): the finalized record (a record), the cell's path
    ('experiments/xb/xb_drive.py', '_run', ['code_record', 'admit', 'method_run_check', 'check_pair_input']),                      # L33: the code, the admission, then the inputs
)


def test_lp_b_reads_after_its_admission():
    for rel, name, calls in ORDER:
        node = function(rel, name)
        lines = [first_call_line(node, c) for c in calls]
        assert lines == sorted(lines), (rel, name, list(zip(calls, lines)))


def test_the_census_rows_of_lp_b_are_registered():
    rows = {('experiments/he/he_holdout.py', 'lpba_admitted_images'): 'G', ('experiments/hb/hb_configc.py', 'admit'): 'G', ('experiments/hb/hb_configc.py', 'lpba_policy_of'): 'D',
            ('experiments/he/run_hx.py', 'lpba_policy_of'): 'D', ('experiments/lpba/lpba_arms.py', 'ArmSpy.see_member'): 'D', ('experiments/lpba/lpba_arms.py', 'detached_problem'): 'D'}
    assert {k: INVENTORY[k][0] for k in rows} == rows and 'LP-B' in INVENTORY[('experiments/hb/hb_configc.py', 'admit')][1]
    assert ROOT.is_dir()
