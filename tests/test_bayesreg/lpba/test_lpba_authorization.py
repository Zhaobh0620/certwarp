"""LP-C2 item 4 (contract C26.8-3, -4, C26.10 T15; D148): the LPBA40 resume (补跑) authorizations through holdout_common with the pair set 'lpba40_test' -- splits/lpba40_test_v1/
lpba40_test_v1_<tool>_<kind>_<k>.json, committed; written by "lead on BZ's behalf" with BZ's words of decision_log line 520 word for word, or by BZ; the cell a failed one; its own
lock not there.  One refusal per case on a temporary git repository; the hold-out's and the development runs' behaviour unchanged (their own tests)."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'holdout'))
import holdout_common as hc  # noqa: E402

from ..sg import _sg_world as W  # noqa: E402

PAIR = 'LPBA40_S05__LPBA40_S06'


@pytest.fixture
def tree(tmp_path, monkeypatch):
    repo = tmp_path/'code'
    repo.mkdir()
    W.git(repo, 'init', '-q')
    W.write_text(repo/'README.md', 'x\n')
    W.commit_all(repo, 'start')
    monkeypatch.setattr(hc, 'AUTH_ROOT', repo)
    lock = tmp_path/'results'/'locks'/f'{PAIR}.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('{"first": true}\n')
    previous = tmp_path/'results'/'xb'/PAIR/'result.json'
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps(dict(status='infra_failure')))
    return repo, lock, previous


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(lock, previous, **changes):
    doc = dict(schema='bayesreg.xb.resume.v1', written_by="lead on BZ's behalf", words=hc.LPBA40_BZ_WORDS, time_local='2026-11-19 3:00 PM CST', utc=hc.utc(),
               reason='infra_failure: the unit was OOM-killed (test)', tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test', lock_sha256=sha(lock),
               previous_record=str(previous), previous_record_sha256=sha(previous))
    doc.update(changes)
    return doc


def write(repo, name, doc, *, commit=True):
    path = W.write_json(repo/'splits'/'lpba40_test_v1'/name, doc)
    if commit:
        W.commit_all(repo, name)
    return path


def resume(path, lock, previous, *, complete=False):
    return hc.load_resume(path, tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test', lock=lock, previous_record=previous, complete=lambda: complete)


def test_t15_the_words_are_line_520s():
    line = (ROOT/'docs'/'decision_log.md').read_text(encoding='utf-8').splitlines()[519]
    assert '"预登记写好、复核过后批准你直接跑, 不用等我"' in line or '预登记写好、复核过后批准你直接跑, 不用等我"' in line
    assert hc.LPBA40_BZ_WORDS == '预登记写好、复核过后批准你直接跑, 不用等我'
    assert hc.AUTH_DIRS['lpba40_test'] == 'splits/lpba40_test_v1' and hc.AUTH_PREFIX['lpba40_test'] == 'lpba40_test_v1_'
    assert hc.AUTH_DIRS['holdout'] == 'splits/bayesreg_v1' and hc.AUTH_PREFIX['holdout'] == 'holdout_B_v1_'        # unchanged


def test_t15_a_sound_resume_is_accepted(tree):
    repo, lock, previous = tree
    path = write(repo, 'lpba40_test_v1_xb_resume_1.json', document(lock, previous))
    got = resume(path, lock, previous)
    assert got.document['pair_set'] == 'lpba40_test' and Path(got.lock_path).name == f'{PAIR}.lock.resume'
    bz = write(repo, 'lpba40_test_v1_xb_resume_2.json', document(lock, previous, written_by='BZ', words='补跑这一格'))
    assert resume(bz, lock, previous).document['written_by'] == 'BZ'                 # BZ himself: his own words
    valid, refused = hc.find_authorizations(kind='resume', tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test')
    assert len(valid) == 2 and refused == []


@pytest.mark.parametrize('case, match', [('writer', 'written_by'), ('words', 'word for word'), ('words_near', 'word for word'), ('name', 'is not splits/lpba40_test_v1'),
                                         ('holdout_place', 'is not splits/lpba40_test_v1'), ('not_failed', 'complete record'), ('lock_taken', 'resumed already'),
                                         ('uncommitted', 'not committed'), ('pair_set', 'pair_set')])
def test_t15_refused(tree, case, match):
    repo, lock, previous = tree
    name, doc, complete = 'lpba40_test_v1_xb_resume_1.json', document(lock, previous), False
    if case == 'writer':
        doc['written_by'] = 'lead'
    elif case == 'words':
        doc['words'] = '预登记写好、复核过后批准你直接跑'
    elif case == 'words_near':
        doc['words'] = '预登记写好、复核过后批准你直接跑，不用等我'          # a full-width comma: not the same words
    elif case == 'name':
        name = 'lpba40_test_v1_xb_resume_01x.json'
    elif case == 'not_failed':
        complete = True
    elif case == 'lock_taken':
        hc.resume_lock_path(lock).write_text('{}\n')
    elif case == 'pair_set':
        doc['pair_set'] = 'lpba40_feas'
    if case == 'holdout_place':
        path = W.write_json(repo/'splits'/'bayesreg_v1'/'holdout_B_v1_xb_resume_1.json', doc)
        W.commit_all(repo, 'wrong place')
    else:
        path = write(repo, name, doc, commit=case != 'uncommitted')
    with pytest.raises(hc.Refused, match=match):
        resume(path, lock, previous, complete=complete)


def test_t15_the_once_rule_with_a_resume(tree):
    repo, lock, previous = tree
    path = write(repo, 'lpba40_test_v1_xb_resume_1.json', document(lock, previous))
    got = resume(path, lock, previous)
    hc.check_once(lock, resume=got)                                                  # the first run's lock there, the resume's not
    Path(got.lock_path).write_text('{}\n')
    with pytest.raises(hc.Refused, match='resumed already'):
        hc.check_once(lock, resume=got)
    with pytest.raises(hc.Refused, match='runs once'):
        hc.check_once(lock)


def test_t15_the_holdout_words_rule_is_unchanged(tree):
    """The words rule is LPBA40's only: a hold-out authorization keeps the hold-out's rule (any non-empty words)."""
    repo, lock, previous = tree
    doc = document(lock, previous, pair_set='holdout', words='其他的话')
    path = W.write_json(repo/'splits'/'bayesreg_v1'/'holdout_B_v1_xb_resume_1.json', doc)
    W.commit_all(repo, 'hold-out')
    auth = hc.load_authorization(path, kind='resume', tool='xb', arm='ants_s', pair=PAIR, pair_set='holdout')
    assert auth.document['words'] == '其他的话'


# ---- the rejudgment records by the pair set (C26.8-4; LP-B's request): LPBA40's in runs/lpba40_test_v1/rejudgments, the hold-out's unchanged ----------------------------------
def rejudgment_record(repo, rel_dir, previous, pair_set, k=1):
    from bayesreg.io import rejudgment as R
    doc = dict(schema=R.REJUDGMENT_SCHEMA, arm='ants_s', pair=PAIR, pair_set=pair_set, status='needs_lead_rejudgment', record=dict(path=str(previous), sha256=sha(previous)),
               evidence=[], judgment='unit_limit', reason='RuntimeMaxSec without OOM evidence (test)', written_by='lead', words='', time_local='2026-11-19 3:00 PM CST', utc=hc.utc())
    path = W.write_json(repo/rel_dir/'ants_s'/f'{PAIR}_{k}.json', doc)
    W.commit_all(repo, f'rejudgment {pair_set}')
    return path


def test_the_rejudgment_directory_is_the_pair_sets(tree):
    from bayesreg.io import rejudgment as R
    repo, _, previous = tree
    assert R.rejudgment_dir('holdout') == R.REJUDGMENT_DIR_REL == 'runs/holdout_B_v1/rejudgments' and R.rejudgment_dir('dev') == R.REJUDGMENT_DIR_REL
    assert R.rejudgment_dir('lpba40_test') == 'runs/lpba40_test_v1/rejudgments' and R.rejudgment_dir('lpba40_feas') is None
    assert hc.REJUDGMENT_DIR_REL == R.REJUDGMENT_DIR_REL                       # holdout_common's constant unchanged
    path = rejudgment_record(repo, 'runs/lpba40_test_v1/rejudgments', previous, 'lpba40_test')
    found, problems = hc.rejudgment('ants_s', PAIR, record=previous, pair_set='lpba40_test')
    assert problems == [] and found['path'] == str(path) and found['document']['judgment'] == 'unit_limit'
    assert hc.rejudgment('ants_s', PAIR, record=previous) == (None, [])          # the hold-out's directory: nothing there
    found, problems = hc.rejudgment('ants_s', PAIR, record=previous, pair_set='lpba40_feas')
    assert found is None and 'no rejudgment records' in problems[0]
    rejudgment_record(repo, 'runs/holdout_B_v1/rejudgments', previous, 'holdout')
    found, problems = hc.rejudgment('ants_s', PAIR, record=previous)             # the hold-out: as before
    assert problems == [] and found['path'].endswith('runs/holdout_B_v1/rejudgments/ants_s/LPBA40_S05__LPBA40_S06_1.json')


# ---- review minor 8 (the lead's ruling): the words of line 520 only for a resume; a no_resume or a rerun carries BZ's words of that time -----------------------------------
def test_minor8_a_no_resume_carries_bzs_words_of_its_time(tree):
    repo, lock, previous = tree
    doc = dict(schema='bayesreg.xb.no_resume.v1', written_by="lead on BZ's behalf", words='这一格不补跑了', time_local='2026-11-19 4:00 PM CST', utc=hc.utc(),
               reason='BZ 2026-11-19 4:00 PM 原话见 decision_log 第 600 行（test）', tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test', record=str(previous),
               record_sha256=sha(previous), note=hc.NO_RESUME_NOTE)
    path = write(repo, 'lpba40_test_v1_xb_no_resume_1.json', doc)
    auth = hc.load_authorization(path, kind='no_resume', tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test')
    assert auth.document['words'] == '这一格不补跑了'
    resume = write(repo, 'lpba40_test_v1_xb_resume_1.json', document(lock, previous, words='这一格补跑'))
    with pytest.raises(hc.Refused, match='word for word'):                         # the resume keeps the rule
        hc.load_authorization(resume, kind='resume', tool='xb', arm='ants_s', pair=PAIR, pair_set='lpba40_test')
