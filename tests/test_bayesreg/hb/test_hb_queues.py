"""HB items 3, 4 (the queues), 6, 7 (the input lock, the move): the method arm's queue (step 4), the review queue (step 5), the store tools.  The units are played by a fake system (what HE, the
review unit and the in-unit supervisor would leave; systemd-run's result line); the real system's start strings are checked, never launched (systemd-run is the sandbox's shim)."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .. import _p0_pins as P0
from . import _world as W

R, he, hc, hb, review, hm = W.R, W.he, W.hc, W.hb, W.hb_review, W.hb_method
QUIET = dict(max=0, oom=0, oom_kill=0)


@pytest.fixture
def unsealed(tmp_path, monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: None)
    monkeypatch.setattr(hm, 'PREREG_SHA256', hb.sha(hb.REPO/hm.PREREG_REL))      # the preregistration as it is now (revision 2 pins the adopted one)
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit=R.code_head(), clean=True))
    sim = W.XW.simulated_unseal(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    W.write_hi()                                                                 # A5b item 13: HI complete over its runs before the method queue starts a unit (hb_core.hi_union)
    W.XW.sealed_module._reset_cache()
    return sim


def pairs(n=2):
    return [hc.pair_name(*r) for r in W.XW.holdout_rows()[:n]]


class FakeSystem(hb.System):
    """Each unit plays one outcome: what its process and the supervisor would write."""

    def __init__(self, outcomes, *, dmesg='', journal='', gate_error=None):
        self.outcomes, self.kernel, self.journal_text, self.gate_error, self.runs, self.gates = list(outcomes), dmesg, journal, gate_error, [], []

    def run(self, argv, log_path):
        self.runs.append(argv)
        outcome = self.outcomes.pop(0)
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text(json.dumps(dict(utc='2026-11-09T00:00:00Z')))
        inner = argv[argv.index('--')+1:]
        script = Path(inner[2]).name
        pair = inner[inner.index('--pair')+1]
        if script == 'run_hx.py':
            self.method(pair, outcome)
        elif script == 'hb_review.py':
            self.review(pair, outcome)
        end = dict(utc='2026-11-09T00:01:00Z', exit_code=outcome.get('exit', 0), killed_by_signal=9 if outcome.get('exit') == -9 else None, killed_by_supervisor=False,
                   sigterm_received=outcome.get('result') == 'timeout', memory_events=outcome.get('events', QUIET), cgroup=None, dmesg_readable=outcome.get('dmesg_readable', True),
                   oom_lines=[] if outcome.get('dmesg_readable', True) else None, dxg_lines=0 if outcome.get('dmesg_readable', True) else None)
        if outcome.get('end', True):
            (unit_dir/'unit_end.json').write_text(json.dumps(end))
        Path(log_path).write_text(f"Finished with result: {outcome.get('result', 'success')}\n" if outcome.get('result', 'success') else 'no result line\n')
        return outcome.get('rc', 0)

    def method(self, pair, outcome):
        line = outcome.get('line')
        if line is None:
            return
        ctx = R.make_holdout_context(pair, check_identity=False)
        if line in he.OUTPUT_STATUSES:
            u, _ = W.write_cell(ctx, status=line, step4=False)          # the queue writes step4.json itself
            if outcome.get('certified') == 'differs':
                np.savez(Path(ctx['out'])/'repair'/'certified.npz', u=u+1e-9*(np.abs(u) > 0))
            if line != he.HX:
                np.savez(Path(ctx['out'])/'repair'/'certified.npz', u=u)        # a fallback pair whose repair also left a field: kept
        else:
            Path(ctx['out']).mkdir(parents=True)
            he.write_json(Path(ctx['out'])/'line.json', dict(line_status=line, rejection=dict(stage='admission') if line == he.REJECTED else None,
                                                            implementation_error=dict(kind='a') if line == he.IMPLEMENTATION else None))

    def review(self, pair, outcome):
        cell = hb.cell(pair)
        rev = cell/'review'
        rev.mkdir(exist_ok=True)
        mode = review.holdout_mode(None)
        if outcome.get('review') == 'finalize':                         # hb_review.finalized_part's fields (PQ, ConvexAdam-v4seg and the evaluation read t88_repair, output, ...)
            line = he.read_json(cell/'output'/'line.json')
            path = hb.write_once(rev/review.REVIEW_RECORD, dict(review.finalized_part(cell, line=line, review_doc=dict(consistent=True)), finalized=True))
            if not outcome.get('no_terminal'):
                hb.write_terminal(cell, pair=pair, status=line['line_status'], written_by='test', output_dir='output', review=dict(path='review/review.json', sha256=hb.sha(path)))
        elif outcome.get('review') == 'implementation':                 # C22.5-6 (a) in the (d) chain: hb_review.implementation_error's records (finalized false; the /home copies, item 10)
            path = hb.write_once(rev/review.REVIEW_RECORD, dict(line_status=he.IMPLEMENTATION, finalized=False, output=None, implementation_error=dict(kind='a', reason='test')))
            hb.write_terminal(cell, pair=pair, status=he.IMPLEMENTATION, written_by='test', output_dir=None, review=dict(path='review/review.json', sha256=hb.sha(path)))
            review.keep_on_home(mode, pair, cell)
        elif outcome.get('review') == 'stop':
            path = hb.write_once(rev/'review_not_finalized.json', dict(kind='rounds', reason='test', code=21, seq=review.next_stop_seq(pair)))     # as finish_not_finalized (n4)
            review.copy_to_home(mode, pair, path)

    def uptime(self):
        return 10.

    def dmesg(self):
        return self.kernel

    def journal(self, unit, since):
        return self.journal_text

    def gate(self, kind):
        self.gates.append(kind)
        if self.gate_error:
            raise self.gate_error
        return dict(passed=True)

    def disk(self, need_d, need_home):
        return dict(passed=True, free_gib={})

    def sleep(self, seconds):
        raise AssertionError('no hold file in the tests')


# ---- the start strings ------------------------------------------------------------------------------------------------------------------------------
def test_the_units_start_strings(tmp_path):
    pair = pairs(1)[0]
    m = hb.unit_argv('method', pair, ['experiments/he/run_hx.py', 'holdout-pair', '--pair', pair], code=hb.REPO, unit_dir=Path('/r/u'), res=dict(he.SUPERVISOR_UNIT), gpu=True, log=Path('/r/u.log'))
    r = hb.unit_argv('review', pair, ['experiments/hb/hb_review.py', 'unit', '--pair', pair], code=hb.REPO, unit_dir=Path('/r/u'), res=dict(he.EVALUATION_UNIT), gpu=False, log=Path('/r/u.log'))
    c = hb.unit_argv('configc', pair, ['experiments/hb/hb_configc.py', 'run', '--pair', pair], code=hb.REPO, unit_dir=Path('/r/u'), res=dict(hb.CONFIGC_UNIT), gpu=True, log=Path('/r/u.log'))
    assert '--unit=tag-hb-he-0423-0099' in m and '--unit=tag-hb-review-0423-0099' in r and '--unit=tag-hb-c-0423-0099' in c
    assert he.unit_argv_problems(m, he.SUPERVISOR_UNIT) == [] and he.unit_argv_problems(r, he.EVALUATION_UNIT) == [] and he.unit_argv_problems(c, hb.CONFIGC_UNIT) == []
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in r and '--setenv=CUDA_VISIBLE_DEVICES=' not in m and '--setenv=CUDA_VISIBLE_DEVICES=' not in c
    for argv in (m, r, c):
        i = argv.index('--')
        assert argv[i-5].endswith('experiments/xb/xb_unit.py') and argv[i-7:i-5] == ['-I', '-B'] and argv[i+1:i+3] == [str(hb.PY), '-B']
        assert f'--setenv=PYTHONPATH={he.core.FROZEN}:{hb.REPO}' in argv and not any(a.startswith('--unit=seq-vp') for a in argv) and '--property=OOMPolicy=continue' in argv
        assert argv[argv.index('--')-5].endswith(hb.XB_UNIT)
    assert '--nice=5' in m and '--nice=5' in r and not any(a.startswith('--nice') for a in c)
    for kind in hb.KINDS:
        q = hb.queue_launch(kind, ['queue'])
        assert f'--unit={hb.QUEUE_UNIT[kind]}' in q and hb.QUEUE_UNIT[kind].startswith('tag-hb-') and '--setenv=CUDA_VISIBLE_DEVICES=' in q
    assert hb.GATE_WAIT_SECONDS == dict(method=15*60, review=30*60, configc=15*60) and hb.GATE_MEM_GIB == he.FROZEN_RUN_GATE_GIB == 10.        # D126 (a): 30 minutes, in the code


# ---- step 4 -----------------------------------------------------------------------------------------------------------------------------------------
def test_the_sandbox_roots_hold_no_pytest_and_the_start_string_check_stays():
    """The roots lie in a tempfile.mkdtemp(prefix='xbt_') directory (the conftest, as XB's f027acc), not under pytest's basetemp, so the units' start strings pass the check under the standard
    command (no --basetemp); the check itself still refuses the word."""
    pair = pairs(1)[0]
    assert 'pytest' not in str(hc.RESULTS) and Path(hc.RESULTS).parent.name.startswith('xbt_') and 'pytest' not in str(hb.record_dir('method', pair))
    unit = hb.record_dir('method', pair)/'unit'
    argv = hb.unit_argv('method', pair, [hb.REPO/'experiments/he/run_hx.py', 'holdout-pair', '--pair', pair], code=hb.REPO, unit_dir=unit, res=dict(he.EVALUATION_UNIT), gpu=False,
                        log=unit/'unit.log')
    assert 'pytest' not in ' '.join(map(str, argv))
    with pytest.raises(ValueError, match='forbidden word'):
        hb.unit_argv('method', pair, [hb.REPO/'experiments/he/run_hx.py', 'holdout-pair', '--pair', pair], code=hb.REPO, unit_dir=Path(hc.RESULTS).parent/'pytest-of-x'/'unit',
                     res=dict(he.EVALUATION_UNIT), gpu=False, log=unit/'unit.log')


def test_run_unit_records_its_units_end_for_the_gate_and_the_sandbox_starts_without_one():
    """A5a review I1 and diff review n5: hb_core.run_unit records the end of the unit it ran (holdout_common.note_unit_end: the machine gate's local lower bound, tested in
    tests/test_bayesreg/holdout), so the next gate of this process waits 10 s after it; HB's sandbox starts every test without a recorded end (the next test checks that this one's does not leak)."""
    assert hc._local_end is None and hc.local_unit_end() is None
    pair = pairs(1)[0]
    unit = hb.record_dir('method', pair)/'unit'
    reading = hb.run_unit('method', pair, [hb.REPO/'experiments/he/run_hx.py', 'holdout-pair', '--pair', pair], FakeSystem([dict()]), batch=hc.batch_root(), unit_dir=unit,
                          res=dict(he.EVALUATION_UNIT), gpu=False, code=hb.REPO)
    end = hc.local_unit_end()
    assert reading['unit'] == hb.unit_name('method', pair) and end is not None and end['unit'] == reading['unit'] and type(end['monotonic']) is float


def test_the_sandbox_reset_the_previous_tests_unit_end():
    """A5a diff review n5: run after the test above in this file -- its recorded end does not reach this test."""
    assert hc._local_end is None and hc.local_unit_end() is None


def test_the_method_queue_without_an_unseal_record_stops_before_any_list_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='c', clean=True))
    monkeypatch.setattr(hm, 'PREREG_SHA256', hb.sha(hb.REPO/hm.PREREG_REL))
    read = []
    monkeypatch.setattr(hc, 'holdout_pairs', lambda repo=None: read.append(1) or [])
    system = FakeSystem([])
    assert hm.queue(system=system, echo=lambda m: None) == hb.EXIT['refused']
    assert read == [] and system.runs == [] and not Path(hc.batch_root()).exists()
    record, problems = hm.start_checks()
    assert any('unseal' in p for p in problems)


def test_the_method_queue_without_a_pinned_preregistration_refuses(tmp_path, monkeypatch, unsealed):
    monkeypatch.setattr(hm, 'PREREG_SHA256', None)
    assert hm.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused']


def test_the_method_queues_preregistration_pin_is_the_committed_file():
    """P0: hb_method.PREREG_SHA256 is the sha256 of the committed preregistration; the queue's check (prereg_problem) finds no problem on this tree.  None is refused in
    the test above; SETTINGS_SHA256 is checked in test_hb_he_holdout.test_the_methods_settings_and_their_pin."""
    P0.assert_pinned(hm.PREREG_SHA256, hm.PREREG_REL)
    assert hm.prereg_problem(P0.REPO) is None


def test_the_method_queue_runs_the_pairs_and_deletes_a_matching_certified_field(tmp_path, monkeypatch, unsealed):
    p1, p2 = pairs(2)
    system = FakeSystem([dict(line=he.HX), dict(line=he.FALLBACK_C)])
    assert hm.queue(pairs=[p1, p2], system=system, echo=lambda m: None) == 0
    assert system.gates == ['method', 'method'] and len(system.runs) == 2
    s1, s2 = (he.read_json(hb.record_dir('method', p)/'step4.json') for p in (p1, p2))
    assert s1['status'] == he.HX and s1['went_on'] and s1['certified']['deleted'] is True and s1['certified']['same'] is True and not (hb.cell(p1)/'repair'/'certified.npz').exists()
    assert s2['status'] == he.FALLBACK_C and s2['certified']['applies'] is False and (hb.cell(p2)/'repair'/'certified.npz').is_file()
    assert hb.lock_path('method', p1).is_file() and s1['unit_reading']['status'] is None and s1['lock']['sha256']
    assert hm.state(p1) == 'done' and hm.queue(pairs=[p1, p2], system=FakeSystem([]), echo=lambda m: None) == 0        # done pairs are skipped


def test_a_certified_field_that_is_not_the_output_stops_and_both_are_kept(tmp_path, monkeypatch, unsealed):
    p1 = pairs(1)[0]
    assert hm.queue(pairs=[p1], system=FakeSystem([dict(line=he.HX, certified='differs')]), echo=lambda m: None) == hb.EXIT['stop']
    step = he.read_json(hb.record_dir('method', p1)/'step4.json')
    assert step['went_on'] is False and step['certified']['passed'] is False and (hb.cell(p1)/'repair'/'certified.npz').is_file() and (hb.cell(p1)/'output'/'final.npz').is_file()


@pytest.mark.parametrize('line', [he.REJECTED, he.IMPLEMENTATION])
def test_no_output_from_he_stops_the_batch_with_the_cells_terminal_record(tmp_path, monkeypatch, unsealed, line):
    p1, p2 = pairs(2)
    system = FakeSystem([dict(line=line, exit=9 if line == he.IMPLEMENTATION else 0)])
    assert hm.queue(pairs=[p1, p2], system=system, echo=lambda m: None) == hb.EXIT['implementation'] and len(system.runs) == 1
    term = he.read_json(hb.terminal_path(hb.cell(p1)))
    assert term['line_status'] == line and term['output_dir'] is None and term['line_json']['sha256'] == hb.sha(hb.cell(p1)/'line.json')
    # the next start stops at this pair until BZ decides (P20: a rerun by a resume record; C22.5-6: (甲) a no_resume record, (乙) a resume record)
    assert hm.state(p1) == 'stopped' and hm.queue(pairs=[p1, p2], system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['resume']
    step = hb.record_dir('method', p1)/'step4.json'
    if line == he.REJECTED:
        assert hm.state(p1) == 'stopped'
        # A5b review I2: BZ's no_resume of a P20 cell names P20 in its reason (bayesreg.io.rejudgment.p20_reason_problem, the evaluation entry's criterion); one that does not is refused,
        # not taken by state(), and stays (files are only added)
        vague = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ: 不批重跑')
        with pytest.raises(hc.Refused, match='P20'):
            hm.no_resume(p1, str(vague))
        assert hm.bz_declined(p1) is None and hm.state(p1) == 'stopped'
        rel = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ: 不批重跑 (P20)')
        assert hm.bz_declined(p1).path.endswith(Path(rel).name)
        assert hm.no_resume(p1, str(rel)) == hb.terminal_path(hb.cell(p1)) and he.read_json(hb.terminal_path(hb.cell(p1))) == term      # the lead's item 13: BZ's no rerun of a P20 cell
        assert hm.state(p1) == 'final'                                       # written (no output); the batch goes past it
        system = FakeSystem([dict(line=he.HX)])
        assert hm.queue(pairs=[p1, p2], system=system, echo=lambda m: None) == 0 and len(system.runs) == 1
    else:
        # runs/20261109-align item 3: the record's reason names the case of C22.5-6 and (甲); one that does not is refused and not taken
        vague = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ: 补成恒等')
        with pytest.raises(hc.Refused, match='C22.5-6'):
            hm.no_resume(p1, str(vague))
        assert hm.state(p1) == 'stopped'
        resume = commit_hb_authorization('resume', p1, arm='method', lock_sha256=hb.sha(hb.lock_path('method', p1)), previous_record=str(step), previous_record_sha256=hb.sha(step),
                                         reason='BZ: fixed, rerun')                                     # (乙) without the case named: refused
        assert hm.queue(pairs=[p1, p2], resumes={p1: str(resume)}, system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused']
        rel = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a): 补成恒等，其余续跑')
        assert (Path(hc.AUTH_ROOT)/vague).is_file() and hm.bz_declined(p1).path.endswith(Path(rel).name)       # the vague record stays (files are only added) and is passed over
        assert hm.no_resume(p1, str(rel)) == hb.terminal_path(hb.cell(p1)) and he.read_json(hb.terminal_path(hb.cell(p1))) == term     # (甲): the terminal record unchanged, final
        assert hm.state(p1) == 'final'
        system = FakeSystem([dict(line=he.HX)])
        assert hm.queue(pairs=[p1, p2], system=system, echo=lambda m: None) == 0 and len(system.runs) == 1 and hm.state(p2) == 'done'


@pytest.mark.parametrize('outcome, described', [(dict(events=dict(oom=1, oom_kill=1), exit=-9, result='oom-kill'), 'infra_failure'),
                                                (dict(result='timeout', exit=143), 'unit_limit'),
                                                (dict(end=False, result=None), 'needs_lead_rejudgment'),
                                                (dict(exit=8), None)])
def test_a_supervisor_that_left_no_line_json_is_infra_failure_whatever_the_unit_reads(tmp_path, monkeypatch, unsealed, outcome, described):
    """Prereg revision 1c item 1, runs/20261109-align item 4: HE's line.json decides; without one the pair is infra_failure, whatever holdout_common.classify_unit_end reads (recorded as a
    description); the batch stops; then the pair needs a resume record (prereg §5.4)."""
    p1 = pairs(1)[0]
    assert hm.queue(pairs=[p1], system=FakeSystem([outcome]), echo=lambda m: None) == hb.EXIT['infra']
    step = he.read_json(hb.record_dir('method', p1)/'step4.json')
    assert step['status'] == 'infra_failure' and step['unit_reading_described']['status'] == described and step['line_json'] is None and step['went_on'] is False
    assert not hb.terminal_path(hb.cell(p1)).exists() and hm.state(p1) == 'stopped' and hm.queue(pairs=[p1], system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['resume']


def test_a_resume_without_a_line_json_again_is_written_as_no_output_infrastructure(tmp_path, monkeypatch, unsealed):
    """Prereg §5.4: one resume; the resumed attempt failing again -> the cell's terminal infra_failure with the resume record (no third run)."""
    p1 = pairs(1)[0]
    killed = dict(events=dict(oom=1, oom_kill=1), exit=-9, result='oom-kill')
    hm.queue(pairs=[p1], system=FakeSystem([killed]), echo=lambda m: None)
    step = hb.record_dir('method', p1)/'step4.json'
    rel = commit_hb_authorization('resume', p1, arm='method', lock_sha256=hb.sha(hb.lock_path('method', p1)), previous_record=str(step), previous_record_sha256=hb.sha(step))
    assert hm.queue(pairs=[p1], resumes={p1: str(rel)}, system=FakeSystem([killed]), echo=lambda m: None) == hb.EXIT['infra']
    term = he.read_json(hb.terminal_path(hb.cell(p1)))
    assert term['line_status'] == 'infra_failure' and term['rerun_record']['path'] == str(rel) and hm.state(p1) == 'final'
    assert hm.queue(pairs=[p1], system=FakeSystem([]), echo=lambda m: None) == 0                         # final: skipped, no third run


def commit_hb_authorization(kind, pair, reason='test', **fields):
    repo = W.XW.auth_repo()
    directory = hc.authorization_dir('holdout')
    k = 1
    while (directory/f'holdout_B_v1_hb_{kind}_{k}.json').exists():
        k += 1
    doc = dict(schema=f'bayesreg.hb.{kind}.v1', written_by="lead on BZ's behalf", words='（测试）BZ：可以', time_local='2026-11-09 7:00 PM CST', utc='2026-11-09T13:00:00Z', reason=reason, tool='hb',
               pair=pair, pair_set='holdout', **fields)
    path = W.XW.write_json(directory/f'holdout_B_v1_hb_{kind}_{k}.json', doc)
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', f'{kind} {k}')
    return path.relative_to(hc.AUTH_ROOT)


def test_a_resume_renames_the_first_attempt_and_runs_once_more(tmp_path, monkeypatch, unsealed):
    p1 = pairs(1)[0]
    assert hm.queue(pairs=[p1], system=FakeSystem([dict(events=dict(oom=1, oom_kill=1), exit=-9, result='oom-kill')]), echo=lambda m: None) == hb.EXIT['infra']
    hb.cell(p1).mkdir(parents=True)
    (hb.cell(p1)/'half.txt').write_text('what the killed attempt left')
    step = hb.record_dir('method', p1)/'step4.json'
    rel = commit_hb_authorization('resume', p1, arm='method', lock_sha256=hb.sha(hb.lock_path('method', p1)), previous_record=str(step), previous_record_sha256=hb.sha(step))
    assert hm.queue(pairs=[p1], resumes={p1: str(rel)}, system=FakeSystem([dict(line=he.HX)]), echo=lambda m: None) == 0
    resumed = he.read_json(hb.record_dir('method', p1)/'step4_resume.json')
    assert resumed['went_on'] and resumed['renamed']['same'] is True and (hb.cell(p1).with_name(f'{p1}.attempt0')/'half.txt').is_file() and hc.resume_lock_path(hb.lock_path('method', p1)).is_file()
    assert hm.state(p1) == 'done'


def test_no_resume_writes_the_terminal_record_of_an_infrastructure_failure(tmp_path, monkeypatch, unsealed):
    p1 = pairs(1)[0]
    hm.queue(pairs=[p1], system=FakeSystem([dict(events=dict(oom=1, oom_kill=1), exit=-9, result='oom-kill')]), echo=lambda m: None)
    step = hb.record_dir('method', p1)/'step4.json'
    with pytest.raises(hc.Refused):
        hm.no_resume(p1, 'splits/bayesreg_v1/holdout_B_v1_hb_no_resume_1.json')
    rel = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE)
    hm.no_resume(p1, str(rel))
    term = he.read_json(hb.terminal_path(hb.cell(p1)))
    assert term['line_status'] == 'infra_failure' and term['rerun_record']['path'] == str(rel) and hm.state(p1) == 'final'


def test_every_hb_start_takes_p1_from_the_pinned_record_and_the_locks_must_name_it(tmp_path, monkeypatch, unsealed):
    """C22.H0-4, decision_log 2026-11-10 ~2:59 AM CST (1): P1 comes from runs/holdout_B_v1/pinned_code.json (hb_core.code_record -> holdout_common.pinned_code_problems; stubbed here: P1 = the
    code tree's HEAD), --commit only a check; the locks' commits must all be P1: another one, or two, refused."""
    Path(hc.LOCK_ROOT).mkdir(parents=True, exist_ok=True)
    p1 = R.code_head()
    seen = []

    def code_record(commit=None, run=None):
        seen.append(commit)
        if commit is not None and commit != p1:
            raise hb.Stop(hb.EXIT['refused'], f'the given commit {commit} is not the pinned record\'s {p1}')
        return dict(commit=p1, clean=True)
    monkeypatch.setattr(hb, 'code_record', code_record)
    (Path(hc.LOCK_ROOT)/'hb__method__p1.lock').write_text(json.dumps(dict(code=p1))+'\n')
    record, problems = hm.start_checks()
    assert problems == [] and seen[-1] is None and record['existing_pins'] == [p1] and record['code']['commit'] == p1
    record, problems = hm.start_checks('f'*40)
    assert any('not the pinned record' in p for p in problems)
    (Path(hc.LOCK_ROOT)/'hb__configc__p2.lock').write_text(json.dumps(dict(code='e'*40))+'\n')
    record, problems = hm.start_checks()
    assert any('more than one pinned commit' in p for p in problems)


# ---- step 5 -----------------------------------------------------------------------------------------------------------------------------------------
def method_done(ps, status=he.HX):
    for p in ps:
        W.write_cell(R.make_holdout_context(p, check_identity=False), status=status)


def test_the_review_queue_writes_the_manifest_and_the_lock_then_reviews_each_pair_then_the_finalized_record(tmp_path, monkeypatch, unsealed):
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    system = FakeSystem([dict(review='finalize'), dict(review='finalize')])
    assert review.queue(system=system, echo=lambda m: None) == 0
    root = hb.review_root()
    manifest = he.read_json(root/'input_manifest.json')
    assert sorted(manifest['pairs']) == sorted(ps) and all(r['files']['output/final.npz'] for r in manifest['pairs'].values()) and (root/'review.lock').is_file()
    lock = json.loads((root/'review.lock').read_text().splitlines()[0])
    assert lock['manifest_sha256'] == hb.sha(root/'input_manifest.json') and system.gates == ['review', 'review']
    final = he.read_json(root/'finalized_outputs.json')
    assert sorted(final['pairs']) == sorted(ps) and all(r['terminal']['sha256'] == hb.sha(hb.terminal_path(hb.cell(p))) for p, r in final['pairs'].items())
    inner = system.runs[0][system.runs[0].index('--')+1:]
    assert inner[2].endswith('hb_review.py') and inner[3:5] == ['unit', '--pair'] and '--setenv=CUDA_VISIBLE_DEVICES=' in system.runs[0]
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused']                                 # the batch lock is there: once


def test_a_step5_implementation_error_stops_until_bz_decides_and_a_restart_goes_on(tmp_path, monkeypatch, unsealed):
    """runs/20261109-align items 3: an hx_implementation_error finalized at step 5 stops the batch; a later start (the batch lock there, the manifest not written again) stops again until BZ's (甲) is
    committed -- a no_resume record of arm review whose record is the pair's review.json and whose reason names C22.5-6 (a) and (甲); then the remaining pairs run and the finalized record is written."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    system = FakeSystem([dict(review='implementation')])
    assert review.queue(system=system, echo=lambda m: None) == hb.EXIT['implementation'] and len(system.runs) == 1
    manifest = hb.sha(hb.review_root()/'input_manifest.json')
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['implementation']
    rec = review.home_records(ps[0])/review.REVIEW_RECORD             # item 10: the /home copy of the cell's review.json
    assert hb.sha(rec) == hb.sha(hb.cell(ps[0])/'review'/review.REVIEW_RECORD)
    cell_rec = hb.cell(ps[0])/'review'/review.REVIEW_RECORD
    commit_hb_authorization('no_resume', ps[0], arm='review', record=str(cell_rec), record_sha256=hb.sha(cell_rec), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a)')   # the cell's: not taken
    assert review.review_decision(ps[0]) is None
    commit_hb_authorization('no_resume', ps[0], arm='review', record=str(rec), record_sha256=hb.sha(rec), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b)')     # the wrong case
    assert review.review_decision(ps[0]) is None and review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['implementation']
    rel = commit_hb_authorization('no_resume', ps[0], arm='review', record=str(rec), record_sha256=hb.sha(rec), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a): 记为失败，其余续跑')
    assert review.review_decision(ps[0]).path.endswith(Path(rel).name)
    system = FakeSystem([dict(review='finalize')])
    assert review.queue(system=system, echo=lambda m: None) == 0 and len(system.runs) == 1 and hb.sha(hb.review_root()/'input_manifest.json') == manifest
    final = he.read_json(hb.review_root()/'finalized_outputs.json')
    assert final['pairs'][ps[0]]['line_status'] == he.IMPLEMENTATION and final['pairs'][ps[0]]['bz_decision']['path'].endswith(Path(rel).name) and final['pairs'][ps[0]]['bz_decision']['step'] == 5
    assert final['pairs'][ps[1]]['line_status'] == he.HX and final['pairs'][ps[1]]['bz_decision'] is None
    import hb_store                                     # item 10: after the cell moved to D:, the /home copies are there with the finalized sha256 and BZ's (甲) still holds
    hb_store.move_pair(ps[0], disk=lambda: None)
    assert not hb.cell(ps[0]).exists() and hb.sha(rec) == final['pairs'][ps[0]]['review']['sha256'] and review.review_decision(ps[0]).path.endswith(Path(rel).name)
    assert hb.sha(review.home_records(ps[0])/'terminal.json') == final['pairs'][ps[0]]['terminal']['sha256'] and he.read_json(review.home_records(ps[0])/'records.json')['files']['review.json']['same']


@pytest.mark.parametrize('case', ['rejected', 'implementation', 'implementation_decided'])
def test_the_review_queue_starts_only_when_every_pair_of_step_4_is_final_or_done(tmp_path, monkeypatch, unsealed, case):
    """HB review I1: D110's gate is hb_method.state: a step-4 hx_provenance_rejected (P20) or an hx_implementation_error without BZ's (甲) stops step 5 before anything is written; with (甲) the pair
    is recorded and the finalized record names the decision (bz_decision, step 4)."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    line = he.REJECTED if case == 'rejected' else he.IMPLEMENTATION
    assert hm.queue(pairs=[ps[0]], system=FakeSystem([dict(line=line, exit=9)]), echo=lambda m: None) == hb.EXIT['implementation']
    method_done(ps[1:])
    if case == 'implementation_decided':
        step = hb.record_dir('method', ps[0])/'step4.json'
        rel = commit_hb_authorization('no_resume', ps[0], arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a)')
        assert hm.state(ps[0]) == 'final'
        assert review.queue(system=FakeSystem([dict(review='finalize')]), echo=lambda m: None) == 0
        final = he.read_json(hb.review_root()/'finalized_outputs.json')
        assert final['pairs'][ps[0]]['line_status'] == he.IMPLEMENTATION and final['pairs'][ps[0]]['bz_decision'] == dict(step=4, path=str(Path(hc.AUTH_ROOT)/rel), sha256=hb.sha(Path(hc.AUTH_ROOT)/rel))
        return
    assert hm.state(ps[0]) == 'stopped'
    system = FakeSystem([])
    assert review.queue(system=system, echo=lambda m: None) == hb.EXIT['refused'] and system.runs == []
    assert not (hb.review_root()/'input_manifest.json').exists() and not (hb.review_root()/'review.lock').exists()


def test_a_reason_naming_both_decisions_or_both_cases_is_not_taken(tmp_path, monkeypatch, unsealed):
    """HB review m9 (ii): the reason names exactly one decision and one case."""
    for reason, ok in (('BZ (甲) on C22.5-6 (a)', True), ('BZ (甲) or (乙) on C22.5-6 (a)', False), ('BZ (甲) on C22.5-6 (a) and C22.5-6 (b)', False), ('BZ (乙) on C22.5-6 (a)', False)):
        assert (hb.decision_reason_problem(dict(reason=reason), 'a', '(甲)') is None) is ok, reason


def test_a_resumed_attempt_in_hx_implementation_error_takes_bz_s_decision_on_its_own_record(tmp_path, monkeypatch, unsealed):
    """HB review m9 (i): (乙) reran the pair, which ended in hx_implementation_error again; BZ's (甲) binds step4_resume.json (the attempt it is about), not the first attempt's step4.json."""
    p1 = pairs(1)[0]
    hm.queue(pairs=[p1], system=FakeSystem([dict(line=he.IMPLEMENTATION, exit=9)]), echo=lambda m: None)
    step = hb.record_dir('method', p1)/'step4.json'
    resume = commit_hb_authorization('resume', p1, arm='method', lock_sha256=hb.sha(hb.lock_path('method', p1)), previous_record=str(step), previous_record_sha256=hb.sha(step),
                                     reason='BZ (乙) on C22.5-6 (a): fixed, rerun')
    assert hm.queue(pairs=[p1], resumes={p1: str(resume)}, system=FakeSystem([dict(line=he.IMPLEMENTATION, exit=9)]), echo=lambda m: None) == hb.EXIT['implementation']
    again = hb.record_dir('method', p1)/'step4_resume.json'
    assert hm.step_record(p1) == again and hm.state(p1) == 'stopped'
    commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a)')       # the first attempt's: not taken
    assert hm.state(p1) == 'stopped'
    rel = commit_hb_authorization('no_resume', p1, arm='method', record=str(again), record_sha256=hb.sha(again), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a)')
    assert hm.bz_declined(p1).path.endswith(Path(rel).name) and hm.state(p1) == 'final'


def test_the_deletion_of_certified_npz_is_recorded_before_the_file_goes(tmp_path, monkeypatch, unsealed):
    """HB review m8: certified_deleted.json (its sha256, the bitwise check) is written before repair/certified.npz is deleted."""
    p1 = pairs(1)[0]
    seen = []
    real = Path.unlink

    def unlink(self, *a, **k):
        if self.name == 'certified.npz':
            seen.append((hb.record_dir('method', p1)/'certified_deleted.json').is_file())
        return real(self, *a, **k)
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert hm.queue(pairs=[p1], system=FakeSystem([dict(line=he.HX)]), echo=lambda m: None) == 0
    doc = he.read_json(hb.record_dir('method', p1)/'certified_deleted.json')
    assert seen == [True] and doc['passed'] is True and doc['certified_sha256'] and not (hb.cell(p1)/'repair'/'certified.npz').exists()


def test_review_json_without_terminal_json_stops_the_queue(tmp_path, monkeypatch, unsealed):
    """HB review m7 (i): a unit that wrote review.json but not terminal.json (killed between them) does not count as finalized; the queue stops (D123), and a later start stops too."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    assert review.queue(system=FakeSystem([dict(review='finalize', no_terminal=True, exit=-9, result='timeout')]), echo=lambda m: None) == hb.EXIT['infra']
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['rejudge'] and not (hb.review_root()/'finalized_outputs.json').exists()


def test_an_exception_of_a_queue_still_writes_its_batch_record(tmp_path, monkeypatch, unsealed):
    """HB review m11: an exception other than Stop and Refused still ends in batch.json (the stop, the traceback)."""
    p1 = pairs(1)[0]
    monkeypatch.setattr(hm, 'run_one', lambda *a, **k: (_ for _ in ()).throw(FileExistsError('test: a record there already')))
    assert hm.queue(pairs=[p1], system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['stop']
    [batch] = [b for b in Path(hc.batch_root()).glob('hb_method_*') if (b/'batch.json').is_file()]
    doc = he.read_json(batch/'batch.json')
    assert doc['stop']['code'] == hb.EXIT['stop'] and 'FileExistsError' in doc['stop']['reason'] and doc['stop']['traceback']


def test_the_review_queue_needs_every_pairs_step4_state(tmp_path, monkeypatch, unsealed):
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps[:1])
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused'] and not (hb.review_root()/'input_manifest.json').exists()


@pytest.mark.parametrize('outcome, code', [(dict(review='stop', exit=21), 21), (dict(events=dict(oom=1, oom_kill=1), exit=-9, result='oom-kill'), hb.EXIT['infra']),
                                           (dict(end=False, result=None), hb.EXIT['rejudge'])])
def test_a_review_unit_that_did_not_finalize_stops_the_queue(tmp_path, monkeypatch, unsealed, outcome, code):
    """C22.H1-4, D123 (b): the unit's own STOP record, or killed without one (OOM evidence, unreadable): not a method failure; the pair stays unfinalized; nothing more runs."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    system = FakeSystem([outcome])
    assert review.queue(system=system, echo=lambda m: None) == code and len(system.runs) == 1
    assert not (hb.review_root()/'finalized_outputs.json').exists() and not hb.terminal_path(hb.cell(ps[0])).exists()


def test_a_closed_gate_stops_the_review_queue_before_the_unit_and_leaves_the_output(tmp_path, monkeypatch, unsealed):
    ps = pairs(1)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    before = {p.name: hb.sha(p) for p in (hb.cell(ps[0])/'output').iterdir()}
    system = FakeSystem([], gate_error=hc.InfrastructureFailure('precheck_timeout', 'MemAvailable < 10 GiB for 30 minutes'))
    assert review.queue(system=system, echo=lambda m: None) == hb.EXIT['infra'] and system.runs == []
    assert {p.name: hb.sha(p) for p in (hb.cell(ps[0])/'output').iterdir()} == before and not (hb.cell(ps[0])/'output_superseded_0').exists()


# ---- the move and the input lock ---------------------------------------------------------------------------------------------------------------
def finalized_world(ps, monkeypatch):
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    review.queue(system=FakeSystem([dict(review='finalize') for _ in ps]), echo=lambda m: None)


def test_the_move_after_step_5_only_and_once(tmp_path, monkeypatch, unsealed):
    import hb_store
    ps = pairs(1)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    with pytest.raises(hc.Refused, match='finalized_outputs'):
        hb_store.move_pair(ps[0], disk=lambda: None)
    review.queue(system=FakeSystem([dict(review='finalize')]), echo=lambda m: None)
    files = {str(p.relative_to(hb.cell(ps[0]))): hb.sha(p) for p in hb.cell(ps[0]).rglob('*') if p.is_file()}
    out = hb_store.move_pair(ps[0], disk=lambda: None)
    store = Path(hc.big_dir('method', ps[0]))
    assert out['passed'] and {r['file']: r['sha256_after'] for r in out['files']} == files and (store/'output'/'final.npz').is_file() and not hb.cell(ps[0]).exists()
    assert hb_store.where(ps[0]) == store and (hb.record_dir('method', ps[0])/'move.json').is_file()
    import xb_core
    import xb_drive
    run = store/'t88'/'run.json'                        # HB review m13: ConvexAdam-v4seg knows the file by the finalized record's sha256 and the move record's dst
    assert xb_core.method_t88_run('holdout', ps[0]) == run and run.is_file() and xb_core.method_t88_sha256('holdout', ps[0]) == hb.sha(run)
    assert xb_drive.method_run_check('convexadam_v4seg', 'holdout', ps[0], run)['sha256'] == hb.sha(run)
    run.write_text(run.read_text().replace('done', 'Done'))
    with pytest.raises(hc.Refused, match='finalized record'):
        xb_drive.method_run_check('convexadam_v4seg', 'holdout', ps[0], run)
    with pytest.raises(hc.Refused, match='once'):
        hb_store.move_pair(ps[0], disk=lambda: None)


def test_a_destination_with_a_file_moves_nothing(tmp_path, monkeypatch, unsealed):
    """HB review I3: the D: store of the pair holds a file already: refused before anything moves, the cell whole on /home, no move record."""
    import hb_store
    ps = pairs(1)
    finalized_world(ps, monkeypatch)
    store = Path(hc.big_dir('method', ps[0]))
    store.mkdir(parents=True)
    (store/'terminal.json').write_text('left over')
    before = {str(q.relative_to(hb.cell(ps[0]))): hb.sha(q) for q in hb.cell(ps[0]).rglob('*') if q.is_file()}
    with pytest.raises(hc.Refused, match='already at'):
        hb_store.move_pair(ps[0], disk=lambda: None)
    assert {str(q.relative_to(hb.cell(ps[0]))): hb.sha(q) for q in hb.cell(ps[0]).rglob('*') if q.is_file()} == before and not (hb.record_dir('method', ps[0])/'move.json').exists()


def test_a_half_move_stops_where_and_the_move_until_move_resume_repairs_it(tmp_path, monkeypatch, unsealed):
    """HB review I3: an OSError half way (D: full): move.json passed false; `move` stops (exit 22) instead of skipping the pair, where() refuses; move-resume with a committed resume record (arm
    move) moves the rest on and checks the moved files; then where() names D: and a second move-resume is refused."""
    import hb_store
    import shutil
    ps = pairs(1)
    finalized_world(ps, monkeypatch)
    monkeypatch.setattr(hm, 'start_checks', lambda commit=None, code_run=None: (dict(), []))          # hb_store.main's start checks (the code identity is tested on its own)
    real, calls = shutil.copyfile, []

    def copyfile(src, dst, *a, **k):
        calls.append(src)
        if len(calls) == 3:
            raise OSError(28, 'No space left on device (test)')
        return real(src, dst, *a, **k)
    monkeypatch.setattr(shutil, 'copyfile', copyfile)
    with pytest.raises(hc.InfrastructureFailure):
        hb_store.move_pair(ps[0], disk=lambda: None)
    monkeypatch.setattr(shutil, 'copyfile', real)
    first = he.read_json(hb.record_dir('method', ps[0])/'move.json')
    assert first['passed'] is False and first['error'] and hb.cell(ps[0]).exists()
    with pytest.raises(hc.Refused, match='half moved'):
        hb_store.where(ps[0])
    assert hb_store.main(['move']) == hb.EXIT['refused']
    move = hb.record_dir('method', ps[0])/'move.json'
    with pytest.raises(hc.Refused):
        hb_store.move_resume(ps[0], 'splits/bayesreg_v1/holdout_B_v1_hb_resume_1.json', disk=lambda: None)          # not committed
    rel = commit_hb_authorization('resume', ps[0], arm='move', lock_sha256=hb.sha(move), previous_record=str(move), previous_record_sha256=hb.sha(move))
    out = hb_store.move_resume(ps[0], str(rel), disk=lambda: None)
    store = Path(hc.big_dir('method', ps[0]))
    assert out['passed'] and hb_store.where(ps[0]) == store and not hb.cell(ps[0]).exists() and (store/'terminal.json').is_file() and (store/'output'/'final.npz').is_file()
    with pytest.raises(hc.Refused):
        hb_store.move_resume(ps[0], str(rel), disk=lambda: None)
    assert hb_store.main(['move']) == 0


@pytest.mark.parametrize('case', ['half', 'equal'])
def test_move_resume_keeps_a_half_copy_and_copies_the_file_again(tmp_path, monkeypatch, unsealed, case):
    """HB diff review n2: D: full while a file is copied leaves half of it at the destination (move.json: that row ok false, problem copying).  move-resume checks the half copy's sha256 is not the
    source's, renames it whole to <file>.partial_1 (kept; move_resume_partials.json: the sha256 before and after the rename), copies the file again; a destination file equal to the source is refused,
    nothing renamed or moved."""
    import hb_store
    import shutil
    ps = pairs(1)
    finalized_world(ps, monkeypatch)
    monkeypatch.setattr(hm, 'start_checks', lambda commit=None, code_run=None: (dict(), []))
    real, calls = shutil.copyfile, []

    def copyfile(src, dst, *a, **k):
        calls.append(src)
        if len(calls) == 3:                             # the third file: half of it written, then the disk is full
            data = Path(src).read_bytes()
            with open(dst, 'wb') as stream:
                stream.write(data[:max(1, len(data)//2)])
            raise OSError(28, 'No space left on device (test)')
        return real(src, dst, *a, **k)
    monkeypatch.setattr(shutil, 'copyfile', copyfile)
    with pytest.raises(hc.InfrastructureFailure):
        hb_store.move_pair(ps[0], disk=lambda: None)
    monkeypatch.setattr(shutil, 'copyfile', real)
    move = hb.record_dir('method', ps[0])/'move.json'
    first = he.read_json(move)
    row = next(r for r in first['files'] if r['problem'] == 'copying')
    half, source = Path(row['dst']), Path(row['src'])
    assert half.is_file() and source.is_file() and hb.sha(half) != hb.sha(source) and half.stat().st_size < source.stat().st_size      # a real half copy at the destination
    half_sha = hb.sha(half)
    if case == 'equal':
        shutil.copyfile(source, half)                   # the copy had ended: not a half copy
    rel = commit_hb_authorization('resume', ps[0], arm='move', lock_sha256=hb.sha(move), previous_record=str(move), previous_record_sha256=hb.sha(move))
    store = Path(hc.big_dir('method', ps[0]))
    if case == 'equal':
        with pytest.raises(hc.Refused, match='equals the source'):
            hb_store.move_resume(ps[0], str(rel), disk=lambda: None)
        assert half.is_file() and not Path(f'{half}.partial_1').exists() and source.is_file() and not (hb.record_dir('method', ps[0])/'move_resume.json').exists()
        return
    out = hb_store.move_resume(ps[0], str(rel), disk=lambda: None)
    kept = Path(f'{half}.partial_1')
    partials = he.read_json(hb.record_dir('method', ps[0])/hb_store.PARTIALS_RECORD)
    assert out['passed'] and kept.is_file() and hb.sha(kept) == half_sha and hb.sha(half) == row['sha256_before'] and not source.exists()
    assert partials['partials'] == [dict(file=row['file'], half=str(half), sha256_half=half_sha, bytes_half=kept.stat().st_size, source=str(source), sha256_source=row['sha256_before'],
                                         kept=str(kept), sha256_kept=half_sha, same=True)]
    assert hb_store.where(ps[0]) == store and not hb.cell(ps[0]).exists()


def test_the_newest_stop_record_is_by_its_number_never_its_mtime(tmp_path, monkeypatch, unsealed):
    """HB diff review n4: the review's stop records (review_not_finalized, review_unit_stopped) carry one sequence number per pair; the newest is the largest -- a touched older record or a name that
    sorts first (_10 before _2) does not change it; a record without a number or two with one number are refused (to the lead)."""
    import os
    pair = pairs(1)[0]
    mode = review.holdout_mode(None)
    rev = hb.cell(pair)/'review'
    rev.mkdir(parents=True)
    assert review.next_stop_seq(pair) == 1 and review.last_stop_record(pair) is None
    first = hb.write_once(rev/'review_not_finalized.json', dict(kind='rounds', reason='test', code=21, seq=review.next_stop_seq(pair)))
    review.copy_to_home(mode, pair, first)
    stopped = review.unit_stop_record(pair, dict(status=None))
    later = hb.write_once(rev/'review_not_finalized_10.json', dict(kind='rounds', reason='test', code=21, seq=review.next_stop_seq(pair)))
    review.copy_to_home(mode, pair, later)
    home = review.home_records(pair)
    assert [he.read_json(home/n)['seq'] for n in ('review_not_finalized.json', 'review_unit_stopped.json', 'review_not_finalized_10.json')] == [1, 2, 3] and stopped == home/'review_unit_stopped.json'
    stamp = (home/'review_not_finalized_10.json').stat().st_mtime_ns
    os.utime(home/'review_not_finalized.json', ns=(stamp+10**9, stamp+10**9))
    assert review.last_stop_record(pair) == home/'review_not_finalized_10.json'
    hb.write_once(home/'review_unit_stopped_2.json', dict(kind='x'))                   # no number: the newest is not known
    with pytest.raises(hc.Refused, match='sequence number'):
        review.last_stop_record(pair)
    assert review.next_stop_seq(pair) == 4                                              # writing a stop record never fails on it
    (home/'review_unit_stopped_2.json').unlink()
    hb.write_once(home/'review_unit_stopped_3.json', dict(kind='x', seq=3))             # two with one number
    with pytest.raises(hc.Refused, match='sharing one'):
        review.last_stop_record(pair)


def test_a_redo_names_the_newest_stop_record_by_its_number(tmp_path, monkeypatch, unsealed):
    """n4 in the queue: the pair stopped (review_not_finalized, number 1), later its unit ended without a record (review_unit_stopped, number 2); the older record touched after: a redo whose previous
    record is the cell's review_not_finalized is refused, one whose previous record is review_unit_stopped runs."""
    import os
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    assert review.queue(system=FakeSystem([dict(review='stop', exit=21)]), echo=lambda m: None) == 21
    stopped = review.unit_stop_record(ps[0], dict(status=None, status_kind='no_record'))          # as the queue writes it after a unit that left no record
    cell_rec = hb.cell(ps[0])/'review'/'review_not_finalized.json'
    stamp = stopped.stat().st_mtime_ns
    for q in (cell_rec, review.home_records(ps[0])/'review_not_finalized.json'):
        os.utime(q, ns=(stamp+10**9, stamp+10**9))
    old = commit_hb_authorization('resume', ps[0], arm='review', lock_sha256=hb.sha(stopped), previous_record=str(cell_rec), previous_record_sha256=hb.sha(cell_rec))
    assert review.queue(redos={ps[0]: str(old)}, system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused']
    rel = commit_hb_authorization('resume', ps[0], arm='review', lock_sha256=hb.sha(stopped), previous_record=str(stopped), previous_record_sha256=hb.sha(stopped))
    assert review.queue(redos={ps[0]: str(rel)}, system=FakeSystem([dict(review='finalize'), dict(review='finalize')]), echo=lambda m: None) == 0
    assert set(he.read_json(hb.review_root()/'finalized_outputs.json')['pairs']) == set(ps)


def locked_world(tmp_path, monkeypatch):
    """Two finalized method cells, the 100 subjects' prediction files, the input lock written and committed in a temporary code tree.  Returns (pairs, repo)."""
    import hb_store
    ps = pairs(2)
    finalized_world(ps, monkeypatch)
    monkeypatch.setattr(hb, 'rows', lambda: [hc.pair_name(*r) for r in W.XW.holdout_rows()])          # HI's 100 subjects: the unsealed fixture's write_hi
    repo = tmp_path/'lock_repo'
    repo.mkdir()
    W.git(repo, 'init', '-q')
    path = hb_store.input_lock(hb_store.DEFAULT_LOCK_REL, repo=repo)
    doc = json.loads(path.read_text())
    fields = [r for r in doc['rows'] if r['kind'] == 'method_field']
    assert len([r for r in doc['rows'] if r['kind'] == 'prediction']) == 300 and len(fields) == 50 and sum(r['sha256'] is not None for r in fields) == 2
    assert len([r for r in doc['rows'] if r['kind'] == 'hi_record']) == 100 and doc['hi']['runs'] == [str(Path(he.HOLDOUT_SOURCES['pred_root']).parent)]
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', 'lock')
    return ps, repo


def test_the_input_lock_keeps_every_field(tmp_path, monkeypatch, unsealed):
    """Step 8 (decision_log 2026-11-10 ruling (5)): the lock names every method field by sha256; no field is deleted (delete-fields is gone), field_0600.npz moves with the cell."""
    import hb_store
    ps, repo = locked_world(tmp_path, monkeypatch)
    assert not hasattr(hb_store, 'delete_fields') and all((hb.cell(p)/'t88'/he.FIELD_FILE).is_file() for p in ps)
    with pytest.raises(hc.Refused, match='once'):
        hb_store.input_lock(hb_store.DEFAULT_LOCK_REL, repo=repo)
    with pytest.raises(hc.Refused, match='runs/holdout_B_v1/'):
        hb_store.input_lock('runs/20261109-hb/hb_input_lock.json', repo=repo)          # a record committed after the pinned commit goes under runs/holdout_B_v1/ (df8541d)
    hb_store.move_pair(ps[0], disk=lambda: None)
    assert (Path(hc.big_dir('method', ps[0]))/'t88'/he.FIELD_FILE).is_file()


def test_a_byte_changed_after_the_finalized_record_is_refused_by_the_move(tmp_path, monkeypatch, unsealed):
    """C22.H1-3: after step 5 the files are known by the finalized record's sha256; a terminal or review record changed by one byte is refused (the next steps: the move here; PQ and the evaluation
    entry check the same sha256)."""
    import hb_store
    ps = pairs(1)
    finalized_world(ps, monkeypatch)
    review_record = hb.cell(ps[0])/'review'/'review.json'
    review_record.write_text(review_record.read_text()+' ')
    with pytest.raises(hc.Refused, match='review.json is not the one'):
        hb_store.move_pair(ps[0], disk=lambda: None)
    assert hb.cell(ps[0]).is_dir() and not Path(hc.big_dir('method', ps[0])).exists()


def test_the_review_queue_refuses_a_pair_whose_step4_stopped_the_batch(tmp_path, monkeypatch, unsealed):
    """D110: an output whose method queue record did not go on (a certified field that is not the output, say) is not a step-4 terminal state: the review does not start."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    step = hb.record_dir('method', ps[1])/'step4.json'
    step.write_text(json.dumps(dict(json.loads(step.read_text()), went_on=False)))
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['refused'] and not (hb.review_root()/'review.lock').exists()


# ---- item 11: the withdrawal (D111 (ii), D123 (ii)) -------------------------------------------------------------------------------------------------
def stopped_review(monkeypatch):
    """Two pairs; the first one's review unit stopped (review_not_finalized.json, its /home copy), the batch stopped."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    assert review.queue(system=FakeSystem([dict(review='stop', exit=21)]), echo=lambda m: None) == 21
    copy = review.home_records(ps[0])/'review_not_finalized.json'
    assert copy.is_file() and hb.sha(copy) == hb.sha(hb.cell(ps[0])/'review'/'review_not_finalized.json')
    return ps, copy


@pytest.mark.parametrize('case', ['reason_a', 'reason_no_decision', 'not_committed', 'cell_record', 'terminal_there'])
def test_a_withdrawal_is_refused_unless_its_record_holds(tmp_path, monkeypatch, unsealed, case):
    ps, copy = stopped_review(monkeypatch)
    pair = ps[0]
    if case == 'reason_a':
        rel = commit_hb_authorization('no_resume', pair, arm='review', record=str(copy), record_sha256=hb.sha(copy), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a): 撤回')
    elif case == 'reason_no_decision':
        rel = commit_hb_authorization('no_resume', pair, arm='review', record=str(copy), record_sha256=hb.sha(copy), note=hc.NO_RESUME_NOTE, reason='BZ: C22.5-6 (b), 撤回')
    elif case == 'not_committed':
        rel = 'splits/bayesreg_v1/holdout_B_v1_hb_no_resume_9.json'
        W.XW.write_json(Path(hc.AUTH_ROOT)/rel, dict(schema='bayesreg.hb.no_resume.v1', tool='hb', arm='review', pair=pair, pair_set='holdout', record=str(copy), record_sha256=hb.sha(copy),
                                                     note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b)', written_by="lead on BZ's behalf", words='x', time_local='x', utc='2026-11-10T08:00:00Z'))
    elif case == 'cell_record':
        cell_rec = hb.cell(pair)/'review'/'review_not_finalized.json'
        rel = commit_hb_authorization('no_resume', pair, arm='review', record=str(cell_rec), record_sha256=hb.sha(cell_rec), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b)')
    else:
        rel = commit_hb_authorization('no_resume', pair, arm='review', record=str(copy), record_sha256=hb.sha(copy), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b)')
        hb.write_terminal(hb.cell(pair), pair=pair, status=he.HX, written_by='test', output_dir='output')
    before = {str(q.relative_to(hb.cell(pair))): hb.sha(q) for q in hb.cell(pair).rglob('*') if q.is_file()}
    with pytest.raises(hc.Refused):
        review.withdraw(pair, str(rel))
    assert {str(q.relative_to(hb.cell(pair))): hb.sha(q) for q in hb.cell(pair).rglob('*') if q.is_file()} == before and not (hb.cell(pair)/'review'/'review.json').exists()


def test_a_withdrawal_writes_the_terminal_record_keeps_every_file_and_the_batch_goes_on(tmp_path, monkeypatch, unsealed):
    """Item 11: BZ's committed no_resume record (arm review; C22.5-6 (b) and (甲); its record the /home copy of the last review_not_finalized): terminal.json hx_implementation_error, no output,
    rerun_record the authorization; review.json withdrawn; both copied to /home; nothing else changed; the queue's next start finalizes the batch with bz_decision (step 5)."""
    ps, copy = stopped_review(monkeypatch)
    pair = ps[0]
    before = {str(q.relative_to(hb.cell(pair))): hb.sha(q) for q in hb.cell(pair).rglob('*') if q.is_file()}
    rel = commit_hb_authorization('no_resume', pair, arm='review', record=str(copy), record_sha256=hb.sha(copy), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b): 撤回这一对的输出')
    term_path = review.withdraw(pair, str(rel))
    term = he.read_json(term_path)
    doc = he.read_json(hb.cell(pair)/'review'/'review.json')
    assert term['line_status'] == he.IMPLEMENTATION and term['output_dir'] is None and term['rerun_record'] == dict(path=str(rel), sha256=hb.sha(Path(hc.AUTH_ROOT)/rel))
    assert term['review_record'] == dict(path='review/review.json', sha256=hb.sha(hb.cell(pair)/'review'/'review.json')) and 'C22.5-6 (b)' in term['reason']
    assert doc['implementation_error']['kind'] == 'b' and doc['withdrawn']['record']['path'] == str(copy) and doc['finalized'] is False
    after = {str(q.relative_to(hb.cell(pair))): hb.sha(q) for q in hb.cell(pair).rglob('*') if q.is_file()}
    assert {k: v for k, v in after.items() if k in before} == before and set(after)-set(before) == {'terminal.json', 'review/review.json'}
    home = review.home_records(pair)
    assert hb.sha(home/'terminal.json') == hb.sha(term_path) and hb.sha(home/'review.json') == hb.sha(hb.cell(pair)/'review'/'review.json')
    with pytest.raises((hc.Refused, FileExistsError)):
        review.withdraw(pair, str(rel))
    assert review.queue(system=FakeSystem([dict(review='finalize')]), echo=lambda m: None) == 0
    final = he.read_json(hb.review_root()/'finalized_outputs.json')
    assert final['pairs'][pair]['line_status'] == he.IMPLEMENTATION and final['pairs'][pair]['bz_decision'] == dict(step=5, path=str(Path(hc.AUTH_ROOT)/rel), sha256=hb.sha(Path(hc.AUTH_ROOT)/rel))


# ---- item 13 (narrowed by decision_log 2026-11-10 N1, N2): the method queue takes no rejudgment; the review's is a trace -------------------------------
def commit_rejudgment(arm, pair, record, *, judgment='no_output', written_by="lead on BZ's behalf", words='（测试）BZ：不再跑', status='needs_lead_rejudgment', pair_set='holdout', commit=True,
                      **override):
    repo = W.XW.auth_repo()
    directory = Path(hc.AUTH_ROOT)/hc.REJUDGMENT_DIR_REL/arm
    k = 1
    while (directory/f'{pair}_{k}.json').exists():
        k += 1
    doc = dict(schema=hc.REJUDGMENT_SCHEMA, arm=arm, pair=pair, pair_set=pair_set, status=status, record=dict(path=str(record), sha256=hb.sha(record)), evidence=[], judgment=judgment,
               reason='test: the evidence could not be read (revision 1c item 1)', written_by=written_by, words=words, time_local='2026-11-10 2:30 AM CST', utc='2026-11-10T08:30:00Z')
    doc.update(override)
    path = W.XW.write_json(directory/f'{pair}_{k}.json', doc)
    if commit:
        W.git(repo, 'add', '-A')
        W.git(repo, 'commit', '-q', '-m', f'rejudgment {arm} {pair} {k}')
    return path


def test_the_rejudgment_reader_holds_only_the_format_of_revision_2_item_18(tmp_path, monkeypatch, unsealed):
    """holdout_common.rejudgment: the one reader (A5a, A5b and HB's configuration C queue): the keys of revision 2 item 18 (no resume key, N1), committed, the record it judges by sha256, the writer
    matching the judgment (no_output and P20: BZ's words)."""
    p1 = pairs(1)[0]
    record = Path(hc.AUTH_ROOT)/'runs'/'holdout_B_v1'/'judged.json'
    W.XW.write_json(record, dict(status='needs_lead_rejudgment'))
    assert 'resume' not in hc.REJUDGMENT_KEYS and hc.rejudgment('configc', p1, record=record) == (None, [])
    path = commit_rejudgment('configc', p1, record)
    found, problems = hc.rejudgment('configc', p1, record=record)
    assert problems == [] and found['k'] == 1 and found['document']['judgment'] == 'no_output' and found['path'] == str(path)
    for k, override in enumerate([dict(resume=None), dict(written_by='lead'), dict(words=''), dict(judgment='later'), dict(status='hx_provenance_rejected', judgment='infrastructure',
                                                                                                                   written_by='lead')], start=2):
        commit_rejudgment('configc', p1, record, **override)
        found, problems = hc.rejudgment('configc', p1, record=record)          # the highest k decides: each of these does not hold
        assert found is None and problems, override
    commit_rejudgment('configc', p1, record, judgment='unit_limit', written_by='lead', words='')
    assert hc.rejudgment('configc', p1, record=record)[0]['k'] == 7
    record.write_text('changed')                                            # NI3: bound to the record it judged
    assert hc.rejudgment('configc', p1, record=record)[0] is None


def test_the_method_queue_takes_no_rejudgment_record(tmp_path, monkeypatch, unsealed):
    """N1: a method cell stopped (here: a line status HE does not name) stays stopped whatever a rejudgment record says; it goes on by holdout_common's resume or no_resume only."""
    p1, p2 = pairs(2)
    assert hm.queue(pairs=[p1, p2], system=FakeSystem([dict(line='hx_unknown')]), echo=lambda m: None) == hb.EXIT['rejudge']
    step = hb.record_dir('method', p1)/'step4.json'
    commit_rejudgment('method', p1, step)
    assert hc.rejudgment('method', p1, record=step)[0] is not None and hm.state(p1) == 'stopped'
    assert hm.queue(pairs=[p1, p2], system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['resume']


def test_a_review_rejudgment_is_a_trace_and_the_pair_goes_on_by_a_withdrawal(tmp_path, monkeypatch, unsealed):
    """N1: the review unit ended without its record and without evidence (needs_lead_rejudgment): review_unit_stopped.json on /home; a committed rejudgment of it is recorded in the batch as a trace,
    the queue still stops (a redo or a withdrawal); BZ's withdrawal binds that stop record and the next start finalizes the batch."""
    ps = pairs(2)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    method_done(ps)
    assert review.queue(system=FakeSystem([dict(end=False, result=None)]), echo=lambda m: None) == hb.EXIT['rejudge']
    stopped = review.home_records(ps[0])/'review_unit_stopped.json'
    assert stopped.is_file() and review.last_stop_record(ps[0]) == stopped
    commit_rejudgment('review', ps[0], stopped)
    assert review.queue(system=FakeSystem([]), echo=lambda m: None) == hb.EXIT['resume']
    batches = sorted(Path(hc.batch_root()).glob('hb_review_*'))
    trace = he.read_json(batches[-1]/'batch.json')['cells'][0]
    assert trace['rejudgment_trace']['judgment'] == 'no_output'
    rel = commit_hb_authorization('no_resume', ps[0], arm='review', record=str(stopped), record_sha256=hb.sha(stopped), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (b): 撤回')
    term = he.read_json(review.withdraw(ps[0], str(rel)))
    assert term['line_status'] == he.IMPLEMENTATION and he.read_json(hb.cell(ps[0])/'review'/'review.json')['withdrawn']['record']['path'] == str(stopped)
    assert review.queue(system=FakeSystem([dict(review='finalize')]), echo=lambda m: None) == 0
    final = he.read_json(hb.review_root()/'finalized_outputs.json')['pairs'][ps[0]]
    assert final['line_status'] == he.IMPLEMENTATION and final['bz_decision']['path'] == str(Path(hc.AUTH_ROOT)/rel) and 'rejudgment' not in final


# ---- HI's predictions read as the union of its first run and its resume (A5b item 13; the A5a notes 6.2 table, rows HE and hb_store) ---------------------------------------------
def hi_subjects():
    return sorted({s for r in W.XW.holdout_rows() for s in r})


def resumed_hi(first_n=60, residue=2):
    """HI's first run killed after `first_n` subjects, its one resume writing the rest into the same root, `residue` half predictions in attempt1/pred/ (W.write_hi)."""
    import shutil
    everyone = hi_subjects()
    root = Path(he.HOLDOUT_SOURCES['pred_root']).parent
    shutil.rmtree(root)
    for path in Path(hc._hi().HOLDOUT_LOCK).parent.glob(Path(hc._hi().HOLDOUT_LOCK).name+'*'):
        path.unlink()
    return W.write_hi(everyone, first=everyone[:first_n], residue=everyone[first_n:first_n+residue])


def test_hb_s_pred_files_are_hi_s(unsealed):
    import hb_store
    assert hb_store.PRED_FILES == hb.HI_PRED_FILES and set(hb.HI_RECORD_OUTPUTS) < set(hb.HI_PRED_FILES)


@pytest.mark.parametrize('how', ['no_lock', 'missing_record', 'residue', 'changed_byte', 'other_root'])
def test_the_method_queue_refuses_before_any_unit_when_hi_is_not_complete(tmp_path, monkeypatch, unsealed, how):
    """A5b item 13 (HE's row): HE's units read pred/<s>/ by id; the queue reads HI as the union of its runs first (hb_core.hi_union) and refuses (22) before any gate or unit when its lock is not
    there, a subject has no record, a prediction without a record lies in pred/ (a killed run's residue no resume moved out), a file of a pair to run is not its record's, or the lock names
    another root.  batch.json records what was read."""
    p1 = pairs(1)[0]
    root, hi = Path(he.HOLDOUT_SOURCES['pred_root']).parent, hc._hi()
    victim = hc.split_pair(p1)[1]
    if how == 'no_lock':
        Path(hi.HOLDOUT_LOCK).unlink()
    elif how == 'missing_record':
        import shutil
        (root/'subjects'/f'{hi_subjects()[-1]}.json').unlink()
        shutil.rmtree(root/'pred'/hi_subjects()[-1])
        (root/'hi_record.json').write_text(json.dumps(dict(rows={})))
    elif how == 'residue':
        (root/'subjects'/f'{hi_subjects()[-1]}.json').unlink()
        (root/'hi_record.json').write_text(json.dumps(dict(rows={})))
    elif how == 'changed_byte':
        path = root/'pred'/victim/'structure_groups.npz'
        path.write_bytes(path.read_bytes()+b' ')
    else:
        Path(hi.HOLDOUT_LOCK).write_text(json.dumps(dict(out=str(root.parent/'elsewhere'), event='first_run'))+'\n')
    system = FakeSystem([dict(line=he.HX)])
    assert hm.queue(pairs=[p1], system=system, echo=lambda m: None) == hb.EXIT['refused']
    assert system.runs == [] and system.gates == [] and not hb.lock_path('method', p1).exists()
    batch = json.loads(next(Path(hc.batch_root()).glob('hb_method_*/batch.json')).read_text())
    assert batch['stop']['code'] == hb.EXIT['refused'] and 'A5b item 13' in batch['stop']['reason']
    if how == 'residue':
        assert batch['hi']['residue'] == [hi_subjects()[-1]] and batch['hi']['missing'] == [hi_subjects()[-1]]
    if how == 'changed_byte':
        assert batch['hi']['records'][victim]['sha256']['structure_groups.npz'] is not None and 'structure_groups.npz' in batch['stop']['reason']


def test_the_method_queue_reads_a_resumed_hi_root_and_hashes_only_the_pairs_to_run(tmp_path, monkeypatch, unsealed):
    """A5b item 13: HI's first run and its resume in the one root (the residue in attempt1/ is never read): complete; the subjects of the pairs to run are hashed against their records."""
    _, names = resumed_hi()
    p1 = pairs(1)[0]
    root = Path(he.HOLDOUT_SOURCES['pred_root']).parent
    opened = []
    real = hb.sha
    monkeypatch.setattr(hb, 'sha', lambda path: opened.append(Path(path)) or real(path))
    system = FakeSystem([dict(line=he.HX)])
    assert hm.queue(pairs=[p1], system=system, echo=lambda m: None) == 0 and len(system.runs) == 1
    batch = json.loads(next(Path(hc.batch_root()).glob('hb_method_*/batch.json')).read_text())
    assert sorted(batch['hi']['records']) == sorted(hc.split_pair(p1)) and batch['hi']['missing'] == [] and batch['hi']['residue'] == []
    assert not any('attempt1' in p.parts for p in opened) and {p.parent.name for p in opened if p.parent.parent == root/'pred'} == set(hc.split_pair(p1))
    assert (root/names['record']).is_file() and (root/'attempt1'/'pred').is_dir()


def test_the_input_lock_reads_a_resumed_hi_root_as_one_and_refuses_a_changed_byte(tmp_path, monkeypatch, unsealed):
    """A5b item 13 (hb_store's row): the lock rows each subject's HI record (kind hi_record, its run 0 or 1) and the three files at the record's sha256; nothing under attempt<k>/; a byte changed
    in a written subject's file, or a record lost, refuses (stage input_lock) and writes nothing."""
    import hb_store
    _, names = resumed_hi(first_n=60)
    ps = pairs(2)
    finalized_world(ps, monkeypatch)
    monkeypatch.setattr(hb, 'rows', lambda: [hc.pair_name(*r) for r in W.XW.holdout_rows()])
    root = Path(he.HOLDOUT_SOURCES['pred_root']).parent
    repo = tmp_path/'lock_repo'
    repo.mkdir()
    W.git(repo, 'init', '-q')
    victim = hi_subjects()[70]
    path = root/'pred'/victim/'probs36.npz'
    good = path.read_bytes()
    path.write_bytes(good+b' ')
    with pytest.raises(hc.Refused, match='A5b item 13'):
        hb_store.input_lock(hb_store.DEFAULT_LOCK_REL, repo=repo)
    path.write_bytes(good)
    record = root/'subjects'/f'{hi_subjects()[3]}.json'
    kept = record.read_bytes()
    record.unlink()
    with pytest.raises(hc.Refused, match='input_lock'):
        hb_store.input_lock(hb_store.DEFAULT_LOCK_REL, repo=repo)
    record.write_bytes(kept)
    assert not (repo/hb_store.DEFAULT_LOCK_REL).exists()
    doc = json.loads(hb_store.input_lock(hb_store.DEFAULT_LOCK_REL, repo=repo).read_text())
    records = {r['subject']: r for r in doc['rows'] if r['kind'] == 'hi_record'}
    assert len(records) == 100 and {r['run'] for r in records.values()} == {0, 1} and records[victim]['run'] == 1
    assert all('attempt1' not in r['path'] for r in doc['rows'] if 'path' in r) and doc['hi']['runs'] == [str(root)]
    preds = {(r['subject'], Path(r['path']).name): r['sha256'] for r in doc['rows'] if r['kind'] == 'prediction'}
    assert preds[(victim, 'probs36.npz')] == hb.sha(path) == json.loads((root/'subjects'/f'{victim}.json').read_text())['output_sha256']['probs36.npz']


def test_a_cell_whose_case_of_c22_5_6_is_not_known_takes_no_decision_in_either_place(tmp_path, monkeypatch, unsealed):
    """A5b diff review minor 1: an hx_implementation_error cell whose line.json does not name its case of C22.5-6 (implementation_kind None): hb_method.no_resume refuses BZ's (甲) and
    bz_declined (hb_core.bz_decision_auth, which state() and the finalized record's bz_decision use) takes it neither -- one rule; with the case named, both take it."""
    p1 = pairs(1)[0]
    assert hm.queue(pairs=[p1], system=FakeSystem([dict(line=he.IMPLEMENTATION, exit=9)]), echo=lambda m: None) == hb.EXIT['implementation']
    line = hb.cell(p1)/'line.json'
    doc = he.read_json(line)
    line.chmod(0o644)
    line.write_text(json.dumps(dict(doc, implementation_error=dict(doc['implementation_error'], kind=None))))
    assert hb.implementation_kind(hb.cell(p1)) is None
    step = hb.record_dir('method', p1)/'step4.json'
    rel = commit_hb_authorization('no_resume', p1, arm='method', record=str(step), record_sha256=hb.sha(step), note=hc.NO_RESUME_NOTE, reason='BZ (甲) on C22.5-6 (a): 补成恒等')
    with pytest.raises(hc.Refused, match='not known'):
        hm.no_resume(p1, str(rel))
    assert hm.bz_declined(p1) is None and hm.state(p1) == 'stopped'
    line.write_text(json.dumps(doc))                                                   # the case named again (a): both take the same record
    assert hm.bz_declined(p1).path.endswith(Path(rel).name) and hm.no_resume(p1, str(rel)) == hb.terminal_path(hb.cell(p1)) and hm.state(p1) == 'final'
