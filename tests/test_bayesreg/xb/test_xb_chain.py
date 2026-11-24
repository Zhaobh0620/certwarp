"""The chain (prereg §2.3 (d)), the regression tools (§5.1 item 7) and the limit experiment tool (item 19, revision 1c item 2): start strings, the queue with a fake machine, the systemd-run shim (exit
99), the unit readings and the terminal records the chain writes; nothing is launched."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .. import _p0_pins as P0
from . import _world as W

hc, xc, xb_drive = W.hc, W.xc, W.xb_drive
import xb_chain  # noqa: E402
import xb_limit_experiment as xl  # noqa: E402
import xb_regress  # noqa: E402

QUIET = dict(max=0, oom=0, oom_kill=0)


def test_the_unit_start_string():
    pair = hc.pair_name(*W.holdout_rows()[0])
    argv = xb_chain.unit_argv('fireants_greedy', 'holdout', pair)
    text = ' '.join(argv)
    assert argv[:4] == ['systemd-run', '--user', '--wait', '--collect'] and '--unit=tag-hb-x-fireants_greedy-0423-0099' in argv
    assert [a for a in argv if a.startswith('--property=')] == [f'--property={p}' for p in ('MemoryMax=10G', 'MemorySwapMax=0', 'OOMPolicy=continue', 'RuntimeMaxSec=1800')]
    assert not any(a.startswith('--unit=seq-vp') for a in argv) and 'pytest' not in text and str(hc.stage_pair_path(pair)) not in text
    i = argv.index('--')
    assert argv[i-5].endswith('experiments/xb/xb_unit.py') and argv[i-4] == '--record-dir' and argv[i+1:i+4] == [str(xc.PY), '-I', '-B'] and argv[i+4].endswith('experiments/xb/xb_drive.py')
    assert argv[i+5:] == ['run', '--arm', 'fireants_greedy', '--pair-set', 'holdout', '--pair', pair]
    assert '--setenv=XB_UNIT=tag-hb-x-fireants_greedy-0423-0099' in argv and not any(a.startswith('--setenv=XB_BATCH_DIR=') for a in argv)
    batch = hc.batch_root()/'xb_x'
    assert f'--setenv=XB_BATCH_DIR={batch}' in xb_chain.unit_argv('fireants_greedy', 'holdout', pair, batch=batch)
    # I4: ConvexAdam-v4seg gets the pinned path of the method's T88 run.json (development: HS's; hold-out: HB's pin, the method cell on D:)
    dev = xb_chain.unit_argv('convexadam_v4seg', 'dev', hc.DEV_PAIRS[7])
    assert dev[dev.index('--method-run')+1] == str(xc.HS_RUNS/'SHx24_rest600'/'0445-0446'/'run.json')
    assert xc.method_t88_run('dev', hc.DEV_PAIRS[4]) == xc.HS_RUNS/'SHx24_first5'/'0442-0443'/'run.json'
    held = xb_chain.unit_argv('convexadam_v4seg', 'holdout', pair)                 # HB's records name the hold-out's (HB review m13): none yet, so no --method-run and the driver refuses
    assert '--method-run' not in held and '--method-run' not in xb_chain.unit_argv('convexadam', 'dev', hc.DEV_PAIRS[0])
    store = hc.big_dir('method', pair)/'t88'/'run.json'                            # after step 5 and the move: the finalized record's sha256 and the move record's row of t88/run.json
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text('{}')
    digest = xc.sha256(store)
    W.write_json(hc.HOLDOUT_ROOT/'review'/'finalized_outputs.json', dict(pairs={pair: dict(part=dict(t88_repair={'t88/run.json': digest}))}))
    W.write_json(hc.record_dir('method', pair)/'move.json', dict(passed=True, files=[dict(file='t88/run.json', dst=str(store), ok=True, sha256_after=digest)]))
    held = xb_chain.unit_argv('convexadam_v4seg', 'holdout', pair)
    assert held[held.index('--method-run')+1] == str(store) and xc.method_t88_sha256('holdout', pair) == digest
    chain = xb_chain.chain_launch_string('holdout', ['fireants_greedy'])
    assert f'--unit={xc.CHAIN_UNIT}' in chain and xc.CHAIN_UNIT.startswith('tag-hb-')


def test_the_sandbox_roots_hold_no_pytest_and_the_start_string_check_stays(monkeypatch):
    """The roots lie in a tempfile.mkdtemp(prefix='xbt_') directory, not under pytest's basetemp, so the start strings pass the check; the check itself still refuses the word."""
    assert 'pytest' not in str(hc.RESULTS) and hc.RESULTS.parent.name.startswith('xbt_')
    assert 'pytest' not in ' '.join(xb_chain.unit_argv('ants_s', 'dev', hc.DEV_PAIRS[0]))
    monkeypatch.setattr(xc, 'DEV_ROOT', hc.RESULTS.parent/'pytest-of-x'/'xb_dev_v1')
    with pytest.raises(ValueError):
        xb_chain.unit_argv('ants_s', 'dev', hc.DEV_PAIRS[0])


def test_plan_lists_the_cells_by_arm_in_the_pair_lists_order(capsys):
    assert xb_chain.main(['plan', '--pair-set', 'holdout', '--arms', 'fireants_greedy,ants_s']) == 0
    lines = capsys.readouterr().out.splitlines()
    rows = [hc.pair_name(*r) for r in W.holdout_rows()]
    assert len(lines) == 100 and [l.split()[1] for l in lines[:50]] == rows and all(l.split()[0] == 'ants_s' for l in lines[50:])
    assert all(' not_run: ' in l for l in lines)


class FakeSystem(xb_chain.System):
    """Each run plays one outcome for its cell: what the driver and the supervisor would write, and systemd-run's result line."""

    def __init__(self, outcomes, *, dmesg='', journal=''):
        self.outcomes, self.kernel, self.journal_text, self.runs = list(outcomes), dmesg, journal, []

    def run(self, argv, log_path):
        self.runs.append(argv)
        outcome = self.outcomes.pop(0)
        i = argv.index('--')
        arm, pair_set, pair = argv[argv.index('--arm')+1], argv[argv.index('--pair-set')+1], argv[argv.index('--pair')+1]
        paths = xc.cell_paths(arm, pair_set, pair)
        resume = '--resume-record' in argv
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text(json.dumps(dict(utc='2026-11-08T00:00:00Z')))
        batch = next(a.split('=', 2)[2] for a in argv if a.startswith('--setenv=XB_BATCH_DIR='))
        if outcome.get('refusal'):                                                     # the driver refused before its lock: a refusal record in the batch (I1), numbered as the driver numbers it (A5a)
            hc.write_refusal(batch, dict(tool='xb', arm=arm, pair=pair, pair_set=pair_set, stage=outcome['refusal'],
                                         seq=hc.next_refusal_seq(xc.roots(pair_set)[0], arm=arm, pair=pair, pair_set=pair_set)))
        elif outcome.get('lock', True):
            hc.take_lock(hc.resume_lock_path(paths['lock']) if resume else paths['lock'], hc.lock_line(arm=arm, pair=pair, frozen_sha256=xc.FROZEN_CONFIG_SHA256))
        if outcome.get('status'):
            record = paths['resume_result'] if resume else paths['result']
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(dict(status=outcome['status'], export_check=dict(suspicious=False) if outcome['status'] == 'ok' else None, failure=dict(kind='x', reason='y'),
                                              wrapper_error_policy=outcome.get('policy'), disp=dict(file='d', sha256='e') if outcome['status'] == 'ok' else None)))
        end = dict(utc='2026-11-08T00:01:00Z', exit_code=outcome.get('exit', 0), killed_by_signal=9 if outcome.get('exit') == -9 else None, killed_by_supervisor=False,
                   sigterm_received=outcome.get('result') == 'timeout', memory_events=outcome.get('events', QUIET), cgroup=None, dmesg_readable=outcome.get('dmesg_readable', True),
                   oom_lines=[] if outcome.get('dmesg_readable', True) else None, dxg_lines=0 if outcome.get('dmesg_readable', True) else None)
        if outcome.get('end', True):
            (unit_dir/'unit_end.json').write_text(json.dumps(end))
        Path(log_path).write_text(f"Running as unit: {argv[4][7:]}.service\nFinished with result: {outcome.get('result', 'success')}\n" if outcome.get('result', 'success') else 'no result line\n')
        return outcome.get('rc', 0)

    def uptime(self):
        return 10.

    def dmesg(self):
        return self.kernel

    def journal(self, unit, since):
        return self.journal_text

    def gate(self):
        return dict(passed=True)

    def disk(self, need_d, need_home):
        return dict(passed=True, free_gib={})

    def sleep(self, seconds):
        raise AssertionError('no hold file in the tests')


def queue(arms, pairs, system, **kw):
    return xb_chain.queue('dev', arms, pairs=pairs, system=system, log=lambda m: None, **kw)


def test_the_queue_continues_on_ok_and_method_failures_and_skips_done_cells(monkeypatch):
    W.fake_checks(monkeypatch)
    pairs = list(hc.DEV_PAIRS[:3])
    system = FakeSystem([dict(status='ok'), dict(status='method_failure', rc=1, result='exit-code'), dict(status='ok')])
    assert queue(['fireants_greedy'], pairs, system) == 0
    assert len(system.runs) == 3
    again = FakeSystem([])
    assert queue(['fireants_greedy'], pairs, again) == 0 and again.runs == []
    reading = json.loads((xb_chain.unit_record_dir('fireants_greedy', 'dev', pairs[1])/'unit_reading.json').read_text())
    assert reading['status'] is None and reading['systemd_result'] == 'exit-code'
    batch = sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json'))
    assert len(batch) == 2 and json.loads(batch[0].read_text())['exit'] == 0


@pytest.mark.parametrize('outcome, code, status', [
    (dict(status='infra_failure', rc=20, result='exit-code'), 20, 'infra_failure'),
    (dict(status=None, result='timeout', rc=1), 0, 'unit_limit'),                                              # RuntimeMaxSec without OOM: unit_limit, written by the chain, the batch goes on
    (dict(status=None, result='timeout', rc=1, events=dict(max=4, oom=1, oom_kill=1)), 20, 'infra_failure'),   # timeout and OOM: OOM first (rule yi)
    (dict(status=None, result='signal', rc=1, exit=-9), 20, 'infra_failure'),                                   # the driver killed: an exit -9 is OOM evidence
    (dict(status=None, result='timeout', rc=1, end=False), 25, 'needs_lead_rejudgment'),                       # no end probe: never read as 0
    (dict(status=None, result='exit-code', rc=1, exit=1), 23, 'wrapper_error'),                                 # the driver exited after its lock without a record
    (dict(status='wrapper_error', rc=23, result='exit-code', policy='stop'), 23, 'wrapper_error'),
    (dict(status='ok', events=dict(max=1, oom=1, oom_kill=1)), 20, 'ok'),                                      # the unit's OOM evidence overrides the driver's ok: STOP, the cell needs a resume
    (dict(status='method_failure', rc=1, result='exit-code', events=dict(max=1, oom=0, oom_kill=1)), 20, 'method_failure'),     # I2: ANTs exit 0 / no Warp after an OOM kill
    (dict(status='ok', result='oom-kill'), 20, 'ok'),                                                           # m3: systemd's Result oom-kill is OOM evidence
])
def test_unit_outcomes(monkeypatch, outcome, code, status):
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[0]
    system = FakeSystem([outcome, dict(status='ok')])
    assert queue(['ants_s'], [pair, hc.DEV_PAIRS[1]], system) == code
    paths = xc.cell_paths('ants_s', 'dev', pair)
    doc = json.loads(paths['result'].read_text())
    assert doc['status'] == status                                                  # the driver's record (or the chain's, when the driver left none) stays as written
    assert len(system.runs) == (2 if code == 0 else 1)
    if code:
        assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'needs_resume'
    overridden = paths['overridden'].exists()
    assert overridden == (status in ('ok', 'method_failure'))
    if overridden:                                                                  # I2: overridden.json supersedes it
        over = json.loads(paths['overridden'].read_text())
        assert over['status'] == 'infra_failure' and over['overrides'] == dict(path=str(paths['result']), sha256=xc.sha256(paths['result']), status=status, disp=doc.get('disp'))
        assert over['disp'] is None and xc.effective_record(paths) == over and not xc.attempt_complete(paths)


def test_kernel_events_stop_the_batch(monkeypatch):
    W.fake_checks(monkeypatch)
    pairs = list(hc.DEV_PAIRS[:2])
    assert queue(['ants_s'], pairs, FakeSystem([dict(status='ok')], dmesg='[ 20.0] dxgk: create_allocation failed: -12\n')) == 20
    pairs = list(hc.DEV_PAIRS[2:4])
    none = '[ 20.000000] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg=/user.slice/x.service,task=x,pid=2,uid=1000\n'
    assert queue(['ants_s'], pairs, FakeSystem([dict(status='ok')], dmesg=none)) == 20
    pairs = list(hc.DEV_PAIRS[4:6])
    assert queue(['ants_s'], pairs, FakeSystem([dict(status='ok', dmesg_readable=False)], dmesg=None)) == 25     # dmesg unreadable for the chain and the supervisor: the lead
    pairs = list(hc.DEV_PAIRS[6:8])
    assert queue(['ants_s'], pairs, FakeSystem([dict(status='ok'), dict(status='ok')], dmesg=None)) == 0          # the supervisor's window stands in for the chain's


def test_an_overridden_cell_resumes_once_through_a_committed_authorization_and_never_stays_ok(monkeypatch, tmp_path):
    """I2: the driver's complete record overridden by the unit's evidence: the record stays, overridden.json supersedes it, the cell needs a resume; without an authorization the queue stops (24);
    the committed resume authorization names the overriding record and load_resume accepts it; the resumed attempt, overridden again, is final with no output (infrastructure), never ok."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[0]
    oom = dict(max=1, oom=1, oom_kill=1)
    assert queue(['ants_s'], [pair], FakeSystem([dict(status='method_failure', rc=1, result='exit-code', events=oom)])) == 20
    paths = xc.cell_paths('ants_s', 'dev', pair)
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'needs_resume'
    assert queue(['ants_s'], [pair], FakeSystem([])) == 24
    path = W.authorize('resume', 'ants_s', 'dev', pair)
    doc = json.loads(path.read_text())
    assert doc['previous_record'] == str(paths['overridden']) and doc['previous_record_sha256'] == xc.sha256(paths['overridden'])
    resume = hc.load_resume(path, tool='xb', arm='ants_s', pair=pair, pair_set='dev', lock=paths['lock'], previous_record=paths['overridden'], complete=lambda: xc.attempt_complete(paths))
    assert resume.previous_record == str(paths['overridden'])
    with pytest.raises(hc.Refused):                                                 # the driver's own record is complete: without the override nothing would be resumed
        hc.load_resume(path, tool='xb', arm='ants_s', pair=pair, pair_set='dev', lock=paths['lock'], previous_record=paths['result'], complete=lambda: True)
    rel = str(path.relative_to(hc.AUTH_ROOT))
    system = FakeSystem([dict(status='ok', events=oom)])
    assert queue(['ants_s'], [pair], system, resumes={('ants_s', pair): rel}) == 20 and '--resume-record' in system.runs[0]
    state, effective = xb_chain.cell_state('ants_s', 'dev', pair)
    assert state == 'final' and effective['status'] == 'infra_failure' and effective['no_output_class'] == 'infrastructure'
    assert json.loads(paths['resume_result'].read_text())['status'] == 'ok' and paths['resume_overridden'].exists()
    assert xb_chain.collect_intermediates('dev', ['ants_s'], tmp_path/'m.tsv')['n_cells'] == 0       # an overridden ok is not an output


def test_an_overridden_cell_is_made_final_by_a_committed_no_resume_authorization(monkeypatch):
    """I2, I3: the no_resume authorization of an overridden cell names overridden.json and its sha256; committed, it makes the cell final."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[1]
    assert queue(['ants_s'], [pair], FakeSystem([dict(status='ok', events=dict(max=1, oom=1, oom_kill=1))])) == 20
    paths = xc.cell_paths('ants_s', 'dev', pair)
    path = W.authorize('no_resume', 'ants_s', 'dev', pair)
    assert json.loads(path.read_text())['record'] == str(paths['overridden'])
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'final'
    again = FakeSystem([])
    assert queue(['ants_s'], [pair], again) == 0 and again.runs == []


def test_a_cell_refused_before_its_lock_runs_again_only_with_a_rerun_authorization(monkeypatch):
    """I1: the driver's refusal is a record in the batch (XB_BATCH_DIR), nothing in the cell; the cell's state is refused; the queue stops (22) until a committed rerun authorization covers it."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[2]
    first = FakeSystem([dict(refusal='admission', lock=False, rc=2, result='exit-code', end=False)])
    assert queue(['fireants_greedy'], [pair], first) == 22
    [record] = W.refusals('dev')
    assert record.parent.parent.parent == hc.batch_root(xc.DEV_ROOT) and not xc.cell_paths('fireants_greedy', 'dev', pair)['result'].exists()
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'refused'
    blocked = FakeSystem([])
    assert queue(['fireants_greedy'], [pair], blocked) == 22 and blocked.runs == []
    W.authorize('rerun', 'fireants_greedy', 'dev', pair)
    allowed = FakeSystem([dict(status='ok')])
    assert queue(['fireants_greedy'], [pair], allowed) == 0 and len(allowed.runs) == 1
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'done'
    first, second = xb_chain.unit_dirs('fireants_greedy', 'dev', pair)                # each attempt its own supervisor directory (unit_start.json is written once)
    assert second.name == 'unit_2' and allowed.runs[0][allowed.runs[0].index('--record-dir')+1] == str(second)
    assert json.loads((first/'unit_reading.json').read_text())['locked'] is False and json.loads((second/'unit_reading.json').read_text())['locked'] is True


def test_the_batch_record_holds_each_arms_checks_and_a_holdout_batch_needs_the_pinned_preregistration(monkeypatch):
    """m14: batch.json records the frozen configuration's sha256 now and pinned, and each arm's frozen and environment checks; I5: a hold-out batch stops at its start while PREREG_SHA256 is None."""
    W.fake_checks(monkeypatch)
    monkeypatch.setattr(xc, 'PREREG_SHA256', None)                  # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, as before
    assert queue(['fireants_greedy'], [hc.DEV_PAIRS[0]], FakeSystem([dict(status='ok')])) == 0
    [batch] = sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json'))
    doc = json.loads(batch.read_text())
    assert doc['frozen']['sha256'] == xc.sha256(xc.REPO/xc.FROZEN_CONFIG_REL) == doc['frozen']['pinned_sha256']
    assert doc['arm_checks']['fireants_greedy']['environment']['passed'] and doc['arm_checks']['fireants_greedy']['frozen']['passed']
    assert doc['preregistration']['pinned_sha256'] is None and doc['preregistration']['problem']
    monkeypatch.setattr(hc, 'code_identity', lambda run=None, repo=None: dict(commit='c'*40, clean=True, status='', untracked=[]))
    system = FakeSystem([])
    system.pinned_code = W.pinned_ok                                                # rule one passes here (its own test below)
    assert xb_chain.queue('holdout', ['fireants_greedy'], system=system, log=lambda m: None) == 22 and system.runs == []
    [hb] = sorted((hc.HOLDOUT_ROOT/'batches').glob('xb_*/batch.json'))
    assert 'I5' in json.loads(hb.read_text())['stop']['reason'] and json.loads(hb.read_text())['pinned_code']['pinned'] == W.PINNED


def test_the_batch_record_holds_the_preregistration_pinned_at_p0(monkeypatch):
    """P0: with PREREG_SHA256 as committed (the committed preregistration's sha256), a batch records the file's sha256 equal to the pin and no preregistration problem."""
    P0.assert_pinned(xc.PREREG_SHA256, xc.PREREG_REL)
    W.fake_checks(monkeypatch)
    assert queue(['fireants_greedy'], [hc.DEV_PAIRS[0]], FakeSystem([dict(status='ok')])) == 0
    [batch] = sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json'))
    doc = json.loads(batch.read_text())
    assert doc['preregistration']['sha256'] == doc['preregistration']['pinned_sha256'] == xc.PREREG_SHA256 and doc['preregistration']['problem'] is None
    assert xc.prereg_problem() is None


def test_a_holdout_batch_stops_at_its_start_unless_the_code_is_the_pinned_commits(monkeypatch):
    """A5a (contract C24.3 rule one; decision_log 2026-11-10 ~2:28 AM CST n8, ~2:59 AM CST (1)): a hold-out batch checks the code against P1 of runs/holdout_B_v1/pinned_code.json before any cell;
    the real check on this code tree (main has no pinned record) refuses, nothing is started."""
    W.fake_checks(monkeypatch)
    monkeypatch.setattr(hc, 'code_identity', lambda run=None, repo=None: dict(commit='c'*40, clean=True, status='', untracked=[]))
    monkeypatch.setattr(xc, 'PREREG_SHA256', 'f'*64)
    system = FakeSystem([])
    assert xb_chain.queue('holdout', ['fireants_greedy'], system=system, log=lambda m: None) == 22 and system.runs == []
    [hb] = sorted((hc.HOLDOUT_ROOT/'batches').glob('xb_*/batch.json'))
    doc = json.loads(hb.read_text())
    assert 'rule one' in doc['stop']['reason'] and doc['pinned_code']['passed'] is False and 'pinned_code.json' in doc['stop']['reason']


def needs_rejudgment(pair, arm='ants_s'):
    """A cell whose unit left no end probe: needs_lead_rejudgment (written by the chain), the batch stops 25."""
    assert queue([arm], [pair], FakeSystem([dict(status=None, result='timeout', rc=1, end=False)])) == 25
    paths = xc.cell_paths(arm, 'dev', pair)
    assert json.loads(paths['result'].read_text())['status'] == 'needs_lead_rejudgment'
    return paths


@pytest.mark.parametrize('judgment, writer, words, written', [('unit_limit', 'lead', '', True), ('no_output', 'BZ', '（测试）BZ：不再跑', True), ('no_output', 'lead', '', False),
                                                              ('infrastructure', 'lead', '', False)])
def test_a_needs_lead_rejudgment_cell_is_written_by_a_committed_rejudgment(monkeypatch, judgment, writer, words, written):
    """A5a (prereg revision 2 item 18; decision_log 2026-11-10 NI2, NI3, N1, N2): holdout_common.rejudgment, the one reader; unit_limit (the lead's) and no_output (BZ's words) write the cell and the
    batch goes past it; no_output written by the lead is not taken; infrastructure only classifies: the cell stops the batch -- 24 (resume) since A5a review m5, 25 when the record does not hold --
    until a committed resume record, and the resumed attempt is final.  A final cell's row in batch.json names the record that made it final, with its sha256 (m5)."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[0]
    paths = needs_rejudgment(pair)
    W.commit_rejudgment('ants_s', pair, paths['result'], judgment=judgment, written_by=writer, words=words, pair_set='dev')
    state, _ = xb_chain.cell_state('ants_s', 'dev', pair)
    again = FakeSystem([dict(status='ok')])
    if written:
        assert state == 'final' and queue(['ants_s'], [pair, hc.DEV_PAIRS[1]], again) == 0 and len(again.runs) == 1        # the next pair runs
        [row] = [c for b in sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json')) for c in json.loads(b.read_text()).get('cells', []) if c['pair'] == pair and c['state_before'] == 'final']
        found, _ = hc.rejudgment('ants_s', pair, record=paths['result'], pair_set='dev')
        assert row['final_by'] == dict(by=f'rejudgment {judgment} (prereg revision 2 item 18)', path=found['path'], sha256=found['sha256'], record=str(paths['result']),
                                       record_sha256=xc.sha256(paths['result']))
        return
    assert state == 'needs_resume'
    assert queue(['ants_s'], [pair], again) == (24 if judgment == 'infrastructure' else 25) and again.runs == []
    [batch] = [b for b in sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json')) if json.loads(b.read_text()).get('cells', [{}])[0].get('state_before') == 'needs_resume']
    doc = json.loads(batch.read_text())
    reason = doc['stop']['reason']
    assert 'rejudg' in reason and ('only classifies' in reason if judgment == 'infrastructure' else 'written_by' in reason), reason
    if judgment == 'infrastructure':
        assert doc['cells'][0]['rejudgment']['k'] == 1 and doc['cells'][0]['rejudgment']['sha256'] == hc.rejudgment('ants_s', pair, record=paths['result'], pair_set='dev')[0]['sha256']
        rel = str(W.authorize('resume', 'ants_s', 'dev', pair).relative_to(hc.AUTH_ROOT))
        resumed = FakeSystem([dict(status=None, result='timeout', rc=1, end=False)])
        assert queue(['ants_s'], [pair], resumed, resumes={('ants_s', pair): rel}) == 25 and '--resume-record' in resumed.runs[0]
        assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'final'                # n5: a resumed attempt is final (the evaluation entry needs the rejudgment of its record)


def test_a_rejudgment_binds_the_record_it_judged(monkeypatch):
    """NI3: the rejudgment's record is the effective terminal record with its sha256 now; a record changed since, or another record, is not judged (the cell stays stopped, 25)."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[2]
    paths = needs_rejudgment(pair)
    W.commit_rejudgment('ants_s', pair, paths['result'], judgment='unit_limit', pair_set='dev', bound=dict(path=str(paths['result']), sha256='0'*64))
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'needs_resume'
    W.commit_rejudgment('ants_s', pair, paths['result'], judgment='unit_limit', pair_set='dev', k=2)
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'final'                    # the highest k, bound to the record
    W.commit_rejudgment('ants_s', pair, paths['result'], judgment='unit_limit', pair_set='dev', k=3, commit=False)
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'needs_resume'             # the highest k is not committed: nothing is judged


def test_a_cell_refused_again_after_its_rerun_is_written_by_bzs_no_resume_and_its_refusals_are_copied(monkeypatch):
    """A5a (prereg revision 2 item 17 (f); decision_log 2026-11-10 ~2:52, ~2:53 AM CST): a cell refused before its lock, run again by a rerun authorization and refused again: every refusal record is
    copied into the cell's record directory (refusals/<batch>__<name>, byte for byte, <copy>.sha256 naming the source); BZ's committed no_resume record (tool xb, 未批补跑) naming the newest copy
    (by seq) writes the cell -- no output, P20 -- and the batch goes past it; one naming an older refusal record is not taken."""
    W.fake_checks(monkeypatch)
    pair, other = hc.DEV_PAIRS[3], hc.DEV_PAIRS[4]
    refused = dict(refusal='admission', lock=False, rc=2, result='exit-code', end=False)
    assert queue(['fireants_greedy'], [pair], FakeSystem([refused])) == 22
    paths = xc.cell_paths('fireants_greedy', 'dev', pair)
    [copy] = hc.refusal_copies(paths['record'])
    [source] = W.refusals('dev')
    assert copy.name == f'{source.parent.parent.name}__{source.name}' and copy.read_bytes() == source.read_bytes()
    assert copy.with_name(copy.name+'.sha256').read_text() == f'{xc.sha256(copy)}  {source}\n'
    W.authorize('rerun', 'fireants_greedy', 'dev', pair)
    assert queue(['fireants_greedy'], [pair], FakeSystem([dict(refused, refusal='method_run')])) == 22
    copies = hc.refusal_copies(paths['record'])
    assert len(copies) == 2 and sorted(hc.refusal_seq(c) for c in copies) == [1, 2]
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'refused'
    older = min(copies, key=hc.refusal_seq)
    W.commit_authorization('dev', 'no_resume', dict(xb_drive.authorization_draft('no_resume', 'fireants_greedy', 'dev', pair, written_by='BZ', words='（测试）BZ：P20 不再跑',
                                                                                  time_local='2026-11-10 11:00 AM CST', reason='P20 twice; BZ declines')[1],
                                                    record=str(older), record_sha256=xc.sha256(older)))
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'refused'          # the older one is not the newest
    newest = max(copies, key=hc.refusal_seq)
    original = Path(newest.with_name(newest.name+'.sha256').read_text().split('  ', 1)[1].strip())
    assert original.is_file() and original.read_bytes() == newest.read_bytes() and original != newest
    W.commit_authorization('dev', 'no_resume', dict(xb_drive.authorization_draft('no_resume', 'fireants_greedy', 'dev', pair, written_by='BZ', words='（测试）BZ：P20 不再跑，指原件',
                                                                                  time_local='2026-11-10 11:00 AM CST', reason='P20 twice; BZ declines')[1],
                                                    record=str(original), record_sha256=xc.sha256(original)))
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'refused'          # A5a review m4: only the copy in the record directory (prereg revision 2 item 17 (f)), not its source
    _, vague = xb_drive.authorization_draft('no_resume', 'fireants_greedy', 'dev', pair, written_by='BZ', words='（测试）BZ：不再跑', time_local='2026-11-10 11:00 AM CST',
                                            reason='refused twice; BZ declines')
    W.commit_authorization('dev', 'no_resume', vague)
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'refused'          # A5b review I2: the reason does not name P20 (bayesreg.io.rejudgment.p20_reason_problem)
    path, draft = xb_drive.authorization_draft('no_resume', 'fireants_greedy', 'dev', pair, written_by='BZ', words='（测试）BZ：P20 不再跑', time_local='2026-11-10 11:00 AM CST',
                                               reason='P20 twice; BZ declines')
    newest = max(copies, key=hc.refusal_seq)
    assert draft['record'] == str(newest) and draft['record_sha256'] == xc.sha256(newest) and draft['note'] == hc.NO_RESUME_NOTE
    W.commit_authorization('dev', 'no_resume', draft)
    assert xb_chain.cell_state('fireants_greedy', 'dev', pair)[0] == 'final'
    after = FakeSystem([dict(status='ok')])
    assert queue(['fireants_greedy'], [pair, other], after) == 0 and len(after.runs) == 1 and other in after.runs[0] and pair not in after.runs[0]
    [row] = [c for b in sorted((xc.DEV_ROOT/'batches').glob('xb_*/batch.json')) for c in json.loads(b.read_text()).get('cells', []) if c['pair'] == pair and c['state_before'] == 'final']
    assert row['final_by']['by'].startswith('no_resume P20') and row['final_by']['record'] == str(newest) and row['final_by']['record_sha256'] == xc.sha256(newest)     # A5a review m5
    assert row['final_by']['sha256'] == xc.sha256(row['final_by']['path'])


def test_refusal_copies_need_their_numbers_to_name_the_newest(monkeypatch):
    """HB diff review n4 (as hb_configc): the newest refusal record is the largest seq; a copy without one, or two with the same, refuse (to the lead) -- never mtime or the name."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[5]
    batch = hc.batch_root(xc.DEV_ROOT)/'xb_test'
    for seq in (1, None):
        hc.write_refusal(batch, dict(tool='xb', arm='ants_s', pair=pair, pair_set='dev', stage='admission', seq=seq))
    record_dir = xc.cell_paths('ants_s', 'dev', pair)['record']
    kept = hc.keep_refusals(xc.DEV_ROOT, record_dir, arm='ants_s', pair=pair, pair_set='dev')
    assert len(kept) == 2 and sorted(k['seq'] is None for k in kept) == [False, True]       # copying does not need the number
    with pytest.raises(hc.Refused, match='sequence number'):
        hc.newest_refusal(record_dir)
    assert hc.keep_refusals(xc.DEV_ROOT, record_dir, arm='ants_s', pair=pair, pair_set='dev') == kept       # copied once; checked again, not rewritten
    copy = Path(kept[0]['path'])
    copy.write_bytes(copy.read_bytes()+b' ')
    with pytest.raises(hc.Refused, match='refusal_copy'):
        hc.keep_refusals(xc.DEV_ROOT, record_dir, arm='ants_s', pair=pair, pair_set='dev')


def test_a_cell_that_needs_a_resume_stops_the_queue(monkeypatch):
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[0]
    paths = xc.cell_paths('deeds', 'dev', pair)
    hc.take_lock(paths['lock'], hc.lock_line(arm='deeds'))                       # a run that left no record (a machine restart)
    system = FakeSystem([])
    assert queue(['deeds'], [pair], system) == 24 and system.runs == []


def test_p19_skips_the_rest_of_a_stopped_arm(monkeypatch):
    W.fake_checks(monkeypatch)
    pairs = list(hc.DEV_PAIRS[:2])
    stop = xc.cell_paths('voxelmorph', 'dev', pairs[0])['arm_stop']
    stop.parent.mkdir(parents=True)
    stop.write_text('{}')
    system = FakeSystem([])
    assert queue(['voxelmorph'], pairs, system) == 0 and system.runs == []


def test_the_real_system_runs_the_shim_and_the_chain_stops(monkeypatch):
    """systemd-run is the PATH shim (exit 99): no unit, no result line, the driver never ran: STOP before anything is recorded for the cell."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[0]

    class Shimmed(xb_chain.System):
        def gate(self):
            return dict(passed=True)

        def disk(self, need_d, need_home):
            return dict(passed=True, free_gib={})
    assert subprocess.run(['systemd-run', '--version'], capture_output=True).returncode == 99
    assert queue(['ants_s'], [pair], Shimmed()) == 20                                # m12: exit 99, no lock, no refusal record: STOP infrastructure
    assert not xc.cell_paths('ants_s', 'dev', pair)['result'].exists() and not xc.cell_paths('ants_s', 'dev', pair)['lock'].exists()


# ---- the regression tools -------------------------------------------------------------------------------------------------------------------------
def test_the_old_harness_differs_from_the_archive_by_three_path_constants(tmp_path):
    out = xb_regress.root()/'old_1'
    record = xb_regress.prepare_old(out)
    assert record['n_changed'] == 6 and all('# XB regression: path constant' in l for l in record['changed_lines'] if l.startswith('+'))
    assert {l.split('=')[0].strip('-+ ') for l in record['changed_lines']} == {'ROOT', 'STAGE', 'BIG'}
    text = (out/'harness/drive.py').read_text()
    assert "load_pair()" in text and 'def sanity' in text                  # the old driver otherwise unchanged
    for name in xb_regress.SCRIPTS:
        assert (out/'harness'/name).read_bytes() == (W.ROOT/xc.ARCHIVE_HARNESS/name).read_bytes()
    with pytest.raises(hc.Refused):
        xb_regress.prepare_old(tmp_path/'elsewhere')
    argv = xb_regress.old_argv('fireants_greedy', hc.DEV_PAIRS[0], out)
    assert '--unit=tag-hb-xr-old-fireants_greedy-0438-0439' in argv and argv[-2:] == [str(out/'harness/drive.py'), 'fireants_greedy'] and '--setenv=BL2_PAIR_SET=stage_L2R_val_v1' in argv
    with pytest.raises(hc.Refused):
        xb_regress.old_argv('fireants_greedy', hc.pair_name(*W.holdout_rows()[0]), out)        # development pairs only
    assert xb_regress.new_argv('ants_s', hc.DEV_PAIRS[0])[-7:] == ['run', '--arm', 'ants_s', '--pair-set', 'dev', '--pair', hc.DEV_PAIRS[0]]


def test_compare_reads_the_two_fields_only(tmp_path):
    out = xb_regress.root()/'old_2'
    big = out/'big'/'bl2val_ants_s_0438__0439_20261108T000000Z'
    big.mkdir(parents=True)
    d = np.random.default_rng(0).standard_normal((3,)+W.SHAPE).astype(np.float32)
    np.savez_compressed(big/'disp.npz', d=d)
    (out/'small'/'ants_s'/hc.DEV_PAIRS[0]).mkdir(parents=True)
    (out/'small'/'ants_s'/hc.DEV_PAIRS[0]/'result.json').write_text('the old record holds sanity Dice: never opened')
    new = tmp_path/'new.npz'
    np.savez_compressed(new, d=d)
    doc = xb_regress.compare('ants_s', hc.DEV_PAIRS[0], out, new_path=new)
    assert doc['bitwise_equal'] and doc['max_abs_diff_voxels'] == 0.
    np.savez_compressed(new, d=d+np.float32(.25))
    out3 = xb_regress.root()/'old_3'
    big3 = out3/'big'/'bl2val_ants_s_0438__0439_X'
    big3.mkdir(parents=True)
    np.savez_compressed(big3/'disp.npz', d=d)
    doc = xb_regress.compare('ants_s', hc.DEV_PAIRS[0], out3, new_path=new)
    assert not doc['bitwise_equal'] and abs(doc['max_abs_diff_voxels']-.25) < 1e-6
    assert doc['fixed_fg'] is None and doc['mean_norm_diff_fixed_fg_voxels'] is None and doc['repeat']['new'] is None and doc['repeat']['old']['sha256'] is None


def test_the_regression_plans_manual_gap_check_reads_the_gates_clock():
    """A5a diff review n4: the manual 10 s check before an old unit (regression.md section 2) compares the journal's __MONOTONIC_TIMESTAMP (microseconds) with time.monotonic() (seconds) -- the
    clock holdout_common.machine_gate compares with -- not with /proc/uptime (CLOCK_BOOTTIME, which counts the host's suspend and runs ahead after it).  The clocks: time.monotonic() is
    CLOCK_MONOTONIC, BOOTTIME is never behind it."""
    import time
    text = (W.ROOT/'runs'/'20261107-xb'/'regression.md').read_text(encoding='utf-8')
    section = text[text.index('## 2.'):text.index('## 3.')]
    assert '__MONOTONIC_TIMESTAMP 与 time.monotonic() 的差 ≥ 10 s' in section and '/ 1e6' in section and "time.monotonic():.6f" in section
    assert '__MONOTONIC_TIMESTAMP 与 `cat /proc/uptime` 的差' not in section and '不要**与 `cat /proc/uptime` 比' in section
    a = time.monotonic()
    b = time.clock_gettime(time.CLOCK_MONOTONIC)
    c = time.monotonic()
    mono = time.clock_gettime(time.CLOCK_MONOTONIC)
    boot = time.clock_gettime(time.CLOCK_BOOTTIME)                     # read after: never behind (it also counts suspend)
    assert a <= b <= c and boot >= mono


def test_compare_takes_the_foreground_difference_and_the_two_repeats(tmp_path):
    """A5a review I3: beside the largest difference, the norm of the difference averaged over fixed > 0 of the new cell's derived inputs (m_fireants.py's measure of its warm repeat), and the two
    method_out.json's repeat keys (only those; the old one beside the old disp.npz).  The old small result is never opened."""
    out = xb_regress.root()/'old_fg'
    big = out/'big'/'bl2val_fireants_greedy_0438__0439_20261108T000000Z'
    big.mkdir(parents=True)
    rng = np.random.default_rng(1)
    d = rng.standard_normal((3,)+W.SHAPE).astype(np.float32)
    np.savez_compressed(big/'disp.npz', d=d)
    (big/'method_out.json').write_text(json.dumps(dict(repeat_mean_norm_diff_fixed_fg_voxels=.01, repeat_max_abs_diff_voxels=.3, seconds_registration_cold=1.)))
    fixed = np.zeros(W.SHAPE, np.float32)
    fixed[2:8, 3:9, 2:7] = 1.
    inputs = tmp_path/'inputs.npz'
    np.savez_compressed(inputs, fixed=fixed, moving=fixed)
    shift = np.zeros_like(d)
    shift[0] = .5                                                      # a half-voxel shift along z: the norm .5 on every voxel
    new = tmp_path/'new.npz'
    np.savez_compressed(new, d=d+shift)
    method_out = tmp_path/'method_out.json'
    method_out.write_text(json.dumps(dict(repeat_mean_norm_diff_fixed_fg_voxels=.02, repeat_max_abs_diff_voxels=None)))
    small = out/'small'/'fireants_greedy'/hc.DEV_PAIRS[0]
    small.mkdir(parents=True)
    (small/'result.json').write_text('the old record holds sanity Dice: never opened')
    doc = xb_regress.compare('fireants_greedy', hc.DEV_PAIRS[0], out, new_path=new, fixed_npz=inputs, new_method_out=method_out)
    assert abs(doc['mean_norm_diff_fixed_fg_voxels']-.5) < 1e-6 and doc['fixed_fg']['n_voxels'] == int((fixed > 0).sum()) and doc['fixed_fg']['sha256'] == xc.sha256(inputs)
    assert doc['repeat']['old']['repeat_mean_norm_diff_fixed_fg_voxels'] == .01 and doc['repeat']['new']['repeat_mean_norm_diff_fixed_fg_voxels'] == .02
    assert doc['repeat']['new']['repeat_max_abs_diff_voxels'] is None and 'seconds_registration_cold' not in doc['repeat']['old'] and doc['repeat']['old']['sha256'] == xc.sha256(big/'method_out.json')
    assert json.loads((out/f'compare_fireants_greedy_{hc.DEV_PAIRS[0]}.json').read_text())['mean_norm_diff_fixed_fg_voxels'] == doc['mean_norm_diff_fixed_fg_voxels']


# ---- the limit experiment tool --------------------------------------------------------------------------------------------------------------------
def test_the_trials_keep_the_arms_start_string_and_lower_only_the_limits():
    out = xl.root()/'t'
    for trial, spec in xl.TRIALS.items():
        argv = xl.trial_argv(trial, 'fireants_greedy', out)
        cell = xb_chain.unit_argv('fireants_greedy', 'dev', hc.DEV_PAIRS[0])
        props = [a for a in argv if a.startswith('--property=')]
        assert len(props) == 4 and 'OOMPolicy=continue' in ' '.join(props)
        changed = {p for p in props} ^ {a for a in cell if a.startswith('--property=')}
        assert {c.split('=')[1] for c in changed} <= {'MemoryMax', 'MemorySwapMax', 'RuntimeMaxSec'}
        i = argv.index('--')
        assert argv[i+1] == str(xc.ENVS/'bl_fireants/bin/python') and argv[i+4].endswith('xb_synthetic_load.py')
        assert not any('xb_drive.py' in a for a in argv) and not any('pair.npz' in a for a in argv)
    assert '--property=MemoryMax=512M' in xl.trial_argv('mem_swap0', 'ants_s', out) and '--property=MemorySwapMax=256M' in xl.trial_argv('mem_swap256m', 'ants_s', out)
    assert '--property=RuntimeMaxSec=20' in xl.trial_argv('runtime', 'ants_s', out)
    assert xl.trial_argv('runtime', 'ants_s', out)[xl.trial_argv('runtime', 'ants_s', out).index('--')+1] == str(xc.PY)
    assert xl.out_problem(xc.REPO/'x') and xl.out_problem(xl.root()) and xl.out_problem(xl.root()/'new') is None


class TrialSystem(FakeSystem):
    def __init__(self, end, result, dmesg=''):
        super().__init__([], dmesg=dmesg)
        self.end, self.result = end, result

    def trial_gate(self, trial):
        return dict(passed=True)

    def run(self, argv, log_path):
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True)
        (unit_dir/'unit_start.json').write_text(json.dumps(dict(utc='2026-11-08T00:00:00Z')))
        if self.end is not None:
            (unit_dir/'unit_end.json').write_text(json.dumps(self.end))
        Path(log_path).write_text(f'Finished with result: {self.result}\n')
        return 0


@pytest.mark.parametrize('trial, end, result, passed', [
    ('mem_swap0', dict(utc='u', exit_code=-9, killed_by_supervisor=False, memory_events=dict(max=9, oom=1, oom_kill=1), dmesg_readable=True, oom_lines=[]), 'exit-code', True),
    ('mem_swap256m', dict(utc='u', exit_code=-9, killed_by_supervisor=False, memory_events=dict(max=9, oom=1, oom_kill=1), dmesg_readable=True, oom_lines=[]), 'exit-code', True),
    ('runtime', dict(utc='u', exit_code=-15, killed_by_supervisor=False, sigterm_received=True, memory_events=QUIET, dmesg_readable=True, oom_lines=[]), 'timeout', True),
    ('runtime', None, 'timeout', False),                                                                       # the supervisor wrote no end record
    ('mem_swap0', dict(utc='u', exit_code=0, killed_by_supervisor=False, memory_events=QUIET, dmesg_readable=True, oom_lines=[]), 'success', False),     # the limit was not reached
])
def test_a_trials_reading(trial, end, result, passed):
    doc = xl.run_trial(trial, 'ants_s', xl.root()/f'{trial}_{passed}_{result}', system=TrialSystem(end, result))
    assert doc['passed'] is passed and doc['systemd_result'] == result
    if passed:
        assert (doc['reading']['status'], doc['reading']['kind']) == tuple(xl.TRIALS[trial]['expected']['yi'])
        assert doc['supervisor']['exit_code'] == end['exit_code'] and doc['supervisor']['start_utc']
    assert json.loads((xl.root()/f'{trial}_{passed}_{result}'/'limit_result.json').read_text())['expected_both_rules']['jia']


# ---- the reviewed deletion step ------------------------------------------------------------------------------------------------------------------------
def test_collect_intermediates_and_the_deletion_step_refuses_outside_its_roots(monkeypatch, tmp_path):
    """The driver deleted nothing; collect-intermediates gathers the ok cells' manifests (checked against their records); delete_verified.sh deletes only under the real XB work roots, so in the
    sandbox every line is refused and every file stays; its log is never overwritten."""
    W.fake_checks(monkeypatch)
    W.write_stage_pair(hc.DEV_STAGE, W.DEV_PAIR)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps()) == 0
    approved = tmp_path/'approved.tsv'
    summary = xb_chain.collect_intermediates('dev', ['fireants_greedy'], approved)
    assert summary['n_cells'] == 1 and summary['n_files'] == 2 and summary['sha256'] == xc.sha256(approved)
    with pytest.raises(FileExistsError):
        xb_chain.collect_intermediates('dev', ['fireants_greedy'], approved)
    log = tmp_path/'deleted.tsv'
    done = subprocess.run(['bash', str(W.ROOT/'experiments/xb/delete_verified.sh'), str(approved), str(log)], capture_output=True, text=True)
    assert done.returncode == 0
    rows = log.read_text().splitlines()[1:]
    assert len(rows) == 2 and all(r.endswith('REFUSED_outside_root') and Path(r.split('\t')[0]).exists() for r in rows)
    assert subprocess.run(['bash', str(W.ROOT/'experiments/xb/delete_verified.sh'), str(approved), str(log)], capture_output=True, text=True).returncode == 2


def test_the_deletion_step_refuses_a_path_that_is_not_canonical(tmp_path):
    """m5: a path with a . or .. component or an empty one is refused before anything else (`.../work/../locks/x` would pass a prefix test); a path inside a root that realpath -e cannot
    resolve is missing.  Nothing under the real roots exists for this test, nothing is created or deleted."""
    work = '/home/rmcfz/seq_vp_tag_results/xb_dev_v1/work'
    rows = [f'{work}/../locks/x.lock', f'{work}/./a/inputs.npz', f'{work}//a/inputs.npz', f'{work}/a/..', f'{work}/xb_test_never_there_9f3c/inputs.npz']
    approved = tmp_path/'approved.tsv'
    approved.write_text(xc.MANIFEST_HEADER+''.join(f'{r}\t1\t{"0"*64}\n' for r in rows))
    log = tmp_path/'deleted.tsv'
    assert subprocess.run(['bash', str(W.ROOT/'experiments/xb/delete_verified.sh'), str(approved), str(log)], capture_output=True, text=True).returncode == 0
    results = [line.split('\t')[3] for line in log.read_text().splitlines()[1:]]
    assert results == ['REFUSED_not_canonical']*4+['MISSING_OR_LINK']


def test_the_deletion_step_takes_the_four_lpba40_external_arms_work_only(tmp_path):
    """LP-F (decision_log 2026-11-20, the lead's ruling 2 (乙) and the handbook review's follow-up): the LPBA40 formal root's work/<arm>/ of the four external arms are roots (a path there
    that is not on disk gets past the root test and reads MISSING_OR_LINK; a kept file's name reads REFUSED_kept_file, so the kept-file rule is unchanged); the whole work/, the method's
    and configuration C's cells, the feasibility root, the formal root's other directories and look-alike prefixes stay outside (REFUSED_outside_root); the two earlier roots unchanged.
    The roots are the arm table's: xb_core.LPBA_ARMS under holdout_common.work_dir's layout.  Paths that are never there: nothing is created, read or deleted under the real roots."""
    never = 'lpf_test_never_there_5d1e'
    T, R = '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1', '/home/rmcfz/seq_vp_tag_results'
    text = (W.ROOT/'experiments/xb/delete_verified.sh').read_text().splitlines()
    roots = text[9].split('"')[1].split()
    assert text[9].startswith('roots=') and roots == [f'{R}/holdout_B_v1/work/', f'{R}/xb_dev_v1/work/', *(f'{T}/work/{arm}/' for arm in xc.LPBA_ARMS)]
    assert all(Path(hc.work_dir(arm, 'P', Path(T))).parent == Path(f'{T}/work/{arm}') for arm in xc.LPBA_ARMS)
    rows = {**{f'{T}/work/{arm}/{never}/inputs.npz': 'MISSING_OR_LINK' for arm in xc.LPBA_ARMS},
            f'{T}/work/convexadam_v4seg/{never}/disp.npz': 'REFUSED_kept_file',
            f'{R}/holdout_B_v1/work/{never}/inputs.npz': 'MISSING_OR_LINK',
            f'{R}/xb_dev_v1/work/{never}/inputs.npz': 'MISSING_OR_LINK',
            f'{T}/work/method/{never}/t88/field_0600.npz': 'REFUSED_outside_root',
            f'{T}/work/configc/{never}/run/committed_0001.npz': 'REFUSED_outside_root',
            f'{T}/work/{never}.npz': 'REFUSED_outside_root',
            f'{T}/work/fireants_synx/{never}/inputs.npz': 'REFUSED_outside_root',
            f'{R}/lpba40_feas_v1/work/fireants_syn/{never}/inputs.npz': 'REFUSED_outside_root',
            f'{T}/stage/{never}/pair.npz': 'REFUSED_outside_root',
            f'{T}/fireants_syn/{never}/result.json': 'REFUSED_outside_root'}
    assert not any(Path(p).exists() for p in rows)
    approved = tmp_path/'approved.tsv'
    approved.write_text(xc.MANIFEST_HEADER+''.join(f'{r}\t1\t{"0"*64}\n' for r in rows))
    log = tmp_path/'deleted.tsv'
    assert subprocess.run(['bash', str(W.ROOT/'experiments/xb/delete_verified.sh'), str(approved), str(log)], capture_output=True, text=True).returncode == 0
    got = [line.split('\t') for line in log.read_text().splitlines()[1:]]
    assert {g[0]: g[3] for g in got} == rows
    assert not any(Path(p).exists() for p in rows)


def test_a_refusal_record_without_a_number_still_runs_again_under_its_rerun_authorization(monkeypatch):
    """A refusal record without seq (an older driver's): the copies are made, P20 cannot name the newest (no number) -- that only rules out P20 -- and the committed rerun authorization still lets
    the cell run (I1)."""
    W.fake_checks(monkeypatch)
    pair = hc.DEV_PAIRS[6]
    hc.write_refusal(hc.batch_root(xc.DEV_ROOT)/'xb_old', dict(tool='xb', arm='ants_s', pair=pair, pair_set='dev', stage='admission'))
    assert xb_chain.cell_state('ants_s', 'dev', pair)[0] == 'refused'
    W.authorize('rerun', 'ants_s', 'dev', pair)
    system = FakeSystem([dict(status='ok')])
    assert queue(['ants_s'], [pair], system) == 0 and len(system.runs) == 1
    assert len(hc.refusal_copies(xc.cell_paths('ants_s', 'dev', pair)['record'])) == 1
