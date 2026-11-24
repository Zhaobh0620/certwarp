"""HB item 5: the configuration C arm (preregistration §2.2; revision 1c item 1, rule (乙)): firewall_c's derived pair and gate (real), the frozen prepare and the frozen run replaced by writers of what
they leave (no frozen tree is copied or run), experiments/d0/classify_run.py unchanged (run as it is), D75's candidates certified again by a stand-in certificate; the step's evidence read by
holdout_common.classify_unit_end.  A development pair of synthetic arrays (the real ids and allowlist), and one hold-out pair on a simulated unseal branch."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import _world as W

he, hc, hb = W.he, W.hc, W.hb
import hb_configc as C  # noqa: E402

fc = C.fc
QUIET = dict(oom=0, oom_kill=0, max=0)


def fake_prepare(derived_dir, run_dir, *, pair, scratch, frozen=None, **kwargs):
    """firewall_c.prepare's stand-in: the run directory with the derived inputs in its slot and a protocol (no frozen tree copied, no prepare run)."""
    run_dir = Path(run_dir)
    slot = run_dir/'inputs'/fc.slot_name(pair)
    slot.mkdir(parents=True)
    for f in ('pair.npz', 'manifest.json'):
        shutil.copyfile(Path(derived_dir)/f, slot/f)
    (run_dir/'protocol.json').write_text(json.dumps(dict(task=dict(name='task'))))
    (run_dir/'snapshot').mkdir()
    return dict(slot=f'test slot {slot}', frozen_files=0, seconds=0.)


class Launcher:
    """The frozen run's stand-in in the run directory: 'ok' (a native commit: final_native.npz of a small field, intermediates), 'timeout' (the watchdog: exit 124 after 300 s), 'none' (no native
    result: method_failure)."""
    kind = 'direct'

    def __init__(self, mode='ok'):
        self.mode = mode

    def __call__(self, run, short):
        run = Path(run)
        task = run/'task'
        task.mkdir()
        g = W.grid()
        stamps = ('2026-11-09T00:00:00Z', '2026-11-09T00:05:16Z' if self.mode == 'timeout' else '2026-11-09T00:01:30Z')
        (run/'supervisor_started_utc.txt').write_text(stamps[0]+'\n')
        (run/'supervisor_finished_utc.txt').write_text(stamps[1]+'\n')
        (run/'started.json').write_text('{}')
        (run/'run.log').write_text('test\n')
        if self.mode == 'timeout':
            (run/'exit_code').write_text('124\n')
            return dict(kind=self.kind, returncode=124)
        u = W.field(g, scale=.1, seed=9)
        np.savez(task/'final_native.npz', u=u)
        np.savez(task/'committed_0001.npz', u=u)
        np.savez(task/'level_2_commit.npz', u=u)
        (task/'final_native.json').write_text(json.dumps(dict(sha256=hb.sha(task/'final_native.npz'), role='native commit (test)')))
        role = 'formal_native_outer_commit' if self.mode == 'ok' else 'no_native_result'
        (task/'summary.json').write_text(json.dumps(dict(status='native_budget_exhausted' if self.mode == 'ok' else 'no_certified_map', reason='test', metrics_role=role, native_commits=1)))
        (task/'worker_completion.json').write_text('{}')
        (run/'completion.json').write_text('{}')
        (run/'exit_code').write_text('0\n')
        return dict(kind=self.kind, returncode=0)


def probe_of(after_events=QUIET):
    seq = [dict(events=QUIET, cgroup='/user.slice/tag-hb-c-0438-0439.service'), dict(events=after_events, cgroup='/user.slice/tag-hb-c-0438-0439.service')]
    return lambda: seq.pop(0)


def kernel(readable=True):
    return (lambda t0: ('', [], 0)) if readable else (lambda t0: (None, None, None))


@pytest.fixture
def dev(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    pair = hc.DEV_PAIRS[0]
    W.XW.write_stage_pair(hc.DEV_STAGE, pair, seg=True)
    batch = C.roots('dev')[0]/'batches'/'test'                     # a queue's batch is under the record root, where the refusal records are looked for (XB's I1)
    batch.mkdir(parents=True)
    return pair, batch


def result(pair_set, pair):
    return he.read_json(C.cell_paths(pair_set, pair)['result'])


def test_ok_takes_final_native_certified_again_and_deletes_the_intermediates(dev):
    pair, batch = dev
    certify = W.Certify()
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=certify, probe=probe_of(), kernel=kernel()) == 0
    r = result('dev', pair)
    paths = C.cell_paths('dev', pair)
    assert r['status'] == 'ok' and r['classification']['class'] == 'ok' and r['candidates'][0]['file'] == 'final_native.npz' and r['candidates'][0]['passed'] is True
    assert r['firewall']['gate']['passed'] is True and r['firewall']['seg_keys_zero'] == list(fc.SEG_KEYS) and r['firewall']['source_keys_read'] == ['fixed', 'moving', 'initial_u']
    assert (paths['work']/'output'/'final.npz').is_file() and he.read_json(paths['work']/'output'/'line.json')['line_status'] == 'configc_certified' and paths['lock'].is_file()
    assert r['intermediates']['n_deleted'] == 2 and not list((paths['work']/'run'/'task').glob('committed_*.npz')) and (paths['work']/'run'/'task'/'final_native.npz').is_file()
    assert r['launch']['step_reading']['status'] is None and len(certify.calls) == 1
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=certify, probe=probe_of(), kernel=kernel()) == C.EXIT['refused']      # once: refused before the lock
    assert [json.loads(p.read_text())['stage'] for p in (batch/'refusals').glob('*.json')] == ['once']


@pytest.mark.parametrize('case, status', [('timeout', 'unit_limit'), ('none', 'method_failure'), ('uncertified', 'method_failure'), ('oom', 'infra_failure'), ('unreadable', 'needs_lead_rejudgment'),
                                          ('wrapper', 'wrapper_error')])
def test_the_failures_by_class(dev, monkeypatch, case, status):
    pair, batch = dev
    if case == 'wrapper':
        monkeypatch.setattr(fc, 'prepare', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('test: our code raised')))
    certify = W.Certify(fail=set()) if case != 'uncertified' else None
    if case == 'uncertified':
        class FailAll(W.Certify):
            def __call__(self, u, g):
                self.calls.append(1)
                return W.result_of(u, g, passed=False)
        certify = FailAll()
    rc = C.run('dev', pair, batch=batch, launcher=Launcher('timeout' if case == 'timeout' else 'none' if case == 'none' else 'ok'), certify=certify,
               probe=probe_of(dict(oom=1, oom_kill=1) if case == 'oom' else QUIET), kernel=kernel(case != 'unreadable'))
    r = result('dev', pair)
    assert r['status'] == status and rc == C.EXIT[status], r
    keep = C.cell_paths('dev', pair)['work']/'run'/'task'
    if case in ('timeout', 'none', 'uncertified', 'oom', 'unreadable') and keep.exists() and case != 'timeout':
        assert list(keep.glob('committed_*.npz')) or case == 'none'                                     # a failure keeps every file (prereg §5.3)
    if case == 'timeout':
        assert r['kind'] == 'configc_timeout' and r['classification']['kind'] == 'timeout'
    if case == 'wrapper':
        assert 'our code raised' in r['error'] and C.cell_paths('dev', pair)['lock'].is_file()


def test_a_pair_that_is_not_one_of_the_cells_is_refused_before_the_lock(dev):
    pair, batch = dev
    other = hc.pair_name(*W.XW.holdout_rows()[0])
    assert C.run('dev', other, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    assert not C.cell_paths('dev', other)['lock'].exists() and [json.loads(p.read_text())['stage'] for p in (batch/'refusals').glob('*.json')] == ['admission']


def test_a_hold_out_pair_is_admitted_through_the_unseal_object_and_moved_to_d(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='c', clean=True))
    monkeypatch.setattr(W.hb_method, 'PREREG_SHA256', hb.sha(hb.REPO/W.hb_method.PREREG_REL))
    pair = W.first_pair()
    W.XW.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    W.XW.simulated_unseal(tmp_path, monkeypatch)
    batch = C.roots('holdout')[0]/'batches'/'test'
    batch.mkdir(parents=True)
    assert C.run('holdout', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == 0
    r = result('holdout', pair)
    paths = C.cell_paths('holdout', pair)
    assert r['status'] == 'ok' and r['admission']['unseal'] and r['move']['passed'] is True and (paths['store']/'output'/'disp.npz').is_file() and not paths['work'].exists()
    assert r['output_store']['disp']['file'] == str(paths['store']/'output'/'disp.npz') and r['firewall']['source_keys_read'] == ['fixed', 'moving', 'initial_u']


def test_the_queue_overrides_a_record_the_units_evidence_contradicts(tmp_path, monkeypatch, dev):
    """xb_chain's I2 for this arm: the driver wrote ok, the unit's evidence (OOM of the unit) says infra_failure: overridden.json supersedes the record (kept), the batch stops."""
    pair, batch = dev

    class System(hb.System):
        def run(self, argv, log_path):
            unit_dir = Path(argv[argv.index('--record-dir')+1])
            unit_dir.mkdir(parents=True, exist_ok=True)
            (unit_dir/'unit_start.json').write_text('{}')
            C.run('dev', pair, batch=Path(argv[argv.index('--batch')+1]), launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())
            (unit_dir/'unit_end.json').write_text(json.dumps(dict(exit_code=0, memory_events=dict(oom=0, oom_kill=1), cgroup=None, dmesg_readable=True, oom_lines=[], dxg_lines=0)))
            Path(log_path).write_text('Finished with result: success\n')
            return 0

        def uptime(self):
            return 1.

        def dmesg(self):
            return ''

        def journal(self, unit, since):
            return ''

        def gate(self, kind):
            return dict(passed=True)

        def disk(self, need_d, need_home):
            return dict(passed=True)
    assert C.queue('dev', pairs=[pair], system=System(), echo=lambda m: None) == hb.EXIT['infra']
    paths = C.cell_paths('dev', pair)
    over = he.read_json(paths['overridden'])
    assert over['status'] == 'infra_failure' and over['overrides']['status'] == 'ok' and he.read_json(paths['result'])['status'] == 'ok' and C.state('dev', pair) == 'stopped'


def test_a_cell_refused_before_its_lock_runs_again_only_with_a_committed_rerun_authorization(dev, monkeypatch):
    """XB's I1 for this arm (C23.12-6, D107): the frozen tree's check refuses the first attempt (a refusal record, no lock); the next attempt is refused (stage rerun) until an authorization listing that
    record is committed; then the cell runs."""
    pair, batch = dev
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), ['test: the frozen tree differs']))
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    stages = sorted(json.loads(p.read_text())['stage'] for p in (batch/'refusals').glob('*.json'))
    assert stages == ['frozen', 'rerun'] and not C.cell_paths('dev', pair)['lock'].exists()
    root = C.roots('dev')[0]
    prior = hc.cell_refusals(root, arm=C.ARM, pair=pair, pair_set='dev')
    repo = W.XW.auth_repo()
    directory = hc.authorization_dir('dev')
    doc = dict(schema='bayesreg.hb.rerun.v1', written_by="lead on BZ's behalf", words='（测试）BZ：可以重跑', time_local='2026-11-09 8:00 PM CST', utc='2026-11-09T14:00:00Z', reason='test', tool='hb', arm=C.ARM,
               pair=pair, pair_set='dev', refusals=[dict(path=r['path'], sha256=r['sha256']) for r in prior])
    W.XW.write_json(directory/'dev_hb_rerun_1.json', doc)
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', 'rerun')
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == 0
    assert result('dev', pair)['rerun_authorization']['path'].endswith('dev_hb_rerun_1.json')


def commit_dev_authorization(kind, pair, **fields):
    repo = W.XW.auth_repo()
    directory = hc.authorization_dir('dev')
    k = 1
    while (directory/f'dev_hb_{kind}_{k}.json').exists():
        k += 1
    doc = dict(schema=f'bayesreg.hb.{kind}.v1', written_by="lead on BZ's behalf", words='（测试）BZ：可以', time_local='2026-11-09 9:00 PM CST', utc='2026-11-09T15:00:00Z', reason='test', tool='hb',
               arm=C.ARM, pair=pair, pair_set='dev')
    doc.update(fields)
    path = W.XW.write_json(directory/f'dev_hb_{kind}_{k}.json', doc)
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', f'{kind} {k}')
    return str(path.relative_to(hc.AUTH_ROOT))


@pytest.mark.parametrize('again', ['ok', 'wrapper', 'oom'])
def test_a_resume_is_one_chain_with_the_first_run_and_a_second_failure_is_written(dev, monkeypatch, again):
    """Prereg §5.4, runs/20261109-align item 3: an infrastructure failure stops the cell; one resume by a committed resume record (its own lock, result_resume.json, work .resume); a resumed run that
    fails again (our wrapper, the infrastructure) is written -- no output, no third run, the batch goes on at the next start."""
    pair, batch = dev
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(dict(oom=1, oom_kill=1)), kernel=kernel()) == C.EXIT['infra_failure']
    paths = C.cell_paths('dev', pair)
    assert C.state('dev', pair) == 'stopped'
    rel = commit_dev_authorization('resume', pair, lock_sha256=hb.sha(paths['lock']), previous_record=str(paths['result']), previous_record_sha256=hb.sha(paths['result']))
    if again == 'wrapper':
        monkeypatch.setattr(fc, 'prepare', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('test: our code raised again')))
    rc = C.run('dev', pair, batch=batch, resume_rel=rel, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(dict(oom=1, oom_kill=1) if again == 'oom' else QUIET), kernel=kernel())
    resumed = he.read_json(paths['record']/'result_resume.json')
    want = dict(ok='ok', wrapper='wrapper_error', oom='infra_failure')[again]
    assert rc == C.EXIT[want] and resumed['status'] == want and resumed['resume']['path'].endswith(Path(rel).name) and hc.resume_lock_path(paths['lock']).is_file()
    assert result('dev', pair)['status'] == 'infra_failure' and C.effective(paths) == resumed and C.chain(paths)[1] is True          # the first record kept; the chain's record is the resume's
    assert C.state('dev', pair) == 'done'                                                   # ok continues; a second failure is written (no output, no third run)
    assert C.run('dev', pair, batch=batch, resume_rel=rel, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']          # at most once


def test_bz_declining_a_resume_writes_the_stopped_cell(dev):
    """Prereg revision 1c item 4: BZ's committed 未批补跑 (no_resume) record of a stopped cell, its record the first run's result.json with that sha256: the cell is written (no output), the batch goes on."""
    pair, batch = dev
    C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(dict(oom=1, oom_kill=1)), kernel=kernel())
    paths = C.cell_paths('dev', pair)
    assert C.state('dev', pair) == 'stopped' and C.declined('dev', pair) is None
    commit_dev_authorization('no_resume', pair, record=str(paths['overridden']), record_sha256='0'*64, note=hc.NO_RESUME_NOTE)           # not the cell's record: not taken
    assert C.state('dev', pair) == 'stopped'
    rel = commit_dev_authorization('no_resume', pair, record=str(paths['result']), record_sha256=hb.sha(paths['result']), note=hc.NO_RESUME_NOTE)
    assert C.declined('dev', pair).path.endswith(Path(rel).name) and C.state('dev', pair) == 'done'


# ---- HB review I3, I4, m12 (i) ------------------------------------------------------------------------------------------------------------------------
class InProcess(hb.System):
    """The configuration C unit played in this process: the driver's run() with the stand-ins; the unit's own records."""

    def __init__(self, pair, launcher='ok', events=QUIET, frozen_problem=None, monkeypatch=None):
        self.pair, self.launcher, self.events, self.runs = pair, launcher, events, []
        self.frozen_problem, self.monkeypatch = frozen_problem, monkeypatch

    def run(self, argv, log_path):
        self.runs.append(argv)
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text('{}')
        inner = argv[argv.index('--')+1:]
        resume = inner[inner.index('--resume')+1] if '--resume' in inner else None
        if self.frozen_problem is not None:
            self.monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), [self.frozen_problem]))
        C.run('dev', self.pair, batch=Path(inner[inner.index('--batch')+1]), resume_rel=resume, launcher=Launcher(self.launcher), certify=W.Certify(), probe=probe_of(self.events), kernel=kernel())
        (unit_dir/'unit_end.json').write_text(json.dumps(dict(exit_code=0, memory_events=self.events, cgroup=None, dmesg_readable=True, oom_lines=[], dxg_lines=0)))
        Path(log_path).write_text('Finished with result: success\n')
        return 0

    def uptime(self):
        return 1.

    def dmesg(self):
        return ''

    def journal(self, unit, since):
        return ''

    def gate(self, kind):
        return dict(passed=True)

    def disk(self, need_d, need_home):
        return dict(passed=True)


def test_the_queue_checks_a_resume_record_before_the_unit(dev, monkeypatch):
    """HB review I4 (as hb_method.queue): a resume record that does not hold (here: another lock's sha256) is refused before any unit starts; nothing is written in the cell."""
    pair, batch = dev
    C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(dict(oom=1, oom_kill=1)), kernel=kernel())
    paths = C.cell_paths('dev', pair)
    bad = commit_dev_authorization('resume', pair, lock_sha256='0'*64, previous_record=str(paths['result']), previous_record_sha256=hb.sha(paths['result']))
    system = InProcess(pair)
    assert C.queue('dev', pairs=[pair], resumes={pair: bad}, system=system, echo=lambda m: None) == hb.EXIT['refused'] and system.runs == []
    assert not (paths['record']/'result_resume.json').exists() and C.state('dev', pair) == 'stopped'


def test_a_resume_refused_before_its_own_lock_writes_no_record_and_a_good_one_runs_after(dev, monkeypatch):
    """HB review I4: the resumed driver refused before its .resume lock (the frozen tree's check, in the unit): no result_resume.json is written for it; the cell stays stopped and resumable."""
    pair, batch = dev
    C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(dict(oom=1, oom_kill=1)), kernel=kernel())
    paths = C.cell_paths('dev', pair)
    rel = commit_dev_authorization('resume', pair, lock_sha256=hb.sha(paths['lock']), previous_record=str(paths['result']), previous_record_sha256=hb.sha(paths['result']))
    system = InProcess(pair, frozen_problem='test: the frozen tree differs', monkeypatch=monkeypatch)
    assert C.queue('dev', pairs=[pair], resumes={pair: rel}, system=system, echo=lambda m: None) == hb.EXIT['refused'] and len(system.runs) == 1
    assert not (paths['record']/'result_resume.json').exists() and not hc.resume_lock_path(paths['lock']).exists() and C.state('dev', pair) == 'stopped'
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    root = C.roots('dev')[0]                            # XB's I1 for the resume's refusal before its lock: a committed rerun authorization listing the refusal record
    prior = hc.cell_refusals(root, arm=C.ARM, pair=pair, pair_set='dev')
    commit_dev_authorization('rerun', pair, refusals=[dict(path=r['path'], sha256=r['sha256']) for r in prior])
    assert C.queue('dev', pairs=[pair], resumes={pair: rel}, system=InProcess(pair), echo=lambda m: None) == 0
    assert he.read_json(paths['record']/'result_resume.json')['status'] == 'ok' and C.state('dev', pair) == 'done'


def test_a_destination_with_files_moves_nothing_and_keeps_the_work_directory(tmp_path, monkeypatch):
    """HB review I3 for the configuration C arm: the D: store holds a file already: nothing moved, the work directory whole on /home, the cell infra_failure (move_destination_exists)."""
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='c', clean=True))
    monkeypatch.setattr(W.hb_method, 'PREREG_SHA256', hb.sha(hb.REPO/W.hb_method.PREREG_REL))
    pair = W.first_pair()
    W.XW.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    W.XW.simulated_unseal(tmp_path, monkeypatch)
    batch = C.roots('holdout')[0]/'batches'/'test'
    batch.mkdir(parents=True)
    paths = C.cell_paths('holdout', pair)
    paths['store'].mkdir(parents=True)
    (paths['store']/'left_over.txt').write_text('x')
    assert C.run('holdout', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['infra_failure']
    r = result('holdout', pair)
    assert r['status'] == 'infra_failure' and r['kind'] == 'move_destination_exists' and 'move' not in r and (paths['work']/'output'/'final.npz').is_file()
    assert sorted(q.name for q in paths['store'].iterdir()) == ['left_over.txt']


def test_a_hold_out_stage_whose_manifest_names_another_pair_is_refused_before_the_lock(tmp_path, monkeypatch):
    """HB review m12 (i): as he_holdout.admitted_images, the stage's manifest must name the pair (fixed <- moving)."""
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(hb, 'code_record', lambda commit=None, run=None: dict(commit='c', clean=True))
    monkeypatch.setattr(W.hb_method, 'PREREG_SHA256', hb.sha(hb.REPO/W.hb_method.PREREG_REL))
    pair = W.first_pair()
    W.XW.write_stage_pair(hc.STAGE_ROOT, pair, seg=False)
    manifest = Path(hc.stage_pair_path(pair)).parent/'manifest.json'
    doc = json.loads(manifest.read_text())
    swapped = json.dumps(doc).replace(pair.split('__')[0], 'TMPX').replace(pair.split('__')[1], pair.split('__')[0]).replace('TMPX', pair.split('__')[1])
    manifest.write_text(swapped)
    W.XW.simulated_unseal(tmp_path, monkeypatch)
    batch = C.roots('holdout')[0]/'batches'/'test'
    batch.mkdir(parents=True)
    assert C.run('holdout', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    assert not C.cell_paths('holdout', pair)['lock'].exists() and [json.loads(q.read_text())['stage'] for q in (batch/'refusals').glob('*.json')] == ['manifest']


@pytest.mark.parametrize('case, done', [('unit_limit', True), ('no_output', True), ('infrastructure', False), ('lead_no_output', False), ('changed_record', False), ('not_committed', False),
                                        ('resumed_infrastructure', True)])
def test_a_cell_stopped_for_the_lead_is_written_by_a_committed_rejudgment(dev, monkeypatch, case, done):
    """The lead's item 13 for the configuration C arm (prereg revision 2 item 18; NI2, NI3): a cell in needs_lead_rejudgment (its evidence unreadable) is written, the batch going past it, by a committed
    rejudgment of its effective record: unit_limit (the lead may write it) or no_output (BZ's words); infrastructure only classifies (a resume record still); the lead writing no_output, a record
    changed after the judgment, or a rejudgment not committed: not taken."""
    pair, batch = dev
    assert C.run('dev', pair, batch=batch, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel(False)) == C.EXIT['needs_lead_rejudgment']
    paths = C.cell_paths('dev', pair)
    assert C.state('dev', pair) == 'stopped' and C.effective_path(paths) == paths['result']
    repo = W.XW.auth_repo()
    if case == 'resumed_infrastructure':               # N2 (i): resumed (by holdout_common's resume record), unreadable again, judged infrastructure: written, no third run
        rel = commit_dev_authorization('resume', pair, lock_sha256=hb.sha(paths['lock']), previous_record=str(paths['result']), previous_record_sha256=hb.sha(paths['result']))
        assert C.run('dev', pair, batch=batch, resume_rel=rel, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel(False)) == C.EXIT['needs_lead_rejudgment']
        assert C.state('dev', pair) == 'stopped' and C.effective_path(paths) == paths['record']/'result_resume.json'
    judged = C.effective_path(paths)
    judgment = dict(unit_limit='unit_limit', no_output='no_output', infrastructure='infrastructure', lead_no_output='no_output', changed_record='unit_limit', not_committed='unit_limit',
                    resumed_infrastructure='infrastructure')[case]
    writer = "lead on BZ's behalf" if case == 'no_output' else 'lead'
    doc = dict(schema=hc.REJUDGMENT_SCHEMA, arm=C.ARM, pair=pair, pair_set='dev', status='needs_lead_rejudgment', record=dict(path=str(judged), sha256=hb.sha(judged)),
               evidence=[], judgment=judgment, reason='test: dmesg unreadable (revision 1c item 1)', written_by=writer, words='（测试）BZ：不再跑' if case == 'no_output' else '',
               time_local='2026-11-10 2:30 AM CST', utc='2026-11-10T08:30:00Z')
    W.XW.write_json(Path(hc.AUTH_ROOT)/hc.REJUDGMENT_DIR_REL/C.ARM/f'{pair}_1.json', doc)
    if case != 'not_committed':
        W.git(repo, 'add', '-A')
        W.git(repo, 'commit', '-q', '-m', 'rejudgment')
    if case == 'changed_record':
        paths['result'].chmod(0o644)
        paths['result'].write_text(paths['result'].read_text().replace('needs_lead_rejudgment', 'needs_lead_rejudgment ', 1))
    assert (C.state('dev', pair) == 'done') is done


def test_a_cell_refused_again_after_its_rerun_is_written_by_bz_s_no_resume_and_its_refusals_are_kept(dev, monkeypatch):
    """The lead's item 15 (decision_log 2026-11-10 ~2:52, ~2:53 AM CST): refused before its lock (P20), run again by a committed rerun authorization and refused again: the refusal records are copied
    into the cell's record directory (refusals/, with their sha256); BZ's committed no_resume record whose record is the newest refusal makes the cell written (no output) and the batch goes past it; a
    no_resume record of an older refusal does not."""
    pair, batch = dev
    system = InProcess(pair, frozen_problem='test: the frozen tree differs', monkeypatch=monkeypatch)
    assert C.queue('dev', pairs=[pair], system=system, echo=lambda m: None) == hb.EXIT['refused']
    paths = C.cell_paths('dev', pair)
    first = sorted((paths['record']/'refusals').glob('*.json'))
    assert len(first) == 1 and (first[0].with_name(first[0].name+'.sha256')).read_text().split()[0] == hb.sha(first[0]) and C.state('dev', pair) == 'not_run'
    root = C.roots('dev')[0]
    prior = hc.cell_refusals(root, arm=C.ARM, pair=pair, pair_set='dev')
    commit_dev_authorization('rerun', pair, refusals=[dict(path=r['path'], sha256=r['sha256']) for r in prior])
    assert C.queue('dev', pairs=[pair], system=InProcess(pair, frozen_problem='test: still differs', monkeypatch=monkeypatch), echo=lambda m: None) == hb.EXIT['refused']
    copies = [q for _, q in C.refusals_by_seq((paths['record']/'refusals').glob('*.json'))]          # HB diff review n4: by the records' sequence numbers
    assert len(copies) == 2 and not paths['lock'].exists() and C.state('dev', pair) == 'not_run'
    older, newest = copies
    assert [he.read_json(q)['seq'] for q in copies] == [1, 2] and [r['seq'] for r in C.keep_refusals('dev', pair)] == [1, 2]
    stamp = newest.stat().st_mtime_ns
    os.utime(older, ns=(stamp+10**9, stamp+10**9))      # the older copy touched later: the newest is still the one numbered 2 (never mtime)
    os.utime(Path(hc.cell_refusals(C.roots('dev')[0], arm=C.ARM, pair=pair, pair_set='dev', counted=False)[0]['path']), ns=(stamp+10**9, stamp+10**9))
    commit_dev_authorization('no_resume', pair, record=str(older), record_sha256=hb.sha(older), note=hc.NO_RESUME_NOTE)     # not the newest: not taken
    assert C.declined_refusal('dev', pair) is None and C.state('dev', pair) == 'not_run'
    commit_dev_authorization('no_resume', pair, record=str(newest), record_sha256=hb.sha(newest), note=hc.NO_RESUME_NOTE)       # A5b review I2: its reason does not name P20
    assert C.declined_refusal('dev', pair) is None and C.state('dev', pair) == 'not_run'
    rel = commit_dev_authorization('no_resume', pair, record=str(newest), record_sha256=hb.sha(newest), note=hc.NO_RESUME_NOTE, reason='P20：拒了两次，BZ 不再跑')
    assert C.declined_refusal('dev', pair).path.endswith(Path(rel).name) and C.state('dev', pair) == 'done'
    system = InProcess(pair)
    assert C.queue('dev', pairs=[pair], system=system, echo=lambda m: None) == 0 and system.runs == []
