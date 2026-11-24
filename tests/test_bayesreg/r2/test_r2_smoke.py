"""R2's GPU smoke before the training (experiments/r2/smoke_r2.py; review I3, the lead's ruling of 11-10 ~10:59 PM CST).  CPU, tmp_path, synthetic subjects; no GPU, no real label.

- the smoke's config is r2s's with a short schedule and every step logged; its training command writes to another root, never the chain's;
- the start string; the measurement from a train log; the rule in its order (finite loss, the steps run, the projected wall per step <= .30 s, the GPU allocated peak <= 18 GiB, the synthetic
  chain), the limits included as passes;
- end to end on synthetic subjects (CPU, a tiny config): the training runs through cli with --profile r2_small, smoke_r2.json is written once with the weights listed, no read lock, nothing in the
  chain's root; on the CPU there is no GPU peak, so the reading is 'stop' for that reason alone.
- R2 follow-up (the diff review's new minors 1-3, 11-11): the rule checks the recipe r2_small, targets.json's T and AMP on; a synthetic chain that raises is a synthetic problem and smoke_r2.json is
  still written ('stop'); the dry run names no limit the synthetic part does not have; run_r2.prepare refuses before any snapshot unless smoke_r2.json reads 'pass', its targets_sha256 is targets.json's
  and the smoke's snapshot is at the training snapshot's commit (smoke_gate, on a tiny git repository).
- R2 follow-up 3 (the diff review 2, 0/0/6): smoke_r2.json carries the code record of the snapshot the unit ran from and smoke_gate checks it (1(b)); the training snapshot must hold the smoke
  snapshot's files, not only its commit label (3); a broken smoke_r2.json or SNAPSHOT.json is a problem, a Stop, not a traceback (2); the synthetic failure keeps its traceback's last lines (4);
  KeyboardInterrupt goes through, the log assertion is a real one, the positive of an existing training snapshot, prepare's second check after the snapshot (5); the chain checks the smoke (6);
  smoke_r2.prepare refuses a smoke root that ran (1(a)); the dry runs check that the snapshot path table carries no sealed list (contract C23.7-4 revision 6n rule 4).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r2', ROOT/'experiments'/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as rc  # noqa: E402
import r2_lock  # noqa: E402
import run_r2  # noqa: E402
import smoke_r2  # noqa: E402

T = [4, 10, 18, 19, 23, 29, 34, 35]


def test_the_smoke_config_is_r2s_with_a_short_schedule():
    config = smoke_r2.smoke_config(T)
    want = rc.train_config('r2s', T)
    assert {k: v for k, v in config.items() if k not in ('max_iters', 'min_iters', 'patience', 'eval_every', 'log_every', 'max_wall_seconds')} == {k: v for k, v in want.items() if k not in ('max_iters', 'min_iters', 'patience', 'eval_every', 'max_wall_seconds')}
    assert config['max_iters'] == config['min_iters'] == config['patience'] == smoke_r2.SMOKE_STEPS == 400 and config['log_every'] == 1 and config['checkpoint'] == 'last'


def test_the_training_command_writes_to_another_root():
    (name, argv), = smoke_r2.steps(rc.SMOKE, T)
    assert name == 'train' and argv[argv.index('--out')+1] == str(rc.SMOKE/'member') and argv[argv.index('--profile')+1] == 'r2_small' and argv[argv.index('--seed')+1] == '2'
    assert not [a for a in argv if str(rc.OUT) in a] and 'max_iters=400' in argv and '--bz-go' not in argv
    launch = smoke_r2.launch_argv()
    assert f'--unit={smoke_r2.UNIT}' in launch and f'--working-directory={rc.SMOKE}/snapshot' in launch and 'pytest' not in ' '.join(launch) and launch[-3:] == ['run', '--out', str(rc.SMOKE)]


def write_log(directory, *, step=.25, evaluation=2., data=30., loss=1., peak=10., stopped='max_iters', steps=400, nonfinite=False, recipe='r2_small', targets=T, amp=True):
    directory.mkdir(parents=True)
    rows = [dict(event='start', data_ready_seconds=data)]
    rows += [dict(event='train', it=i, loss=loss, ce=.5, soft_dice=.4, target_soft_dice=.3, step_seconds=(9. if i < 10 else step), gpu_peak_allocated_gib=peak) for i in range(steps)]
    rows += [dict(event='eval', it=k, eval_seconds=evaluation) for k in (200, 400)]
    if nonfinite:
        rows.append(dict(event='nonfinite_loss', it=3))
    (directory/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (directory/'train_record.json').write_text(json.dumps(dict(stopped=stopped, iterations_run=steps, recipe=recipe, config=dict(target_labels=targets, amp=amp), memory=dict(gpu_peak_allocated_gib=peak))))
    return directory


def test_the_measurement(tmp_path):
    measured = smoke_r2.measure(write_log(tmp_path/'m'))
    assert measured['finite'] and measured['step_seconds_median'] == .25 and measured['eval_seconds_median'] == 2. and measured['gpu_peak_allocated_gib'] == 10.
    assert measured['projected_wall_per_step'] == pytest.approx((30.+40000*.25+80*2.)/40000) and measured['recipe'] == 'r2_small' and measured['targets'] == T


@pytest.mark.parametrize('kwargs, synthetic, word', [
    (dict(), [], None), (dict(step=.30-(30.+80*2.)/40000), [], None), (dict(step=.30), [], 'projected'), (dict(peak=18.), [], None), (dict(peak=18.1), [], 'GPU'),
    (dict(loss=float('nan')), [], 'finite'), (dict(nonfinite=True), [], 'finite'), (dict(stopped='wall_budget'), [], 'stopped'), (dict(steps=300), [], 'stopped'), (dict(), ['x'], 'synthetic'),
    (dict(recipe='v3_large'), [], 'recipe'), (dict(recipe=None), [], 'recipe'), (dict(targets=[4, 10, 18, 19, 23, 29, 34]), [], 'target_labels'), (dict(targets=None), [], 'target_labels'),
    (dict(amp=False), [], 'AMP'), (dict(amp=None), [], 'AMP')])
def test_the_rule(tmp_path, kwargs, synthetic, word):
    reading, why = smoke_r2.reading_of(smoke_r2.measure(write_log(tmp_path/'m', **kwargs)), synthetic, targets=T)
    assert reading == ('pass' if word is None else 'stop') and (word is None or any(word in w for w in why)), why


def test_the_smoke_end_to_end_on_synthetic_subjects(tmp_path, monkeypatch):
    import synthetic_tree
    work = tmp_path/'world'
    synthetic_tree.build(work)
    targets = work/'targets.json'
    targets.write_text(json.dumps(dict(targets=T)))
    inputs = ['--synthetic', '--folds', str(work/'folds_full_v2.json'), '--allowlist', str(work/'dev_subjects.json'), '--training-source', synthetic_tree.SOURCE, '--expect-counts', '5,1', '--data-root', str(work/'data')]
    tiny = dict(run_r2.TINY_UF, max_iters=12, min_iters=12, patience=12, eval_every=6, log_every=1, amp=True)  # past the first 10 steps, which the median leaves out; AMP as planned
                                                                                                                # (the record says on; the CPU does not autocast)
    monkeypatch.setattr(run_r2, 'synthetic_chain', lambda work, env, device='cpu': [])          # the synthetic chain has its own test (test_r2_seg_level)
    monkeypatch.setattr(r2_lock, 'DEV_READ_LOCK', tmp_path/'never.lock')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    out = tmp_path/'smoke'
    document = smoke_r2.run(out, gate=False, python=sys.executable, device='cpu', inputs=inputs, targets_path=targets, synthetic_device='cpu', env=env, config=tiny)
    assert document['measured']['recipe'] == 'r2_small' and document['measured']['finite'] and document['measured']['iterations_run'] == 12 and document['measured']['targets'] == T
    assert document['reading'] == 'stop' and len(document['why']) == 1 and 'GPU' in document['why'][0]           # the CPU has no GPU peak
    assert set(document['weights_to_delete_after_the_reading']) >= {'model_best.pt', 'model_last.pt'} and document['reads_development_truth'] is False
    assert json.loads((out/'smoke_r2.json').read_text())['reading'] == 'stop' and not (tmp_path/'never.lock').exists()
    with pytest.raises(run_r2.Stop, match='ran before'):
        smoke_r2.run(out, gate=False, python=sys.executable, device='cpu', inputs=inputs, targets_path=targets, synthetic_device='cpu', env=env, config=tiny)


def test_the_smoke_needs_the_targets(tmp_path):
    with pytest.raises(run_r2.Stop, match='no such file'):
        smoke_r2.run(tmp_path/'smoke', gate=False, targets_path=tmp_path/'none.json')


def test_the_dry_run_writes_nothing(capsys):
    assert smoke_r2.dry_run() == ([] if not rc.SMOKE.exists() else [f'{rc.SMOKE} exists'])
    out = capsys.readouterr().out
    assert 'tag-r2-smoke' in out and 'AMP on' in out and 'carries no sealed list (C23.7-4 revision 6n rule 4; the guard reads them from SNAPSHOT.json\'s source_repo): yes' in out
    assert '--repo' in out and 'KeyboardInterrupt goes through' in out
    synthetic, = [line for line in out.splitlines() if line.startswith('synthetic')]
    assert '[limit' not in synthetic and '1800' not in synthetic and 'no limit of its own' in synthetic and set(smoke_r2.TIMEOUTS) == {'train'}          # new minor 2: no limit that is not enforced


# ---- R2 follow-up: the synthetic part in a try (new minor 2) ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('raised', [run_r2.Stop('boom'), RuntimeError('boom'), SystemExit('boom')])
def test_a_synthetic_chain_that_raises_is_a_synthetic_problem_and_the_reading_is_written(tmp_path, monkeypatch, raised):
    targets = tmp_path/'targets.json'
    targets.write_text(json.dumps(dict(targets=T)))
    out = tmp_path/'smoke'

    def fake_step(name, action, *, state, log_path, label):           # the training's record without a training (a passing member)
        write_log(out/'member'/rc.member_name('A'))

    def raising(work, env, device='cpu'):
        run_r2.set_run_log(work)                                         # as the real synthetic chain does: the log pointed at its own world (review 2, minor 5)
        raise raised
    monkeypatch.setattr(run_r2.ufrun, 'run_step', fake_step)
    monkeypatch.setattr(run_r2, 'synthetic_chain', raising)
    document = smoke_r2.run(out, gate=False, targets_path=targets, env=dict(os.environ), code=dict(root='/s', commit='c'))
    head, tail = document['synthetic_problems']
    assert head == f'the synthetic chain raised {type(raised).__name__}: boom' and tail.startswith('traceback (last lines): ') and 'in raising' in tail and 'boom' in tail      # minor 4
    assert document['reading'] == 'stop' and document['why'] == [f'the synthetic chain on the GPU: {head}', f'the synthetic chain on the GPU: {tail}'] and document['measured']['finite']
    assert json.loads((out/'smoke_r2.json').read_text())['reading'] == 'stop' and document['code'] == dict(root='/s', commit='c')
    assert run_r2.ufrun.LOG_FILE == out/'run_r2.log' and 'smoke: stop' in (out/'run_r2.log').read_text()
    assert 'smoke: stop' not in (out/'synthetic'/'run_r2.log').read_text() and 'the synthetic chain raised' in (out/'synthetic'/'run_r2.log').read_text()


def test_a_keyboard_interrupt_goes_through_and_nothing_is_written(tmp_path, monkeypatch):
    """Review 2, minor 5: KeyboardInterrupt is not a synthetic problem; it leaves run() and smoke_r2.json is not written."""
    targets = tmp_path/'targets.json'
    targets.write_text(json.dumps(dict(targets=T)))
    out = tmp_path/'smoke'
    monkeypatch.setattr(run_r2.ufrun, 'run_step', lambda name, action, **kw: write_log(out/'member'/rc.member_name('A')))

    def interrupted(work, env, device='cpu'):
        raise KeyboardInterrupt
    monkeypatch.setattr(run_r2, 'synthetic_chain', interrupted)
    with pytest.raises(KeyboardInterrupt):
        smoke_r2.run(out, gate=False, targets_path=targets, env=dict(os.environ))
    assert not (out/'smoke_r2.json').exists()


def test_the_smoke_prepare_refuses_a_root_that_ran(tmp_path):
    """Review 2, minor 1(a): smoke_r2.json or member/ there: refused before anything (the targets are not even read)."""
    for name in ('smoke_r2.json', 'member'):
        out = tmp_path/name/'smoke'
        out.mkdir(parents=True)
        (out/name).mkdir() if name == 'member' else (out/name).write_text('{}')
        with pytest.raises(run_r2.Stop, match='the smoke ran'):
            smoke_r2.prepare(out, tmp_path/'no_repo')
        assert not (out/'snapshot').exists()


def test_a_passing_member_and_synthetic_chain_read_pass(tmp_path, monkeypatch):
    targets = tmp_path/'targets.json'
    targets.write_text(json.dumps(dict(targets=T)))
    out = tmp_path/'smoke'
    monkeypatch.setattr(run_r2.ufrun, 'run_step', lambda name, action, **kw: write_log(out/'member'/rc.member_name('A')))
    monkeypatch.setattr(run_r2, 'synthetic_chain', lambda work, env, device='cpu': [])
    document = smoke_r2.run(out, gate=False, targets_path=targets, env=dict(os.environ), code=dict(root='/s'))
    assert document['reading'] == 'pass' and document['why'] == [] and document['targets_sha256'] == rc.sha256_file(targets) and document['code'] == dict(root='/s')


# ---- R2 follow-up: prepare checks the smoke first (new minor 1) --------------------------------------------------------------------------------------------
def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', *args], check=True, capture_output=True)


@pytest.fixture
def smoked(tmp_path):
    """A tiny git repository, a smoke root with its snapshot made from the repository's HEAD and smoke_r2.json reading 'pass' on targets.json."""
    from bayesreg.seg.unet.snapshot import make_snapshot
    repo = tmp_path/'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    (repo/'a.txt').write_text('a\n')
    git(repo, 'add', 'a.txt')
    git(repo, 'commit', '-q', '-m', 'one')
    targets = tmp_path/'targets.json'
    targets.write_text(json.dumps(dict(targets=T)))
    smoke = tmp_path/'smoke'
    record = make_snapshot(repo, smoke/'snapshot', paths=('a.txt',))
    code = run_r2.snapshot_identity(smoke/'snapshot')                     # what the smoke unit writes (main: require_snapshot, then snapshot_identity of CODE_ROOT)
    (smoke/'smoke_r2.json').write_text(json.dumps(dict(reading='pass', why=[], targets_sha256=rc.sha256_file(targets), code=code)))
    return dict(repo=repo, targets=targets, smoke=smoke, commit=record['commit'], out=tmp_path/'out', code=code)


def gate_of(s, snapshot=None):
    return run_r2.smoke_gate(snapshot or s['out']/'snapshot', s['repo'], smoke=s['smoke'], targets_path=s['targets'])


def test_the_smoke_gate_passes_on_the_smoke_s_commit(smoked):
    problems, code = gate_of(smoked)
    assert problems == [] and code == smoked['code'] and code['commit'] == smoked['commit'] and code['root'] == str((smoked['smoke']/'snapshot').resolve())


def test_an_existing_training_snapshot_at_the_smoke_s_commit_passes_after_the_repository_moved(smoked):
    """Review 2, minor 5: the branch of an existing training snapshot, positive: the snapshot was made at the smoke's commit with the same paths; the repository moved on since."""
    from bayesreg.seg.unet.snapshot import make_snapshot
    make_snapshot(smoked['repo'], smoked['out']/'snapshot', paths=('a.txt',))
    _moved(smoked)
    assert gate_of(smoked)[0] == []


def _moved(s):
    (s['repo']/'b.txt').write_text('b\n')
    git(s['repo'], 'add', 'b.txt')
    git(s['repo'], 'commit', '-q', '-m', 'two')


def _tampered(s):
    (s['smoke']/'snapshot'/'a.txt').write_text('changed\n')


def _training_snapshot_elsewhere(s):
    from bayesreg.seg.unet.snapshot import make_snapshot
    _moved(s)
    make_snapshot(s['repo'], s['out']/'snapshot', paths=('a.txt',))


def _training_snapshot_other_files(s):
    """Review 2, minor 3: the same commit, other files (another path table): the commit label alone would let it through."""
    from bayesreg.seg.unet.snapshot import make_snapshot
    (s['repo']/'c.txt').write_text('c\n')
    git(s['repo'], 'add', 'c.txt')
    git(s['repo'], 'commit', '-q', '--amend', '--no-edit')                     # c.txt in a commit the smoke snapshot is not at...
    record = make_snapshot(s['repo'], s['out']/'snapshot', paths=('a.txt', 'c.txt'))
    document = json.loads((s['out']/'snapshot'/'SNAPSHOT.json').read_text())
    document['commit'] = s['commit']                                           # ...relabelled as the smoke's commit (SNAPSHOT.json itself is not in its files)
    (s['out']/'snapshot'/'SNAPSHOT.json').write_text(json.dumps(document))
    assert record['commit'] != s['commit']


def _relabelled_smoke(s):
    """Review 2, minor 3 (the probe w4): the smoke snapshot's commit label changed to the repository's new HEAD; its files are the old commit's."""
    _moved(s)
    new = subprocess.run(['git', '-C', str(s['repo']), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    path = s['smoke']/'snapshot'/'SNAPSHOT.json'
    document = json.loads(path.read_text())
    document['commit'] = new
    path.write_text(json.dumps(document))


def _rebuilt_smoke_snapshot(s):
    """Review 2, minor 1(b) (the probe w6): the smoke snapshot deleted and made again at a newer HEAD; smoke_r2.json's code is the old one's."""
    import shutil
    from bayesreg.seg.unet.snapshot import make_snapshot
    shutil.rmtree(s['smoke']/'snapshot')
    _moved(s)
    make_snapshot(s['repo'], s['smoke']/'snapshot', paths=('a.txt',))


def _write(s, text, name='smoke_r2.json'):
    (s['smoke']/name).write_text(text)


def _no_commit_in_smoke_snapshot(s):
    path = s['smoke']/'snapshot'/'SNAPSHOT.json'
    document = json.loads(path.read_text())
    document.pop('commit')
    path.write_text(json.dumps(document))


def _code(s, **changes):
    document = json.loads((s['smoke']/'smoke_r2.json').read_text())
    document['code'] = None if changes.get('none') else dict(document['code'], **{k: v for k, v in changes.items() if k != 'none'})
    _write(s, json.dumps(document))


@pytest.mark.parametrize('change, words', [
    (lambda s: (s['smoke']/'smoke_r2.json').unlink(), 'no such file'),
    (lambda s: (s['smoke']/'smoke_r2.json').write_text(json.dumps(dict(reading='stop', why=['GPU'], targets_sha256=rc.sha256_file(s['targets'])))), "reading is 'stop'"),
    (lambda s: s['targets'].write_text(json.dumps(dict(targets=T[:-1]))), 'targets_sha256'),
    (lambda s: s['targets'].unlink(), 'targets_sha256'),
    (_moved, 'the smoke ran on commit'),
    (_tampered, "the smoke's snapshot"),
    (_training_snapshot_elsewhere, 'the smoke ran on commit'),
    (_training_snapshot_other_files, 'holds other files'),                                  # review 2, minor 3
    (_relabelled_smoke, "smoke_r2.json's code"),                                            # review 2, minor 3 (w4)
    (_rebuilt_smoke_snapshot, "smoke_r2.json's code"),                                      # review 2, minor 1(b) (w6)
    (lambda s: _code(s, none=True), "smoke_r2.json's code None"),                          # minor 1(b): no code record
    (lambda s: _code(s, files_sha256='0'*64), "smoke_r2.json's code"),
    (lambda s: _code(s, root='/elsewhere/snapshot'), "smoke_r2.json's code"),
    (lambda s: _write(s, json.dumps(['pass'])), 'not a JSON object'),                      # review 2, minor 2
    (lambda s: _write(s, '{"reading": "pass",'), 'cannot be read'),
    (_no_commit_in_smoke_snapshot, "the smoke's snapshot"),
])
def test_the_smoke_gate_refuses(smoked, change, words):
    change(smoked)
    problems, _ = gate_of(smoked)
    assert problems and any(words in p for p in problems), problems


@pytest.mark.parametrize('change', [lambda s: _write(s, json.dumps(['pass'])), lambda s: _write(s, '{"reading"'), _no_commit_in_smoke_snapshot])
def test_prepare_refuses_a_broken_input_with_a_stop(smoked, change):
    """Review 2, minor 2: a Stop (main: exit 8), not a traceback; before any snapshot."""
    change(smoked)
    with pytest.raises(run_r2.Stop, match='GPU smoke'):
        run_r2.prepare(smoked['out'], smoked['repo'], smoke=smoked['smoke'], targets_path=smoked['targets'])
    assert not (smoked['out']/'snapshot').exists()


def test_prepare_checks_the_snapshot_it_made_again(smoked, monkeypatch):
    """Review 2, minor 5: the repository moves between smoke_gate and make_snapshot: the snapshot made is at another commit; Stop, and the snapshot is kept."""
    from bayesreg.seg.unet import snapshot as S
    real = S.make_snapshot

    def moving(repo, dest, *, paths):
        _moved(smoked)
        return real(repo, dest, paths=('a.txt',))
    monkeypatch.setattr(run_r2, 'make_snapshot', moving)
    with pytest.raises(run_r2.Stop, match='kept, not used'):
        run_r2.prepare(smoked['out'], smoked['repo'], smoke=smoked['smoke'], targets_path=smoked['targets'])
    assert (smoked['out']/'snapshot'/'SNAPSHOT.json').is_file()


def test_the_chain_checks_the_smoke_before_its_first_step(tmp_path, monkeypatch):
    """Review 2, minor 6: cmd_chain refuses before the lock, the plan or any step when the smoke does not pass (here: no smoke at all)."""
    import types
    monkeypatch.setattr(run_r2, 'require_snapshot', lambda: None)
    monkeypatch.setattr(rc, 'SMOKE', tmp_path/'no_smoke')
    monkeypatch.setattr(rc, 'PP_ROOT', tmp_path/'no_pp')
    monkeypatch.setattr(r2_lock, 'check', lambda *a, **k: pytest.fail('the lock is asked after the smoke'))
    monkeypatch.setattr(run_r2, 'real_spec', lambda *a, **k: pytest.fail('the chain went on'))
    with pytest.raises(run_r2.Stop, match='the chain refused before its first step .the GPU smoke.: .*no such file'):
        run_r2.cmd_chain(types.SimpleNamespace(out=tmp_path/'out', wait_minutes=30))


def test_the_snapshot_path_table_carries_no_sealed_list():
    """Follow-up 3, item 3, as contract C23.7-4 revision 6n rule 4 reads it: the paths of SEALED_LISTS (taken from the constant; no list opened) are not in SNAPSHOT_PATHS_R2, and the check
    would see one that were."""
    from bayesreg.io.sealed import SEALED_LISTS
    assert run_r2.snapshot_path_problems() == []
    assert run_r2.snapshot_path_problems(paths=run_r2.SNAPSHOT_PATHS_R2+('splits/bayesreg_v1',)) == [f"the snapshot path table carries the sealed list {SEALED_LISTS[0]['path']}"]
    assert len(run_r2.snapshot_path_problems(paths=run_r2.SNAPSHOT_PATHS_R2+(SEALED_LISTS[1]['path'],))) == 1


def test_prepare_refuses_before_any_snapshot(smoked):
    (smoked['smoke']/'smoke_r2.json').write_text(json.dumps(dict(reading='stop', why=['GPU'], targets_sha256=rc.sha256_file(smoked['targets']))))
    with pytest.raises(run_r2.Stop, match='GPU smoke'):
        run_r2.prepare(smoked['out'], smoked['repo'], smoke=smoked['smoke'], targets_path=smoked['targets'])
    assert not (smoked['out']/'snapshot').exists()
