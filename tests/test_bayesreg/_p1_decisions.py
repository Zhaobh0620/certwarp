"""P1 (runs/20261115-p1/实现说明_P1.md; prereg revision 2 item 9 step 5; contract C24.3-6): the four decision constants filled after the development gates ran on P0'' (318e72ad,
D5-D8, 2026-11-15; runs/20261115-d58/result_C1.md, C2, C4, C6).  The values are written here again from those result documents, so a test reads each constant against them,
against the real record and lock (read only: their sha256 before and after), and the fixed range P0''..P1 (318e72ad..3005a57e) of this repository against rule two (until 2026-11-20 the range
ran to HEAD; rule_two_record says why it was re-pinned).  Each module's tests call it next to the test that sets its constant back to None and checks the old refusal (the P0 state).  Nothing
here reads a label, a segmentation or a truth file."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
P0PP = '318e72ad3d097bcbd03db180a56c0144ee4344bc'      # P0'' (AF, merged b595d110): every gate's lock and record name it
P1_COMMIT = '3005a57eb06ac820a620138aa5a09cebebfcdf45'  # P1 = P0'' + the four constants (decision_log 2026-11-15 ~10:05 PM CST); runs/holdout_B_v1/pinned_code.json names it, the hold-out ran on its code
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
HOMES = ('bayesreg/eval/holdout.py', 'experiments/pq/pqh_pins.py', 'experiments/holdout/prep_stage.py', 'experiments/holdout/atropos_holdout.py')
REVIEW = re.compile(r'(?P<path>docs/bayesreg/reviews/[^ ]+\.md) \(sha256 (?P<sha>[0-9a-f]{64})[;)]')
REVIEW_FIELD = ('docs/bayesreg/reviews/D58_复核_20261115.md (sha256 9629f2495a4f099347871e4eb3d606a22f2484838745e65e494fe1145f946c4b; 0 blocking, 1 important, 8 minor; '
                'reading holds)')                     # the independent review of D1's rerun and D5-D8 on P0'' (main be86939c), as the lead gave it
DECIDED_BY = 'lead (total responsible), 2026-11-15, per {}'          # HI's DEVCHECK_DECISION way (hi_core.py), each with its own plan and sections

# ---- the values, from the result documents (runs/20261115-d58/) ------------------------------------------------------------------------------------
PREP = dict(devcheck_json=str(RESULTS/'hp_devcheck_v1'/'prep'/'devcheck_prep.json'), devcheck_sha256='ab6a22199311088534f809afa2d336742ed0aa337670833030c84b7464c75b34',
            reading='逐位相同 19/19', plan_sha256='aee9ff25eaee3f467de664c66c229f169b56f77f2a19fb7f40f57fb30f2f8c97', code_commit=P0PP, bz_words=None,
            decided_by=DECIDED_BY.format('prep_devcheck_plan.md §4'), review=REVIEW_FIELD)
PREP_LOCK = (RESULTS/'hp_devcheck_v1'/'locks'/'prep_devcheck.lock', '101120b7509b73397e6d6d4ed089524be172f43ef72bcdfaa597e24d228daaeb')
ATROPOS = dict(devcheck_json=str(RESULTS/'hp_devcheck_v1'/'atropos'/'devcheck_atropos.json'), devcheck_sha256='63ce8c7e548547745f2b1b0ea5944c3d480d226eae707660a2c06bacb37128c3',
               reading='逐位相同 2/2', plan_sha256='678ea102dfa310222e26ace9ce9da3ac239c5d1372cecf8784e28cf44d706a23', code_commit=P0PP, bz_words=None,
               decided_by=DECIDED_BY.format('atropos_devcheck_plan.md §4'), review=REVIEW_FIELD)
ATROPOS_LOCK = (RESULTS/'hp_devcheck_v1'/'locks'/'atropos_devcheck.lock', '8772a16fef4df4c97a34a9efafd1385975562a69649534b7009a195a2be29930')
TRUTH = dict(result_json=str(RESULTS/'ev'/'truth_gate_v1'/'truth_gate.json'), result_sha256='3abeee225a5e27dc1e8ad31a6017b804775d6d8487162d7be07b6c750a131eb0',
             reading='逐位相同 20/20（seg35、seg4）；指标逐位相同 19/19', code_commit=P0PP, environment=dict(numpy='2.0.2', nibabel='5.4.0'),
             counts=dict(reader=dict(subjects=20, identical=20), metrics=dict(pairs=19, identical=19)), plan_sha256='eb76887ec71ca3626a6af600e78d87953ec560f02f82d3be58798d20178ba302',
             decided_by=DECIDED_BY.format('truth_gate_plan.md §3, §4'), review=REVIEW_FIELD)
TRUTH_LOCK = (RESULTS/'ev'/'truth_gate.lock', 'ce498aea9aaa39c53866cb62216645e3f7c2005b4e63b85acbfe0ac2a2501846')
PQ = dict(result_json=str(RESULTS/'pq_replay_v1'/'result'/'replay_result.json'), result_sha256='3f37ef8e9c4dc5955cb7e6a8fd29e95aa891824d7925bd83f742bbf809bae91a',
          reading='逐位相同 19/19', code_commit=P0PP, plan_sha256='045e68cc5ae224916de5918e2c63a169a40d79b40bd0fc88c56e42386b9924d1',
          environment=dict(python='3.10.20', numpy='2.0.2', scipy='1.15.2', nibabel='5.4.0'), decided_by=DECIDED_BY.format('replay_gate_plan.md §4, §6'), review=REVIEW_FIELD)
PQ_LOCK = (RESULTS/'pq_replay_v1'/'pq_replay_gate.lock', 'd23b9f90dc2ae612735488c083a1883e82bc11e0c61444abed5a05075a787bbe')


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def at_gate(text, name):
    """The text of a constant's file as at the gates' commit (P0''): its one top-level `name = <literal>` statement (the lines from its first to its last, found on the AST) back to
    `name = None` -- the P0 state the tests that build a gate's tree start from (rule two allows exactly the other direction).  The P1 test of each file checks that this is the blob of P0''."""
    import ast
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign) and [getattr(t, 'id', None) for t in n.targets] == [name]]
    assert len(nodes) == 1, f'{name}: {len(nodes)} top-level assignments'
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:nodes[0].lineno-1]+[f'{name} = None\n']+lines[nodes[0].end_lineno:])


def blob_at(commit, rel, repo=REPO):
    done = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{rel}'], capture_output=True)
    assert done.returncode == 0, done.stderr.decode('utf-8', 'replace')[:300]
    return done.stdout.decode('utf-8')


def plain(value):
    """A literal as JSON would read it back (the constants are plain dict literals: C24.3-6 (a))."""
    return json.loads(json.dumps(value))


def assert_values(constant, keys, values):
    """The constant is a plain dict with exactly `keys`, equal to `values` (every key: decided_by and review included); review names a committed review file of this repository by its
    sha256 (no placeholder left)."""
    assert type(constant) is dict and sorted(constant) == sorted(keys) == sorted(values), constant
    assert plain(constant) == plain(values)
    assert_review(constant['review'])


def assert_review(review):
    assert isinstance(review, str) and '<' not in review and '>' not in review, f'a placeholder is left in the review field: {review!r}'
    found = REVIEW.match(review)
    assert found, review
    assert (REPO/found['path']).is_file() and sha256_of(REPO/found['path']) == found['sha'], review


def real_evidence(*paths):
    """Decision diff review u1 of HI (tests/test_bayesreg/hi/test_hi_once.py): where any of the record and its lock is on this machine, all must be (a missing one is lost evidence); where
    none is (another machine, a fresh checkout), the test says so and is skipped -- the values are checked by the test next to it.  Returns their sha256 (read only)."""
    paths = [Path(p) for p in paths]
    if not any(p.exists() for p in paths):
        pytest.skip(f'none of {[str(p) for p in paths]} is on this machine: only the constant\'s values are checked')
    assert all(p.is_file() for p in paths), f'only some of {[(str(p), p.exists()) for p in paths]} are there: evidence of the gate is missing'
    return tuple(sha256_of(p) for p in paths)


def lock_line(path):
    return json.loads(Path(path).read_text(encoding='utf-8').splitlines()[0])


WORKTREE_CALLS = (('status', '--porcelain=v1', '-z', '--untracked-files=no'), ('ls-files', '-z', '--others', '--'), ('ls-files', '-z', '--others', '--directory'))    # rule two's step 3


def rule_two_record(commit=P0PP, repo=REPO, head=P1_COMMIT):
    """bayesreg.io.code_identity.gate_identity_problems from `commit` to `head` of this repository: (record, its problems).  The working tree is checked by the gates themselves at run time, not
    here, so a local edit does not fail this test: the three git calls of rule two's step 3 (tracked changes of the code, untracked or ignored files in the code directories, importable files at
    the top -- WORKTREE_CALLS) are answered as a clean tree's.  Every other problem is kept (P1 review minor 1: the rows of `changes`, the ancestry and any git call that fails -- a failing
    `git diff` leaves `changes` empty and must not pass).

    `head` is P1 (3005a57e), a fixed commit, not HEAD: re-pinned 2026-11-20 (the lead's brief to the implementer; tests only, no code changed).  Rule two (contract C24.3-6) is the code
    identity of the gates' four constants: from the commit the gates ran on (P0'') to the commit the hold-out ran on, the code changes only by those four values.  That commit is P1: the
    unseal commit 7d6b6653 has P1 as its parent, runs/holdout_B_v1/pinned_code.json names P1, every hold-out unit checked P1..its checkout's HEAD by rule one (C24.3-4, -5), and the lead
    recorded exactly this range as passed: gate_identity_problems('318e72ad', head='3005a57e') (decision_log 2026-11-15 ~10:05 PM CST).  While main stood at P1's code (to 2026-11-19)
    P0''..HEAD was that range.  After the hold-out run ended, main added code that rule two refuses by design -- T24's bayesreg/eval/holdout_tet24.py (C24.12, its own checkout and its own
    exception D) and the LPBA40 tools (contract C26) -- so P0''..HEAD stopped being the gates' range and the test failed on main's new code, not on anything rule two is about."""
    from bayesreg.io.code_identity import gate_identity_problems

    def run(args, **kwargs):
        if any(tuple(args[6:6+len(call)]) == call for call in WORKTREE_CALLS):      # git_runner: ['git', '--literal-pathspecs', '-c', 'core.quotepath=off', '-C', <repo>, *args]
            return subprocess.CompletedProcess(args, 0, '' if kwargs.get('text') else b'', '' if kwargs.get('text') else b'')
        return subprocess.run(args, **kwargs)
    record, problems = gate_identity_problems(commit, repo=repo, run=run, head=head)
    if not record.get('gate_is_ancestor'):
        problems = problems+[f'{commit} is not an ancestor of {head}']
    return record, problems
