"""PQH's admission and code tree (review B1, B2, I1 准入, I5; contract C22.H4-4, C22.H5-2, C22.H8 实现门槛 准入, C23.4-2, C23.8-1): SG's real loaders on the simulated unseal branch
(tests/test_bayesreg/xb/_world.simulated_unseal: the real sealed id lists copied, a record written as the lead would on BZ's behalf), the real ids of the pair list and the sealed lists (id lists
only, no subject's file), HI's real check of its development-check decision, and the synthetic world for the dev mode.

- B1: the hold-out members unit, under the simulated unseal and the spy (watching the results root and the code tree), reads HI's devcheck.json and its lock through hi_core.check_decision and
  reaches its lock without a breach;
- B2: the allowlist's sources are allowed where they are re-hashed (the main checkout), path by path; the dev members unit with the real allowlist loader from a checkout other than the main one;
- I1: H1-H4, T0 and training ids are refused by both units; a pair not in holdout_B_pairs.tsv is refused; an allowlist copied, replaced or pickled is refused where PQH uses it;
- I5: the smoke runs from he_smoke_v3/code, clean (pqh_units.code_check with the real git)."""
import copy
import dataclasses
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest
import torch

from . import _world as W

hp, hin, hu, io_, hc, XW = W.hp, W.hin, W.hu, W.io_, W.hc, W.XW


def allowlist_document():
    return json.loads((W.ROOT/W.pins.ALLOWLIST_REL).read_text(encoding='utf-8'))


def source_ids(name):
    return sorted({s for src in allowlist_document()['sources'] if src['name'] == name for s in src['subjects']})


def sealed_ids(name):
    from bayesreg.io.sealed import load_sealed_lists
    return sorted(load_sealed_lists(W.ROOT).subjects[name])


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """The detached checkout holdout_B_v1/code (a link to this tree), the preregistration pinned to this tree's file."""
    path = Path(hc.HOLDOUT_ROOT)/'code'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(W.ROOT)
    monkeypatch.setattr(hp, 'PREREG_SHA256', io_.sha_file(W.ROOT/hp.PREREG_REL))
    return path


def out_layout(checkout, pairs):
    layout = hin.holdout_layout(checkout, pairs=pairs)
    W.finalized(layout, {p: W.no_output_row() for p in layout.pairs})
    return layout


# ---- I1: the ids the units never admit -----------------------------------------------------------------------------------------------------------------
def foreign_ids():
    """One id of each kind the hold-out units refuse: H1, H4 (OASIS ids of the paper-1 hold-out lists), H3 (IBSR18), H2 (LPBA40 S37-S40), T0, the training set."""
    from bayesreg.io.sealed import SEALED_LPBA40
    return dict(H1=sealed_ids('H1')[0], H4=sealed_ids('H4')[0], H3=sealed_ids('H3')[0], H2=sorted(SEALED_LPBA40)[0], T0=source_ids('stage_T0_tune_v1')[0],
                training=source_ids('bayesreg_train_v1')[0])


@pytest.mark.parametrize('side', ['fixed', 'moving'])
@pytest.mark.parametrize('kind', ['H1', 'H4', 'H3', 'H2', 'T0', 'training'])
def test_the_out_unit_refuses_h1_h4_t0_and_training_ids(checkout, tmp_path, monkeypatch, kind, side):
    """One run per test: a refusal record makes the next attempt need a rerun authorization (holdout_common I1)."""
    XW.simulated_unseal(tmp_path, monkeypatch)
    foreign = foreign_ids()[kind]
    fixed, moving = hc.holdout_pairs()[0]
    layout = out_layout(checkout, [(foreign, moving) if side == 'fixed' else (fixed, foreign)])
    assert W.run_out(layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(layout)[-1])
    assert refusal['stage'] in ('never', 'admission'), refusal['reason']
    assert refusal['stage'] == 'never' if kind in ('H1', 'H4') else True
    assert not [f for f in refusal['spy']['files'] if foreign in f['path']] and not Path(layout.out_lock).exists()


@pytest.mark.parametrize('kind', ['H1', 'H4', 'H3', 'H2', 'T0', 'training'])
def test_the_members_admission_refuses_h1_h4_t0_and_training_ids(checkout, tmp_path, monkeypatch, kind):
    """The members unit admits through hi_core (C22.H5-2: load_unseal, bind, admit) -- not admit_holdout_pair (review m11)."""
    import pq_members
    sim = XW.simulated_unseal(tmp_path, monkeypatch)
    layout = hin.holdout_layout(checkout)
    monkeypatch.setattr(pq_members, 'member_files', lambda root, members=None: {'m': Path(root)/'m'/'member.pt'})     # the four files are pinned by sha256 (pq_members' own tests)
    deps = W.deps(allowlist=lambda: (sim.allowlist, dict(allowlist_sha256=sim.allowlist.sha256)))
    with pytest.raises(Exception) as refused:
        hu.members_admission(layout, deps, {}, subjects=[foreign_ids()[kind]])
    assert type(refused.value).__name__ in ('HIRefused', 'SubjectSealed'), refused.value


def test_a_pair_not_in_the_pair_list_is_refused(checkout, tmp_path, monkeypatch):
    XW.simulated_unseal(tmp_path, monkeypatch)
    rows = hc.holdout_pairs()
    layout = out_layout(checkout, [(rows[0][0], rows[1][1])])                     # two unsealed subjects, not a row
    assert W.run_out(layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(layout)[-1])
    assert refusal['stage'] == 'admission' and 'is not a row' in refusal['reason']


def test_a_copied_replaced_or_pickled_allowlist_is_refused_where_pqh_uses_it(checkout, tmp_path, monkeypatch):
    """C23.3-5 (ii), C23.8-1: PQH passes the bound allowlist admit_holdout_pair returns to load_pair (the out unit) and hi_core.admit (the members unit); a copy, a replaced one or a pickled one
    is not that object."""
    sim = XW.simulated_unseal(tmp_path, monkeypatch)
    fixed, moving = hc.holdout_pairs()[0]
    layout = out_layout(checkout, [(fixed, moving)])
    _, bound = hc.admit_holdout_pair(fixed, moving)
    XW.write_stage_pair(layout.stage, layout.pairs[0], seg=False)
    assert hin.load_pair_checked(layout, layout.pairs[0], bound).fixed_subject == fixed
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(bound)
    for forged in (copy.copy(bound), dataclasses.replace(bound)):
        with pytest.raises(Exception) as refused:
            hin.load_pair_checked(layout, layout.pairs[0], forged)
        assert type(refused.value).__name__ in ('SubjectSealed', 'InputValidationError', 'UnsealRefused'), refused.value
    import hi_core
    unseal = hi_core.load_unseal(Path(sim.repo)/hi_core.UNSEAL_REL, anchor=hp.ANCHOR)
    good, _ = hi_core.bind(unseal, sim.allowlist)
    for forged in (copy.copy(good), dataclasses.replace(good)):
        with pytest.raises(Exception) as refused:
            hi_core.admit(fixed, unseal=unseal, allowlist=forged, sealed=None, training=[])
        assert type(refused.value).__name__ in ('HIRefused', 'SubjectSealed'), refused.value


# ---- B1: the hold-out members unit reaches its lock under the spy ---------------------------------------------------------------------------------
def test_b1_the_hold_out_members_unit_reaches_its_lock_without_a_breach(checkout, tmp_path, monkeypatch):
    """Review B1: hi_core.check_decision (the real one) opens devcheck.json and the development check's lock; both are allowed items of the hold-out members unit (the lead's ruling (1) of 11-09).
    The spy watches the sandbox's results root and the code tree; the unit runs the whole admission (load_unseal, bind, admit of the 100 subjects, HI's runs, the decision) and takes its lock.
    The inference itself is stopped at its first step (the test's stand-in raises), so the records after the lock are written too."""
    import hi_core
    import pq_members
    sim = XW.simulated_unseal(tmp_path, monkeypatch)
    layout = hin.holdout_layout(checkout)
    layout.extra.update(watched=(tmp_path, W.ROOT, XW.ROOT), unwatched=())
    W.finalized(layout, {p: W.no_output_row() for p in layout.pairs})
    W.hi_run(layout.hi_lock, Path(hp.RESULTS)/'seg'/'hi_run_1', layout.subjects)
    W.decided(monkeypatch)
    monkeypatch.setattr(pq_members, 'member_files', lambda root, members=None: {'m': Path(root)/'m'/'member.pt'})
    monkeypatch.setattr(pq_members, 'plan_for', lambda registry, subject, allowlist, available: ([('model', 'm', 'path')], {'model': {}}, 'all_folds_mean'))

    def stop_here(layout, subject, reference):
        raise RuntimeError('the test stops at the first subject, after the lock')
    monkeypatch.setattr(hu, 'reference_prediction', stop_here)
    deps = W.deps(allowlist=lambda: (sim.allowlist, dict(allowlist_sha256=sim.allowlist.sha256)), device=lambda device, free: torch.device('cpu'),
                  environment=lambda device, decision: dict(device=str(device), decision=decision['reading']))
    XW.sealed_module._reset_cache()
    assert hu.members_unit(layout, device='cpu', deps=deps) == hu.EXIT['stopped']
    small, _ = hin.run_dirs(layout, 'members', 'first_run')
    assert Path(layout.members_lock).is_file() and not W.refusals(layout) and not (small/hu.STOPPED).exists()
    record = io_.read_json(small/hin.MEMBERS_RECORD)
    assert record['stage'] == 'error' and 'stops at the first subject' in record['error'] and record['checks']['decision']['reading'] == hi_core.READ_SELECT.format(n=20)
    opened = io_.read_json(small/'opened_files_members.json')
    assert opened['m6'] is True and opened['breaches'] == []
    names = {f['path'] for f in opened['files']}
    assert str((Path(hi_core.DEVCHECK_OUT)/hi_core.DEVCHECK_RECORD).resolve()) in names and str(Path(hi_core.DEVCHECK_LOCK).resolve()) in names
    assert len(record['admission']) == 100


def test_b1_without_the_allowed_devcheck_files_the_unit_would_breach(checkout, tmp_path, monkeypatch):
    """The rule itself: the two files are in the members unit's allowed items in the hold-out mode only."""
    import hi_core
    layout = hin.holdout_layout(checkout)
    layout.extra.update(watched=(tmp_path,), unwatched=())
    files = (Path(hi_core.DEVCHECK_OUT)/hi_core.DEVCHECK_RECORD, Path(hi_core.DEVCHECK_LOCK))
    members, out = hu.rules_for(layout, 'members'), hu.rules_for(layout, 'out')
    assert all(members.judge(f) is None for f in files) and all('default deny' in out.judge(f) for f in files)
    assert 'default deny' in members.judge(Path(hi_core.DEVCHECK_OUT)/'subjects'/'x.json')


# ---- B2: the allowlist's sources where they are re-hashed ------------------------------------------------------------------------------------------------
def test_b2_the_allowlist_sources_are_allowed_path_by_path_under_the_main_checkout(checkout, tmp_path):
    layout = hin.holdout_layout(checkout, pairs=[hc.holdout_pairs()[0]])
    layout.extra.update(watched=(tmp_path, W.ROOT, hp.MAIN_REPO), unwatched=())
    sources = allowlist_document()['sources']
    assert [s for s in sources if s['path'] == 'splits/bayesreg_v1/train_subjects.txt']
    for unit in ('members', 'out'):
        rules = hu.rules_for(layout, unit)
        for s in sources:
            assert (Path(hp.MAIN_REPO)/s['path']).resolve() in rules.files, (unit, s['path'])
        assert 'default deny' in rules.judge(Path(hp.MAIN_REPO)/'splits'/'bayesreg_v1'/'eb_pairs.tsv')


def test_b2_the_dev_members_unit_with_the_real_allowlist_loader(tmp_path, monkeypatch):
    """Review B2: the dev members unit loads the pinned allowlist with bayesreg.seg.unet.pins.load_pinned_allowlist from its checkout (he_smoke_v3/code) and re-hashes the sources under the
    main checkout (MAIN_REPO, here a directory of its own with the synthetic sources): no breach.  Only the clean-tree check of the import tree is stood in (the synthetic tree is not that tree)."""
    import bayesreg.seg.unet.pins as unet_pins
    import hi_core
    import pq_members
    w, layout, _ = W.dev_world(tmp_path, monkeypatch)
    small, big = hin.run_dirs(layout, 'members', 'first_run')
    shutil.rmtree(small)
    shutil.rmtree(big)
    main = tmp_path/'main'
    for s in json.loads((w.repo/W.pins.ALLOWLIST_REL).read_text())['sources']:
        (main/s['path']).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(w.repo/s['path'], main/s['path'])
    monkeypatch.setattr(hp, 'MAIN_REPO', main)
    monkeypatch.setattr(unet_pins, 'DEFAULT_ALLOWLIST', Path(layout.code)/W.pins.ALLOWLIST_REL)
    monkeypatch.setattr(unet_pins, 'code_record', lambda: dict(kind='test'))
    monkeypatch.setattr(unet_pins, 'require_clean_code', lambda record=None: dict(kind='git_worktree', clean=True, note='test: the synthetic tree'))
    monkeypatch.setattr(hi_core, 'check_registry', lambda registry, anchor: dict(path=str(registry.path), sha256=registry.sha256))
    monkeypatch.setattr(pq_members, 'member_files', lambda root, members=None: {'m': Path(root)/'m'/'member.pt'})
    monkeypatch.setattr(pq_members, 'plan_for', lambda registry, subject, allowlist, available: ([('model', 'm', 'path')], {'model': {}}, 'all_folds_mean'))

    def mean_and_variance(subject, *, files, models, plan, device):
        with np.load(w.ctx.pred_root/subject/'probs36.npz') as z:
            mean = np.array(z['p']).astype(np.float32)
        with np.load(w.ctx.pq_root/'members'/subject/io_.VAR_FILE) as z:
            V = np.array(z['V'])
        return mean, V, dict(sha256=W.PW.source_sha(subject)), dict(percentile=99.5), [dict(member_sha256='m')]
    monkeypatch.setattr(pq_members, 'mean_and_variance', mean_and_variance)
    assert Path(layout.code).resolve() != main.resolve()
    deps = W.deps(device=lambda device, free: torch.device('cpu'), environment=lambda device, decision: dict(device=str(device)))
    XW.sealed_module._reset_cache()
    assert hu.members_unit(layout, device='cpu', deps=deps) == 0
    record = io_.read_json(small/hin.MEMBERS_RECORD)
    assert record['checks']['allowlist']['source_root'] == str(main) and record['checks']['allowlist']['allowlist_sources_verified']
    opened = io_.read_json(small/'opened_files_members.json')
    assert opened['m6'] is True and any(f['path'] == str((main/'splits'/'bayesreg_v1'/'train_subjects.txt').resolve()) for f in opened['files'])


# ---- I5: the smoke from its checkout, clean ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def dev(tmp_path, monkeypatch):
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    return W.namespace(w=w, layout=layout, tmp=tmp_path)


def test_i5_the_dev_out_unit_runs_from_the_smoke_checkout_with_the_real_git(dev):
    assert Path(dev.layout.code) == Path(hp.SMOKE_ROOT)/'code'
    assert W.run_out(dev.layout, deps=W.deps(identity=None)) == 0                    # pqh_units.checkout_code_identity: bayesreg.io.code_identity on the real git
    document = io_.read_json(dev.layout.pq_small/W.PAIR/hu.PQ_JSON)
    head = XW.git(dev.w.repo, 'rev-parse', 'HEAD').stdout.strip()
    assert document['code'] == dict(commit=head, head=head, clean=True)


def test_i5_a_dirty_smoke_checkout_is_refused(dev):
    (dev.w.repo/'configs'/'bayesreg'/'extra.json').write_text('{}')                     # untracked in a code directory
    assert W.run_out(dev.layout, deps=W.deps(identity=None)) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['stage'] == 'code' and 'configs/bayesreg/extra.json' in refusal['reason'] and refusal['checks']['code']['clean'] is False


def test_i5_a_modified_tracked_file_outside_the_code_refuses_the_smoke(dev):
    """Review 3 r1 (the stricter reading): the smoke's checkout has no tracked file modified anywhere, docs/ and runs/ included -- rule one's working-tree part; an untracked file outside the
    code directories is not a change of a tracked file."""
    note = dev.w.repo/'docs'/'note.md'
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text('committed\n')
    XW.git(dev.w.repo, 'add', '-A')
    XW.git(dev.w.repo, 'commit', '-q', '-m', 'a tracked document (test)')
    (dev.w.repo/'docs'/'untracked.md').write_text('not tracked\n')
    code, problems = hu.smoke_code_identity(dev.layout.code)
    assert problems == [] and code['clean'] is True and code['tracked_changes'] == []
    note.write_text('changed, not committed\n')
    code, problems = hu.smoke_code_identity(dev.layout.code)
    assert problems and code['clean'] is False and any('docs/note.md' in c for c in code['tracked_changes'])
    assert hu.checkout_code_identity(dev.layout.code)[1] == []                      # the replay gate's run only checks the code (B1 of 1:53 AM), unchanged
    assert W.run_out(dev.layout, deps=W.deps(identity=None)) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['stage'] == 'code' and 'tracked files modified' in refusal['reason']


def test_i5_another_tree_than_the_smoke_checkout_is_refused(dev):
    other = dev.tmp/'other_tree'
    shutil.copytree(dev.w.repo, other, symlinks=True)
    dev.layout.code = other
    assert W.run_out(dev.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(dev.layout)[-1])
    assert refusal['stage'] == 'code' and 'is not the detached checkout' in refusal['reason']
