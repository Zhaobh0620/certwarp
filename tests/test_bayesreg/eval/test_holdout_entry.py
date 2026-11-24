"""bayesreg.eval.holdout (contract C23.12, C22.H6-H7; the interface alignment runs/20261109-align/接口对齐.md): the four conditions and the machine gate before
any truth, the code identity, the one reader, the shared metric function, locks, manifest, reevaluation, the spy (C23.6-5), the D107 classes of a written cell
with holdout_common's records, the method arm's decision certificate (D112 (a)), PQ's list, release and acceptance (C22.H7-1) and the three functions of PQH's
replay gate.  Synthetic world only (_holdout_world.py: a temporary git repository with synthetic sealed lists and BZ's synthetic unseal record, synthetic
subjects OASIS_OAS1_09xx_MR1, NIfTI labels, HB- and XB-shaped cells and PQ outputs; SG's and holdout_common's real functions); no real subject file is opened.
C23.12-7 maps to these tests in runs/20261107-ev/实现说明.md §8."""
import builtins
import copy
import io
import json
import os
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256, load_scalar, save_scalar

import bayesreg.io.sealed as S
from bayesreg.eval import cert_eval, cli, metrics
from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as D
from bayesreg.eval.fields import read_disp
from bayesreg.io import unseal as U
from bayesreg.io.guard import SubjectCheck, SubjectSealed, load_dev_allowlist

from .. import _p0_pins as P0
from .. import _p1_decisions as P1
from . import _synthetic as syn
from ._holdout_world import BASELINE_KEYS, DEV_ID, H1_ID, H4_ID, MAIN, REPLAY_PASS, SUBJECTS, World, commit, git, units

REAL_LISTS = dict(REPO=S.REPO, SEALED_LISTS=S.SEALED_LISTS)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_LAST_SPY", None)
    w = World(tmp_path)
    w.patch(monkeypatch)
    yield w
    H.stop_spy()
    S._reset_cache()


def refused(world, stage, *, check_spy=True, **kwargs):
    """A refusal before any truth: no truth read through the reader, no lock, the spy stopped, and (review m6) the audit spy saw no open under DATA_ROOT."""
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(**kwargs)
    assert info.value.stage == stage, info.value
    assert world.reads == [], "a truth file was read"
    assert not Path(H.MASTER_LOCK).exists(), "the lock was written"
    assert H._SPY is None, "the spy was left on after a refusal"
    if check_spy and H._LAST_SPY is not None:
        data = str(world.data.resolve())
        assert not [e for e in H._LAST_SPY.events if e["path"].startswith(data)], "the audit spy saw an open under DATA_ROOT"
    return info.value


def repin_pairs(world, monkeypatch, rows):
    world.write_pairs(rows)
    commit(world.repo, "pairs")
    monkeypatch.setattr(H, "PAIRS_SHA256", file_sha256(world.repo/H.PAIRS_REL))


def table_of(world):
    return H.validate_arm_table(json.loads((world.repo/H.ARM_TABLE_REL).read_text()))


def evaluate(world, **kwargs):
    admission, context = world.open(**kwargs)
    return H.evaluate_holdout(admission, context), context


def pair_record(context, row):
    return json.loads((context["out"]/"pairs"/f"{row[3]}.json").read_text())


def big(world, name, row, resume=False):
    return world.tmp/"big"/name/(row[3]+(".resume" if resume else ""))/"disp.npz"


# ---- C23.12-2: each condition missing refuses, nothing of the truth read --------------------------------------------------------------------------------
def test_condition_0_decision_none_or_wrong(world, monkeypatch):
    monkeypatch.setattr(H, "TRUTH_READER_DECISION", None)
    refused(world, "gate", decision=None)
    for change in (dict(reading="不过（交 BZ）"), dict(result_sha256="0"*64),
                   dict(counts=dict(reader=dict(subjects=20, identical=19), metrics=dict(pairs=19, identical=19))),
                   dict(code_commit="0"*40), dict(code_commit="HEAD"), dict(decided_by=""), dict(review=" ")):      # review m8: commit and names
        refused(world, "gate", decision=dict(world.decision(), **change))
    monkeypatch.setattr(H, "TRUTH_GATE_PLAN_SHA256", None)
    refused(world, "gate")


def test_condition_0_pq_replay_gate(world):
    """Alignment item 6: (0) calls pqh_replay.replay_decision_problem() (here the world's stand-in) and checks the decision's code identity; a problem refuses
    the whole entry, nothing read (C22.H7-1 last item, D119)."""
    exc = refused(world, "pq_replay", pq_replay=dict(world.pq_replay(), reading="不过（交 BZ）"))
    assert "PQ_REPLAY_DECISION" in str(exc)
    refused(world, "pq_replay", pq_replay=dict(world.pq_replay(), code_commit="1"*40))


def test_the_replay_check_is_pqh_s_function_and_refuses_while_pqh_is_not_in_the_tree(monkeypatch):
    """The local stand-in of pqh_replay.replay_decision_problem (PQH not merged into 73723d7): same signature, a problem while the module is not in the tree;
    no second pin of PQ_REPLAY_DECISION in this module."""
    import inspect
    from bayesreg.eval import holdout as fresh
    assert not hasattr(fresh, "PQ_REPLAY_DECISION")
    params = inspect.signature(fresh.replay_decision_problem).parameters
    assert list(params) == ["decision", "repo"] and params["repo"].kind is inspect.Parameter.KEYWORD_ONLY
    if not (H.REPO/"experiments"/"pq"/"pqh_replay.py").is_file():
        assert "PQH not merged" in fresh.replay_decision_problem()
        with pytest.raises(H.HoldoutRefused) as info:
            fresh.check_pq_replay()
        assert info.value.stage == "pq_replay"


def test_condition_0_comes_before_a(world, monkeypatch):
    calls = []
    monkeypatch.setattr(H, "load_unseal", lambda path: calls.append(path))
    exc = refused(world, "gate", decision=dict(world.decision(), reading="x"))
    assert calls == [] and "TRUTH_READER_DECISION" in str(exc)
    refused(world, "pq_replay", pq_replay=dict(world.pq_replay(), reading="x"))
    assert calls == []


def test_code_identity_the_gates_commit_has_the_code_directories_of_head(world):
    """The global rule (decision_log df8541d; review2 m3): a code directory that changed after a gate's commit refuses at (0); records may be added."""
    decision, replay = world.decision(), world.pq_replay()
    (world.repo/H.REEVAL_DIR_REL/"some_authorization.json").write_text("{}")
    commit(world.repo, "an authorization: allowed")
    (world.repo/"experiments"/"pq").mkdir(parents=True)
    (world.repo/"experiments"/"pq"/"pq_stats.py").write_text("# changed\n")
    commit(world.repo, "a code change after the gate")
    exc = refused(world, "gate", decision=decision, pq_replay=replay)
    assert "rule two" in str(exc) and "experiments/pq/pq_stats.py" in str(exc)
    refused(world, "pq_replay", pq_replay=replay)                   # a fresh truth decision: now the replay decision's commit is the old one


def test_code_identity_an_uncommitted_change_in_experiments_refuses(world, monkeypatch):
    """Review2 N1: the clean-tree check covers experiments/ and scripts/ (the PQ arithmetic and the warp): an uncommitted change refuses, nothing read.  Since A5b
    rule two (gate_identity_problems, at (0)) refuses it first (an untracked file in a code directory); with the gates' check passed over here, the clean-tree
    check of (d) still refuses it."""
    (world.repo/"experiments"/"pq").mkdir(parents=True)
    (world.repo/"experiments"/"pq"/"pq_eval.py").write_text("# not committed\n")
    exc = refused(world, "gate")
    assert "experiments/pq/pq_eval.py" in str(exc)
    monkeypatch.setattr(H, "gate_identity_problems", lambda commit, *, repo=None, run=None: (dict(stub=True), []))
    exc = refused(world, "code")
    assert "experiments/" in str(exc)
    (world.repo/"experiments"/"pq"/"pq_eval.py").unlink()
    (world.repo/"scripts").mkdir()
    (world.repo/"scripts"/"evaluate_l2r_style.py").write_text("# not committed\n")
    refused(world, "code")


def test_helper_modules_must_come_from_this_code_tree(monkeypatch, tmp_path):
    """Review2 N1: a module of the same flat name cached from elsewhere (here a stand-in in sys.modules) is refused."""
    H._pq_modules()
    fake = types.ModuleType("pq_stats")
    fake.__file__ = str(tmp_path/"pq_stats.py")
    monkeypatch.setitem(sys.modules, "pq_stats", fake)
    with pytest.raises(H.HoldoutRefused) as info:
        H._pq_modules()
    assert info.value.stage == "code" and "pq_stats" in str(info.value)


def test_condition_a_no_unseal_record(world):
    git(world.repo, "rm", "-q", H.UNSEAL_REL)
    commit(world.repo, "no record")
    exc = refused(world, "unseal")
    assert "UnsealRefused" in str(exc)


@pytest.mark.parametrize("how", ["not_in_the_decision_log", "uncommitted", "paper1_list"])
def test_condition_a_the_real_loader_refuses_a_record_that_does_not_hold(world, how):
    log = world.repo/U.DECISION_LOG_REL
    if how == "not_in_the_decision_log":
        log.write_text("| 2026-11-07 | nothing about the record | x |\n", encoding="utf-8")
        commit(world.repo, "log without the record")
    elif how == "uncommitted":
        (world.repo/H.UNSEAL_REL).write_text((world.repo/H.UNSEAL_REL).read_text()+"\n", encoding="utf-8")
    else:
        world.write_record(sealed_list="H1")                         # (v): H1-H4 are never unsealed
        commit(world.repo, "a record naming H1")
    refused(world, "unseal")


def test_condition_a_a_pair_with_a_subject_off_the_holdout_list(world, monkeypatch):
    repin_pairs(world, monkeypatch, ((SUBJECTS[0], SUBJECTS[1]), (SUBJECTS[2], DEV_ID)))
    exc = refused(world, "admission")
    assert "not on the sealed list holdout_B_v1" in str(exc)


@pytest.mark.parametrize("other", ["H1", "H4"])
def test_condition_a_h1_h4_never(world, monkeypatch, other):
    if other == "H1":
        repin_pairs(world, monkeypatch, ((SUBJECTS[0], SUBJECTS[1]), (SUBJECTS[2], H1_ID)))
    else:
        world.write_lists(h4=(H4_ID, SUBJECTS[3]))                  # a hold-out subject also on H4
        commit(world.repo, "H4 holds a hold-out subject")
        world.pin_lists(monkeypatch)
    exc = refused(world, "admission")
    assert "H1-H4 never" in str(exc)


def test_condition_a_check_subject_refusal_stops(world, monkeypatch):
    monkeypatch.setattr(H, "check_subject", lambda subject, allowlist: SubjectCheck(subject, step="sealed", item="test", message="refused (test)"))
    with pytest.raises(SubjectSealed, match="check_subject"):
        world.open()
    assert world.reads == [] and not Path(H.MASTER_LOCK).exists() and H._SPY is None


def test_condition_a_the_guard_admits_through_the_bound_allowlist_only(world):
    unseal = U.load_unseal(world.repo/H.UNSEAL_REL)
    bound = U.bind_unseal(load_dev_allowlist(world.repo/H.ALLOWLIST_REL), unseal)
    rows, _ = H.admit_subjects(list(SUBJECTS), unseal, bound)
    assert all(r["admitted_by_unseal"] for r in rows)
    with pytest.raises(SubjectSealed):
        H.admit_subjects([SUBJECTS[0]], unseal, load_dev_allowlist(world.repo/H.ALLOWLIST_REL))     # not bound
    with pytest.raises(SubjectSealed):
        H.admit_subjects([SUBJECTS[0]], unseal, copy.copy(bound))                                     # a copy of the bound one


def test_condition_a_pair_not_on_the_list_or_reversed(world):
    rows, _ = H.load_pairs(world.repo)
    with pytest.raises(H.HoldoutRefused, match="other direction"):
        H.check_pair_row(rows[0][2], rows[0][1], rows)
    with pytest.raises(H.HoldoutRefused, match="not a row"):
        H.check_pair_row(rows[0][1], rows[1][2], rows)
    world.write_stage(world.rows[0], reversed_=True)                 # the stage manifest names the pair in the other direction
    world.write_manifest()
    refused(world, "pairs")


def test_condition_a_pairs_file_pinned(world, monkeypatch):
    monkeypatch.setattr(H, "PAIRS_SHA256", "0"*64)
    refused(world, "pairs")


def test_condition_b_prereg_and_arm_table_pinned(world, monkeypatch):
    monkeypatch.setattr(H, "PREREG_SHA256", None)
    refused(world, "prereg")
    monkeypatch.setattr(H, "PREREG_SHA256", world.prereg_sha)
    monkeypatch.setattr(H, "ARM_TABLE_SHA256", None)
    refused(world, "arms")
    monkeypatch.setattr(H, "ARM_TABLE_SHA256", world.table_sha)
    (world.repo/H.PREREG_REL).write_text("# a preregistration that does not name the arm table\n")
    commit(world.repo, "prereg without the table")
    monkeypatch.setattr(H, "PREREG_SHA256", H.sha_bytes((world.repo/H.PREREG_REL).read_bytes()))
    refused(world, "arms")


def test_condition_b_a_missing_cell_stops_every_pair(world):
    (world.record_dir("ants", world.rows[1])/"result.json").unlink()
    exc = refused(world, "arms")
    assert "ants" in str(exc)


@pytest.mark.parametrize("status", ["admission_refused", "code_not_clean", "list_error", "machine_refused", "needs_lead_rejudgment", "something_unmapped"])
def test_condition_b_refusals_are_not_written(world, status):
    world.write_cell(world.arm("fga"), world.rows[0], status=status, field=False)
    refused(world, "arms")


def test_condition_b_method_provenance_rejected_is_not_written(world):
    world.write_cell(world.arm("method"), world.rows[0], status="hx_provenance_rejected", field=False)
    refused(world, "arms")


def test_condition_b_method_needs_its_review_record_finalized(world):
    """C22.H1, D111 (review I7; alignment item 1: review/review.json, finalized true)."""
    world.write_review(world.rows[0], finalized=False)
    exc = refused(world, "arms")
    assert "not finalized" in str(exc)
    (world.cell("method", world.rows[0])/"review"/"review.json").unlink()
    exc = refused(world, "arms")
    assert "review.json" in str(exc)


def test_condition_b_machine_gate(world):
    refused(world, "machine", run=units("tag-hb-x-ants-p1.service"))
    refused(world, "machine", run=units("seq-vp-chain.service"))


def test_arm_table_v3_forms(world):
    """Alignment items 1-3: record and field places apart; holdout_common's names; review I1, I3, I4 (pq required, BL1 on every arm, the decision
    certificate); review2 m2 (no path under the data root)."""
    table = json.loads((world.repo/H.ARM_TABLE_REL).read_text())
    data = str(world.data)
    for change in (lambda t: t.update(pq=None), lambda t: t["pq"].update(small_root="relative"), lambda t: t["pq"].update(big_root=data+"/pq"),
                   lambda t: t["arms"][0].update(cert_mode="full", jacobian_config="C07-2"), lambda t: t["arms"][1].update(jacobian_config="C07-2"),
                   lambda t: t["arms"][0].update(decision_certificate=None), lambda t: t["arms"][0]["decision_certificate"].update(final="output/u.npz"),
                   lambda t: t["arms"][1].update(decision_certificate=dict(final="disp.npz", cert="result.json")),
                   lambda t: t["arms"][0].update(review_record="review/review.json"), lambda t: t["arms"][0]["review_record"].update(final=[]),
                   lambda t: t["arms"][1]["terminal"]["classes"].update(implementation_error=["x"]),
                   lambda t: t["arms"][0]["terminal"]["classes"].update(implementation_error=["implementation_error"]),
                   lambda t: t["arms"][1].update(record_dir=data+"/{arm}/{pair}"), lambda t: t["arms"][1].update(record_dir="/x/{arm}"),
                   lambda t: t["arms"][1]["field"].update(root=data+"/{arm}"), lambda t: t["arms"][1]["field"].update(root="/x/{arm}/{pair}"),
                   lambda t: t["arms"][1].update(field=dict(dir="/x/{arm}")), lambda t: t["arms"][0].update(field=dict(file_key="disp.file")),
                   lambda t: t["arms"][1]["authorization"].update(records=data+"/{pair}"), lambda t: t["arms"][1]["authorization"].pop("tool"),
                   lambda t: t["arms"][1].update(name="eval"), lambda t: t["arms"][1].update(name="pq"), lambda t: t.update(stage_root=data+"/stage"),
                   lambda t: t["arms"][1]["terminal"].update(files=[]), lambda t: t["arms"][1].update(export_check=dict(file="result.json", key="x")),
                   lambda t: t["arms"][1].update(cell_dir="/x/{pair}"), lambda t: t.update(schema="bayesreg.holdout.arm_table.v2")):
        bad = copy.deepcopy(table)
        change(bad)
        with pytest.raises(H.HoldoutRefused) as info:
            H.validate_arm_table(bad)
        assert info.value.stage == "arms"
    assert H.validate_arm_table(table) == table


def test_condition_c_manifest_committed_and_complete(world):
    (world.repo/H.MANIFEST_REL).write_text((world.repo/H.MANIFEST_REL).read_text()+"\n")       # changed against HEAD
    refused(world, "manifest")
    commit(world.repo, "restore")
    files = world.needed()
    world.write_manifest(drop=files[0])
    refused(world, "manifest")
    world.write_manifest(drop=next(p for p in files if "pq_big" in p))     # a PQ output this entry reads is an input too (C22.H7-1, Ev1)
    refused(world, "manifest")
    world.write_manifest(drop=str(world.pq_list()))                      # PQ's list itself
    refused(world, "manifest")
    stray = world.tmp/"stray.json"
    stray.write_text("{}")
    world.write_manifest(extra_input=stray)                           # a file outside C23.12-4
    refused(world, "manifest")
    extra = world.record_dir("ants", world.rows[0])/"log.txt"
    extra.write_text("log")
    world.write_manifest(extra_input=extra)                           # a cell file this entry does not read: it belongs to the record-only file
    refused(world, "manifest")


def test_condition_c_pq_release_or_bz_ii_is_needed(world):
    """Alignment item 5: without PQ's release (or BZ's (ii)) PQ's outputs cannot be found: refused before any truth; a (ii) record that is not committed or
    not written by BZ refuses too."""
    git(world.repo, "rm", "-q", H.PQ_RELEASE_SHA_REL)
    commit(world.repo, "no release")
    world.write_manifest()
    exc = refused(world, "manifest")
    assert "neither PQ's release" in str(exc)
    world.bz_ii = "runs/holdout_B_v1/none.json"
    world.write_manifest()
    refused(world, "manifest")
    world.bz_ii = None
    world.write_manifest(header=["# x_sha256 "+"1"*64])
    refused(world, "manifest")


def test_condition_c_an_uncommitted_manifest(world):
    git(world.repo, "rm", "-q", "--cached", H.MANIFEST_REL)
    git(world.repo, "commit", "-q", "-m", "manifest untracked")       # the file stays in the tree, untracked
    assert (world.repo/H.MANIFEST_REL).is_file()
    refused(world, "manifest")


def test_condition_c_the_record_only_file_is_committed_with_the_manifest_and_not_read(world, monkeypatch):
    opened = []
    real_open = Path.read_bytes

    def spy(self):
        opened.append(str(self))
        return real_open(self)
    monkeypatch.setattr(Path, "read_bytes", spy)
    world.open()
    assert not any(p.endswith(H.RECORDS_REL) for p in opened)
    monkeypatch.setattr(Path, "read_bytes", real_open)
    Path(H.MASTER_LOCK).unlink()
    (world.repo/H.MANIFEST_REL).write_text((world.repo/H.MANIFEST_REL).read_text()+"\n")
    commit(world.repo, "the manifest alone")                        # the record-only file was committed earlier
    refused(world, "manifest", out=Path(H.EVAL_DIR)/"out2")
    (world.repo/H.RECORDS_REL).unlink()
    commit(world.repo, "no record-only file")
    refused(world, "manifest", out=Path(H.EVAL_DIR)/"out3")


def test_condition_c_a_record_read_in_b_must_have_its_listed_sha(world):
    path = world.record_dir("cadam", world.rows[0])/"result.json"
    world.write_manifest()
    path.write_text(path.read_text().replace("}", ', "note": "x"}', 1))
    refused(world, "manifest")


def test_condition_c_a_field_listed_at_another_sha_than_its_terminal_record(world):
    """A field the terminal record names with its sha256 (XB's disp) must be listed at that sha256."""
    path = world.record_dir("fga", world.rows[0])/"result.json"
    doc = json.loads(path.read_text())
    doc["disp"]["sha256"] = "0"*64
    path.write_text(json.dumps(doc))
    world.write_manifest()
    exc = refused(world, "manifest")
    assert "names sha256" in str(exc)


def test_condition_d_lock_once(world):
    H.take_lock(H.MASTER_LOCK, dict(earlier=True))
    with pytest.raises(H.HoldoutRefused) as info:
        world.open()
    assert info.value.stage == "once" and world.reads == []


def test_out_dir_rules(world):
    (world.results/"exists").mkdir(parents=True)
    refused(world, "output", out=world.results/"exists")
    refused(world, "output", out=world.record_dir("ants", world.rows[0])/"eval")
    refused(world, "output", out=world.tmp/"pq_small"/world.rows[0][3]/"eval")
    refused(world, "output", out=world.tmp/"records"/"ants"/"eval_out")         # under an arm's record root: the spy watches it
    refused(world, "output", out=world.tmp/"big"/"ants"/"eval_out")             # under its field root
    refused(world, "output", out=world.tmp/"pq_big"/"eval_out")
    refused(world, "output", out=world.data/"eval_out")
    refused(world, "output", out=H.REPO/"some_out")


def test_lock_directories_under_a_watched_root_refuse(world, monkeypatch):
    """Review2 m4: the master lock is written last, under the spy; its directory under a watched root refuses before it."""
    monkeypatch.setattr(H, "EVAL_DIR", world.tmp/"records"/"ants"/"eval")
    exc = refused(world, "output")
    assert "lock directories" in str(exc)


def test_the_imported_package_must_be_the_checked_tree(tmp_path):
    """Review m7, review2 N1: bayesreg and the helper modules this process imported are those of the checked tree."""
    assert H.imported_from(H.REPO) is True and H.imported_from(tmp_path) is False
    origins = H.helper_origins()
    assert set(origins) == {"pq_core", "pq_eval", "pq_stats", "evaluate_l2r_style", "holdout_common", "hi_core"}
    assert all(H._inside(p, H.REPO) for p in origins.values())


# ---- all four: the evaluation, equal to bayesreg.eval.cli -----------------------------------------------------------------------------------------------
def clean(c):
    skip = {"seconds"}
    return {k: (v if k != "cells" or v is None else {kk: vv for kk, vv in v.items() if kk not in skip}) for k, v in c.items() if k not in skip}


def test_full_evaluation_and_bitwise_equal_to_cli(world, tmp_path, monkeypatch):
    admission, context = world.open()
    assert Path(H.MASTER_LOCK).is_file() and world.reads == []
    line = json.loads(Path(H.MASTER_LOCK).read_text())
    assert set(line) >= {"unseal_sha256", "prereg_sha256", "manifest_sha256", "manifest_commit", "code_commit", "utc", "pid"}
    assert H.evaluate_holdout(admission, context) == 0
    assert len(world.reads) == 4*2                                   # each subject once, seg35 and seg4
    row = world.rows[0]
    record = pair_record(context, row)
    assert record["complete"] and record["seg4_equals_G"]["fixed"]["equal"] and set(record["arms"]) == {"method", *MAIN}
    assert record["truth"]["fixed"]["seg35"]["sha256"] and record["identity"]["metrics"]["seg35"] is not None
    assert all(record["arms"][a]["certificate"]["configuration"] == "BL1" for a in ("method", *MAIN))      # every arm shares the BL1 certificate
    assert record["arms"]["method"]["review"]["value"] is True and record["arms"]["method"]["terminal"].endswith("terminal.json")
    assert record["arms"]["ants"]["terminal"].endswith("result.json")
    summary = json.loads((context["out"]/"holdout_eval.json").read_text())
    assert summary["evaluated"] == [r[3] for r in world.rows] and summary["failed"] == [] and summary["record"]["pq_replay"]["reading"] == REPLAY_PASS
    assert summary["record"]["code"]["dirs"] == list(H.CLEAN_DIRS) and "pq_stats" in summary["record"]["code"]["helper_modules"]
    # the same fields and truth through bayesreg.eval.cli (truth from a development-style pair.npz holding the same labels).  The synthetic ids are sealed in
    # the world; the cli runs against the real pinned lists, where they are not
    monkeypatch.setattr(S, "REPO", REAL_LISTS["REPO"])
    monkeypatch.setattr(S, "SEALED_LISTS", REAL_LISTS["SEALED_LISTS"])
    S._reset_cache()
    data = tmp_path/"cli_data"
    fixed, moving = row[1], row[2]
    arrays = dict(fixed=np.zeros(syn.SHAPE), moving=np.zeros(syn.SHAPE), fixed_seg35=world.labels[fixed]["seg35"], moving_seg35=world.labels[moving]["seg35"],
                  fixed_seg4=world.labels[fixed]["seg4"], moving_seg4=world.labels[moving]["seg4"], initial_u=np.zeros(syn.SHAPE+(3,)))
    syn.write_pair(data, row[3], fixed, moving, arrays)
    allowlist = syn.write_allowlist(tmp_path/"allow.json", subjects=(fixed, moving))
    for arm in ("configc", "ants"):
        code = cli.main([str(big(world, arm, row).parent), "--pairs-stage", syn.STAGE, "--pair", row[3], "--method", arm, "--run-id", arm, "--out", "m.json",
                         "--long", str(tmp_path/f"long_{arm}.csv"), "--out-dir", str(tmp_path/"cli_out"), "--data-root", str(data), "--allowlist", str(allowlist)])
        assert code == 0
        theirs = json.loads((tmp_path/"cli_out"/arm/"m.json").read_text())
        mine = record["arms"][arm]
        assert mine["metrics"] == theirs["metrics"] and mine["per_label"] == theirs["per_label"]
    # BL1's certificate: the same as certify_field_bl1 on u = h d (the configuration is tested against the archive in test_bl1_record_is_the_bl1_certificate)
    d = read_disp(big(world, "ants", row))[0]
    direct = json.loads(H.dumps(H.certify_field_bl1(H.d_to_u(d, world.grid.spacing), world.grid, brain_nodes=world.labels[fixed]["seg4"] > 0)))
    mine = {k: v for k, v in clean(record["arms"]["ants"]["certificate"]).items() if k not in ("certified_field", "u_equals_run_u_bitwise")}
    assert mine == clean(direct)


def test_the_method_decision_certificate_is_on_final_u_with_certify_field_defaults(world):
    """D112 (a), C22.H1-6 (review I3; the lead's ruling (4)): output/final.npz's u, cert_eval.certify_field(mode='full', tile_size=65536) with its defaults
    (split_cells), always on the method arm, recorded only; state_hash against cert.json; u against h d."""
    code, context = evaluate(world)
    assert code == 0
    row = world.rows[0]
    mine = pair_record(context, row)["arms"]["method"]["decision_certificate"]
    u = world.method_u(row)
    direct = json.loads(H.dumps(cert_eval.certify_field(u, world.grid, mode="full", tile_size=65536, brain_nodes=world.labels[row[1]]["seg4"] > 0)))
    assert clean(mine["certificate"]) == clean(direct) and mine["certificate"]["cells"] is not None
    assert mine["certificate"]["config"]["tile_size"] == 65536 and mine["certificate"]["mode"] == "full"
    assert mine["state_hash_matches_cert_json"] is True and mine["certificate_state_hash_matches_u"] is True and mine["u_equals_h_d_bitwise"] is True
    assert mine["final"]["u_sha256"] == H.array_sha256(u)
    summary = D.certificate_summary(D.collect([context["out"]]), world.table, world.rows)       # the method's certificate statistics: this one only
    assert summary["method"]["decision_configuration"]["n"] == 2 and summary["method"]["decision_configuration"]["passed"] == \
        sum(pair_record(context, r)["arms"]["method"]["decision_certificate"]["certificate"]["passed"] for r in world.rows)
    assert summary["ants"]["jacobian_only_bl1"]["n"] == 2 and "decision_configuration" not in summary["ants"]


def test_second_run_refused_and_first_files_unchanged(world):
    code, context = evaluate(world)
    assert code == 0
    before = {p: p.read_bytes() for p in context["out"].rglob("*") if p.is_file()}
    world.reads.clear()
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(out=Path(H.EVAL_DIR)/"out2")
    assert info.value.stage == "once" and world.reads == []
    assert {p: p.read_bytes() for p in context["out"].rglob("*") if p.is_file()} == before


def test_reader_equals_load_scalar_and_checks_the_grid(world, monkeypatch):
    admission, context = world.open()
    H.stop_spy()                                                      # the reader is called directly here (evaluate_pair lets the pair's truth through the spy)
    row = world.rows[0]
    metadata = json.loads((Path(world.table["stage_root"])/row[3]/"manifest.json").read_text())
    H.take_pair_lock(admission, row, context["line"])
    values, record = H.read_truth(row[1], admission=admission, grid_metadata=metadata, pair_name=row[3])
    for kind in ("seg35", "seg4"):
        direct = load_scalar(world.data/row[1]/H.LABEL_FILES[kind], labels=True).values
        assert np.array_equal(np.ascontiguousarray(direct, dtype=np.float64).view(np.uint64), values[kind].view(np.uint64))
        assert record[kind]["stored_dtype"] == "float64"
    for key, value in (("spacing_xyz", [1., 1., 2.]), ("origin_xyz", [0., 0., 1.]), ("shape_zyx", [18, 20, 23])):
        with pytest.raises(H.TruthMismatch) as info:
            H.read_truth(row[2], admission=admission, grid_metadata=dict(metadata, **{key: value}), pair_name=row[3])
        assert info.value.stage == "truth_grid"


def test_a_truth_volume_off_the_grid_stops_the_evaluation(world):
    """C23.12-4 "不同就停" (review m3): the evaluation stops (exit 4) instead of recording one failed pair."""
    from jd_curl.three_d.grid import make_grid_3d
    subject = world.rows[1][1]
    grid = make_grid_3d(nx=syn.SHAPE[2], ny=syn.SHAPE[1], nz=syn.SHAPE[0], hx=1., hy=1., hz=2., unit="mm")
    path = world.data/subject/H.LABEL_FILES["seg35"]
    path.unlink()
    save_scalar(path, world.labels[subject]["seg35"], grid)
    code, context = evaluate(world)
    assert code == 4
    stopped = json.loads((context["out"]/"stopped.json").read_text())
    assert stopped["pair"] == world.rows[1][3] and "TruthMismatch" in stopped["reason"]
    assert pair_record(context, world.rows[0])["complete"]


def test_reader_refuses_without_a_genuine_admission(world):
    admission, context = world.open()
    row = world.rows[0]
    metadata = json.loads((Path(world.table["stage_root"])/row[3]/"manifest.json").read_text())
    with pytest.raises(H.HoldoutRefused, match="pair lock"):
        H.read_truth(row[1], admission=admission, grid_metadata=metadata, pair_name=row[3])
    H.take_pair_lock(admission, row, context["line"])
    for clone in (pickle.dumps, copy.deepcopy):
        with pytest.raises(TypeError):
            clone(admission)                                         # cannot even be pickled or deep-copied (its record is a mapping proxy)
    for fake in (copy.copy(admission), H.Admission(admission.mode, admission.subjects, admission.pairs, admission.record)):
        with pytest.raises(H.HoldoutRefused, match="not an admission issued"):
            H.read_truth(row[1], admission=fake, grid_metadata=metadata, pair_name=row[3])
    with pytest.raises(H.HoldoutRefused, match="admitted pair"):
        H.read_truth(world.rows[1][1], admission=admission, grid_metadata=metadata, pair_name=row[3])
    assert world.reads == []


def test_a_changed_listed_file_stops_and_keeps_what_was_written(world):
    admission, context = world.open()
    path = big(world, "fsyn", world.rows[1])
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    assert H.evaluate_holdout(admission, context) == 4
    assert pair_record(context, world.rows[0])["complete"]
    second = pair_record(context, world.rows[1])
    assert not second["complete"] and "manifest_changed" in second["error"] and set(second["arms"]) == {"method", "configc", "ants", "fga"}
    assert json.loads((context["out"]/"stopped.json").read_text())["pair"] == world.rows[1][3]


def test_partial_record_kept_on_an_evaluation_error(world, monkeypatch):
    admission, context = world.open()
    original = H.score_cell

    def flaky(info, grid, truth, **kwargs):
        if "fga" in str(info.run_dir) and world.rows[0][3] in str(info.run_dir):
            raise MemoryError("synthetic")
        return original(info, grid, truth, **kwargs)
    monkeypatch.setattr(H, "score_cell", flaky)
    assert H.evaluate_holdout(admission, context) == 1
    first = pair_record(context, world.rows[0])
    assert not first["complete"] and "MemoryError" in first["error"] and set(first["arms"]) == {"method", "configc", "ants"}
    assert pair_record(context, world.rows[1])["complete"]


def test_entry_calls_the_shared_metric_function(world, monkeypatch):
    admission, context = world.open()
    seen = []
    original = H.score_cell

    def spy(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(H, "score_cell", spy)
    assert H.evaluate_holdout(admission, context) == 0
    assert len(seen) == 2*6                                          # every output cell through score_cell


# ---- PQ (C22.H6, C22.H7-1; alignment items 5, 6) -------------------------------------------------------------------------------------------------
def test_pq_acceptance_is_computed_in_the_entry_with_experiments_pq_functions(world):
    """The acceptance numbers equal pq_stats.score_record on the entry's own truth arrays and the pair's d; T~_M is the evaluation's order-0 pull-back."""
    code, context = evaluate(world)
    assert code == 0
    row = world.rows[0]
    pq_record = pair_record(context, row)["pq"]
    assert pq_record["state"] == "present" and pq_record["complete"] and pq_record["ev1"]["passed"] and pq_record["ev2"]["passed"]
    assert pq_record["ev2"]["key"] == "sampling.d_sha256"
    pq_core, pq_eval, pq_stats = H._pq_modules()
    small, big_ = world.pq_dirs(row)
    with np.load(big_/"pq.npz") as z:
        q = np.array(z["q"])
    with np.load(big_/"baselines.npz") as z:
        h = np.array(z["h"])
    d = read_disp(world.cell("method", row)/"output"/"disp.npz")[0]
    T_F = world.labels[row[1]]["seg35"]
    T_tilde = metrics.warp_labels(world.labels[row[2]]["seg35"], d)
    assert np.array_equal(T_tilde, pq_core.warp_labels_l2r(world.labels[row[2]]["seg35"], d))       # the development set's warp: the same labels
    e = T_F != T_tilde
    omega = (T_F > 0) | (T_tilde > 0)
    scores = pq_record["acceptance"]["scores"]["omega_e"]
    assert scores["q"]["ause"] == pq_stats.score_record([q[omega]], e[omega])["ause"]
    assert scores["h"]["ause"] == pq_stats.score_record([h[omega]], e[omega])["ause"]
    assert set(scores) == {"q", *D.PQ_BASELINES, "G_minus"} and set(pq_record["acceptance"]["scores"]) == set(pq_eval.MASKS)
    assert pq_record["acceptance"]["n_errors"]["omega_e"] == int(e[omega].sum())
    assert tuple(BASELINE_KEYS) == tuple(pq_core.BASELINE_KEYS)


def test_pq_pair_acceptance_is_pqh_s_interface(world):
    """Alignment item 6: pq_pair_acceptance(*, truth_fixed, truth_moving, d, pq, baselines, spacing_xyz) gives what PQH's reference (pq_eval.evaluate_pair's
    middle on arrays) gives: every mask, score and field, and e, e4."""
    pq_core, pq_eval, pq_stats = H._pq_modules()
    rng = np.random.default_rng(3)
    fixed, moving = world.labels[SUBJECTS[0]], world.labels[SUBJECTS[1]]
    d = world.field("method", world.rows[0])
    pq = dict(q=rng.random(syn.SHAPE), q4=rng.random(syn.SHAPE))
    base = {k: rng.random(syn.SHAPE).astype(np.float32) for k in BASELINE_KEYS}
    mine = H.pq_pair_acceptance(truth_fixed=fixed, truth_moving=moving, d=d, pq=pq, baselines=base, spacing_xyz=(1., 1., 1.))
    T_F, T_tilde = np.asarray(fixed["seg35"]), pq_core.warp_labels_l2r(moving["seg35"], d)
    e, e4 = T_F != T_tilde, np.asarray(fixed["seg4"]) != pq_core.warp_labels_l2r(moving["seg4"], d)
    masks = dict(omega_e=(T_F > 0) | (T_tilde > 0), band=pq_eval.band_mask(T_F, T_tilde, (1., 1., 1.)), whole=np.ones(T_F.shape, dtype=bool))
    for mask_name, mask in masks.items():
        for name, keys in pq_eval.score_keys(pq, base).items():
            ref = pq_stats.score_record(pq_eval.masked(keys, mask), e[mask])
            for field in ("ause", "auroc", "largest_tie_share"):
                assert mine["scores"][mask_name][name][field] == ref[field], (mask_name, name, field)
    assert np.array_equal(mine["e"], e) and np.array_equal(mine["e4"], e4)


def test_pq_b_class_is_pq_stats_b_class_with_the_interval_passed_in():
    """Alignment item 6: pq_b_class(ause_table, *, interval) with the development block_interval gives pq_stats.b_class's Δ, LB, R and pass bit for bit."""
    _, _, pq_stats = H._pq_modules()
    rng = np.random.default_rng(8)
    table = dict(q=list(.1+rng.normal(0, .01, 19)))
    for k in pq_stats.JUDGED_BASELINES:
        table[k] = list(np.asarray(table["q"])+.01+rng.normal(0, .005, 19))
    index = np.arange(19)
    theirs = pq_stats.b_class(table, index)
    mine = H.pq_b_class(table, interval=lambda delta: pq_stats.block_interval(delta, index))
    assert mine["passed"] == theirs["passed"] and mine["failed"] == theirs["failed"]
    for k in pq_stats.JUDGED_BASELINES:
        for field in ("delta", "LB", "R", "passed"):
            assert mine["baselines"][k][field] == theirs["baselines"][k][field], (k, field)
    assert H.pq_b_class(dict(table, h=[None]*19), interval=lambda delta: dict(lo=0.))["judged"] is False


def test_pq_dev_truth_reads_development_subjects_through_the_one_reader(world, monkeypatch):
    """Alignment item 6: pq_dev_truth(subject) -> {seg35, seg4, files}: the development admission (as the truth-reader gate's), then read_truth; a hold-out id
    is refused.  Synthetic labels under the development ids, in tmp_path."""
    subjects = H.DEV_SUBJECTS[:2]
    for k, s in enumerate(subjects):
        for kind in ("seg35", "seg4"):
            save_scalar(world.data/s/H.LABEL_FILES[kind], world.labels[SUBJECTS[k]][kind], world.grid)
    metadata = json.loads((Path(world.table["stage_root"])/world.rows[0][3]/"manifest.json").read_text())
    name = H.PAIR_NAME.format(fixed=subjects[0], moving=subjects[1])

    def fake_admit(*, data_root=None, repo=None, allowlist_path=None):
        admission = H._issue("truth_gate", subjects, ((0, subjects[0], subjects[1], name),), dict(test=True))
        return admission, [dict(row=(0, subjects[0], subjects[1], name), metadata=metadata)]
    monkeypatch.setattr(H, "admit_truth_gate", fake_admit)
    monkeypatch.setattr(H, "_PQ_DEV", None)
    out = H.pq_dev_truth(subjects[1])
    direct = load_scalar(world.data/subjects[1]/H.LABEL_FILES["seg35"], labels=True).values
    assert np.array_equal(out["seg35"], np.asarray(direct, dtype=np.float64)) and set(out["files"]) == {"seg35", "seg4"}
    assert world.reads and all(subjects[1] in r for r in world.reads)
    with pytest.raises(H.HoldoutRefused):
        H.pq_dev_truth(SUBJECTS[0])


def test_pq_acceptance_takes_arrays_only(world, monkeypatch):
    """C22.H7-1, C22.H8: the acceptance function opens no file."""
    H._pq_modules()
    rng = np.random.default_rng(1)
    T_F, T_tilde = world.labels[SUBJECTS[0]]["seg35"], world.labels[SUBJECTS[1]]["seg35"]
    pq = dict(q=rng.random(syn.SHAPE), q4=rng.random(syn.SHAPE))
    base = {k: rng.random(syn.SHAPE).astype(np.float32) for k in BASELINE_KEYS}

    def no(*args, **kwargs):
        raise AssertionError(f"a file was opened: {args[:1]}")
    for owner, name in ((builtins, "open"), (io, "open"), (os, "open"), (np, "load")):
        monkeypatch.setattr(owner, name, no)
    out = H.pq_acceptance(pq, base, T_F, T_tilde, world.labels[SUBJECTS[0]]["seg4"], world.labels[SUBJECTS[1]]["seg4"], (1., 1., 1.))
    assert out["n_errors"]["omega_e"] > 0


def test_pq_inputs_against_the_step8_input_lock(world):
    """C22.H4-2 Hm2 after the lock (revision 7a m2), by this entry (the lead's ruling (10) on the PQH review I9): pq.json's probs36, structure_groups and
    provenance sha256 (and the method's field_0600.npz) equal the input lock's rows -> passed, the numbers are computed."""
    code, context = evaluate(world)
    assert code == 0
    for row in world.rows:
        check = pair_record(context, row)["pq"]["input_lock"]
        assert check["passed"] is True and len(check["rows"]) == 7 and all(r["same"] for r in check["rows"])
        assert check["lock"]["path"] == H.INPUT_LOCK_REL and pair_record(context, row)["pq"]["acceptance"] is not None
    rows = D.pq_rows(D.collect([context["out"]]), world.table, world.rows)
    assert D.decide_f4(rows, D.draw_idx(n=2, b=100), n_prime=2)["eval_side"]["input_lock"] == []


@pytest.mark.parametrize("how", ["prediction", "field", "no_lock"])
def test_pq_inputs_not_the_input_lock_s_stop_that_pairs_pq_only(world, how):
    """A different sha256 (one subject's structure_groups, or one pair's field_0600.npz), or no committed lock: the pair's PQ stops, F4 reads 不判（评测侧）;
    F1-F3 go on (the lead's ruling (10), 11-09)."""
    if how == "prediction":
        world.write_input_lock(**{world.rows[0][2]: "structure_groups.npz"})
    elif how == "field":
        world.write_input_lock(**{world.rows[0][3]: "field_0600.npz"})
    else:
        git(world.repo, "rm", "-q", H.INPUT_LOCK_REL)
    commit(world.repo, "the input lock changed")
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    first, second = pair_record(context, world.rows[0]), pair_record(context, world.rows[1])
    assert first["complete"] and set(first["arms"]) == {"method", *MAIN}
    assert first["pq"]["input_lock"]["passed"] is False and first["pq"]["acceptance"] is None and first["pq"]["complete"] is True
    assert second["pq"]["input_lock"]["passed"] is (how != "no_lock")
    rows = D.pq_rows(D.collect([context["out"]]), world.table, world.rows)
    out = D.decide_f4(rows, D.draw_idx(n=2, b=100), n_prime=2)
    assert out["reading"] == D.F4_EVAL and world.rows[0][3] in out["eval_side"]["input_lock"]


def test_pq_ev2_a_different_d_stops_the_pairs_pq_only(world):
    world.write_pq(world.rows[0], files=("pq.json",), d_sha256="0"*64)
    world.release_pq()
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    record = pair_record(context, world.rows[0])
    assert record["complete"] and record["pq"]["ev2"]["passed"] is False and record["pq"]["acceptance"] is None and record["pq"]["complete"]
    assert pair_record(context, world.rows[1])["pq"]["ev2"]["passed"] is True


def test_pq_ev1_the_list_against_the_manifest(world):
    pq_npz = world.pq_dirs(world.rows[0])[1]/"pq.npz"
    world.release_pq(wrong=pq_npz)
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    assert pair_record(context, world.rows[0])["pq"]["ev1"]["passed"] is False and pair_record(context, world.rows[1])["pq"]["ev1"]["passed"] is True
    assert pair_record(context, world.rows[0])["complete"] and pair_record(context, world.rows[0])["pq"]["acceptance"] is not None     # F1-F3 go on


@pytest.mark.parametrize("how", ["note", "two_commits", "refusal_word"])
def test_pq_ev1_the_release_note_and_its_commit(world, how):
    """PQ's release (alignment item 5; pq_eval.release_note_problem's rule): one line 放行：<the list's sha256>, no word of refusal, one commit with the sha256
    file.  Not so: Ev1 fails for every pair, F1-F3 go on."""
    if how == "note":
        world.release_pq(note="放行："+"0"*64)
    elif how == "two_commits":
        note = world.repo/H.PQ_RELEASE_NOTE_REL
        text = note.read_text(encoding="utf-8")
        git(world.repo, "rm", "-q", H.PQ_RELEASE_NOTE_REL)
        commit(world.repo, "the release note removed")
        note.write_text(text, encoding="utf-8")
        commit(world.repo, "the release note again, in another commit than the list's sha256")
    else:
        sha = file_sha256(world.pq_list())
        world.release_pq(note=f"放行：{sha} not")
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    assert all(pair_record(context, r)["pq"]["ev1"]["passed"] is False for r in world.rows)
    assert pair_record(context, world.rows[0])["pq"]["ev1"]["release"]["problems"]


def test_pq_missing_for_a_pair_does_not_stop_f1_f3(world):
    (world.pq_dirs(world.rows[1])[1]/"pq.npz").unlink()
    world.release_pq()
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    record = pair_record(context, world.rows[1])
    assert record["complete"] and record["pq"]["state"] == "missing" and "baselines.npz" in record["pq"]["reason"]
    rows = D.pq_rows(D.collect([context["out"]]), world.table, world.rows)
    assert D.decide_f4(rows, D.draw_idx(n=2, b=100), n_prime=2)["excluded"][0]["pair"] == world.rows[1][3]


def test_pq_states_come_from_the_list(world):
    """Alignment item 5: PQ's files are found through the list, never by a path template: an unlisted file does not count, a listed pq_failed.json is a
    reason, a rerun directory counts."""
    table = table_of(world)
    cells = H.check_cells(table, world.rows, repo=world.repo)
    small, big_ = world.pq_dirs(world.rows[0])
    entries = {str(p): file_sha256(p) for p in world.pq_files()}
    states = H.pq_states(table, world.rows, cells, dict(method_side="released", entries=entries))
    assert all(s["state"] == "present" for s in states.values())
    states = H.pq_states(table, world.rows, cells, dict(method_side="released", entries={k: v for k, v in entries.items() if not k.startswith(str(small))}))
    assert states[world.rows[0][3]]["state"] == "missing"
    rerun = {k.replace(str(world.tmp/"pq_big"), str(world.tmp/"pq_big"/"rerun")): v for k, v in entries.items()}
    states = H.pq_states(table, world.rows, cells, dict(method_side="released", entries=rerun))
    assert states[world.rows[0][3]]["state"] == "present" and "/rerun/" in states[world.rows[0][3]]["files"]["pq.npz"]
    failed = dict({k: v for k, v in entries.items() if world.rows[1][3] not in k}, **{str(world.pq_dirs(world.rows[1])[0]/"pq_failed.json"): "a"*64})
    assert H.pq_states(table, world.rows, cells, dict(method_side="released", entries=failed))[world.rows[1][3]]["state"] == "pq_failed"
    outside = dict(entries, **{str(world.data/world.rows[0][3]/"pq.json"): "b"*64})
    assert H.pq_states(table, world.rows, cells, dict(method_side="released", entries=outside))[world.rows[0][3]]["state"] == "missing"


def test_pq_bz_choice_ii_reads_no_pq_file_and_f4_is_only_described(world):
    rel = H.PQ_METHOD_SIDE_REL                                       # the one path (the interface alignment revision 1 item 3; A5b item 9)
    (world.repo/rel).write_text(json.dumps(dict(schema=H.SCHEMA_PQ_METHOD_SIDE, written_by="lead on BZ's behalf", words="（合成测试）F4 只描述",
                                                time_local="11-09 8:00 PM CST", utc=H.utc(), reason="Hm6 (synthetic)", choice="ii"), ensure_ascii=False))
    commit(world.repo, "BZ (ii)")
    world.bz_ii = rel
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0
    opened = json.loads((context["out"]/"holdout_eval.json").read_text())["opened_files"]["events"]
    assert not [e for e in opened if "pq_small" in e["path"] or "pq_big" in e["path"]]
    records = D.collect([context["out"]])
    assert all(r["pq"]["method_side"] == "bz_ii" for r in records.values())
    assert D.decide_f4(D.pq_rows(records, world.table, world.rows), D.draw_idx(n=2, b=100), n_prime=2)["reading"] == D.F4_METHOD
    bad = dict(json.loads((world.repo/rel).read_text()), written_by="lead")
    (world.repo/rel).write_text(json.dumps(bad))
    commit(world.repo, "a decision not written by BZ")
    world.write_manifest()
    Path(H.MASTER_LOCK).unlink()                                     # (c) refuses the record before (d) would look at the lock
    world.reads.clear()
    refused(world, "manifest", out=Path(H.EVAL_DIR)/"out_c")


def test_a_pq_failure_midway_leaves_an_incomplete_pq_record(world, monkeypatch):
    """Review2 m10: the PQ evaluation writes its record as it goes; an exception leaves record['pq'] incomplete, and the decision lists it as such."""
    original = H.pq_pair_acceptance

    def broken(**kwargs):
        raise MemoryError("synthetic")
    monkeypatch.setattr(H, "pq_pair_acceptance", broken)
    code, context = evaluate(world)
    assert code == 1
    record = pair_record(context, world.rows[0])
    assert not record["complete"] and record["pq"]["complete"] is False and record["pq"]["state"] == "present"
    monkeypatch.setattr(H, "pq_pair_acceptance", original)
    rows = D.pq_rows(D.collect([context["out"]]), world.table, world.rows)
    excluded = D.decide_f4(rows, D.draw_idx(n=2, b=100), n_prime=2)["excluded"]
    assert {e["pair"] for e in excluded} == {r[3] for r in world.rows} and all("不完整" in e["reason"] for e in excluded)


# ---- reevaluation (D109) -----------------------------------------------------------------------------------------------------------------------------
def reeval_record(world, n, pairs, **overrides):
    master = json.loads(Path(H.MASTER_LOCK).read_text())
    doc = dict(schema=H.SCHEMA_REEVAL, written_by="lead on BZ's behalf", words="BZ: 补评", time_local="11-08 1:00 AM CST", utc=H.utc(),
               reason="the evaluation process was killed (infrastructure)", reason_class="evaluation_killed", master_lock_sha256=file_sha256(H.MASTER_LOCK),
               manifest_sha256=master["manifest_sha256"], pairs=list(pairs), previous_out=[master["out_dir"]])
    doc.update(overrides)
    path = world.repo/H.REEVAL_DIR_REL/f"holdout_B_v1_reeval_{n}.json"
    path.write_text(json.dumps(doc))
    commit(world.repo, f"reeval {n}")
    return path


def test_reevaluation_once_same_manifest_incomplete_pairs_only(world, monkeypatch):
    admission, context = world.open()
    original = H.score_cell

    def flaky(info, grid, truth, **kwargs):
        if world.rows[1][3] in str(info.run_dir) and "cadam" in str(info.run_dir):
            raise OSError("synthetic read error")
        return original(info, grid, truth, **kwargs)
    monkeypatch.setattr(H, "score_cell", flaky)
    assert H.evaluate_holdout(admission, context) == 1
    monkeypatch.setattr(H, "score_cell", original)
    done = world.rows[0][3]
    bad = reeval_record(world, 1, [done])
    with pytest.raises(H.HoldoutRefused, match="evaluated completely"):
        world.open(out=Path(H.EVAL_DIR)/"re1", reeval_path=bad)
    good = reeval_record(world, 2, [world.rows[1][3]])
    world.reads.clear()
    admission2, context2 = world.open(out=Path(H.EVAL_DIR)/"re2", reeval_path=good)
    assert H.evaluate_holdout(admission2, context2) == 0
    assert world.reads and all(world.rows[1][1] in r or world.rows[1][2] in r for r in world.reads)
    assert (H.PAIR_LOCK_DIR/f"{world.rows[1][3]}.reeval.lock").is_file()
    assert pair_record(context, world.rows[1])["complete"] is False      # the first record kept
    assert [str(p.name) for p, _ in H.run_lines()] == [Path(H.MASTER_LOCK).name, "reeval_2.lock"] and H.run_outs() == [str(context["out"]), str(context2["out"])]
    again = reeval_record(world, 3, [world.rows[1][3]])
    with pytest.raises(H.HoldoutRefused, match="evaluated completely|at most once"):
        world.open(out=Path(H.EVAL_DIR)/"re3", reeval_path=again)


def test_reevaluation_at_most_once_also_for_a_pair_the_first_run_never_reached(world, monkeypatch):
    """Review m10: the first reevaluation of a pair the first run never reached takes <pair>.lock; a second one is refused by the earlier lock line."""
    admission, context = world.open()
    path = big(world, "fsyn", world.rows[0])
    raw = path.read_bytes()
    path.write_bytes(raw[:-1]+bytes([raw[-1] ^ 1]))
    assert H.evaluate_holdout(admission, context) == 4                     # stops at the first pair; the second is never reached
    path.write_bytes(raw)
    original = H.score_cell

    def flaky(info, grid, truth, **kwargs):
        if "cadam" in str(info.run_dir):
            raise OSError("synthetic")
        return original(info, grid, truth, **kwargs)
    monkeypatch.setattr(H, "score_cell", flaky)
    first = reeval_record(world, 1, [world.rows[1][3]])
    admission1, context1 = world.open(out=Path(H.EVAL_DIR)/"re1", reeval_path=first)
    assert H.evaluate_holdout(admission1, context1) == 1
    assert (H.PAIR_LOCK_DIR/f"{world.rows[1][3]}.lock").is_file() and not (H.PAIR_LOCK_DIR/f"{world.rows[1][3]}.reeval.lock").exists()
    second = reeval_record(world, 2, [world.rows[1][3]])
    with pytest.raises(H.HoldoutRefused, match="at most once"):
        world.open(out=Path(H.EVAL_DIR)/"re2", reeval_path=second)


def test_reevaluation_reason_class(world):
    """Review m18: only the evaluation's own infrastructure failures (killed, OOM, I/O)."""
    code, context = evaluate(world)
    assert code == 0
    for value in ("method_failed", "", None):
        record = reeval_record(world, 1, [world.rows[1][3]], reason_class=value)
        with pytest.raises(H.HoldoutRefused, match="reason_class"):
            world.open(out=Path(H.EVAL_DIR)/"re", reeval_path=record)


def test_reevaluation_refuses_a_changed_manifest(world):
    code, context = evaluate(world)
    assert code == 0
    record = reeval_record(world, 1, [world.rows[1][3]], manifest_sha256="0"*64)
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(out=Path(H.EVAL_DIR)/"re", reeval_path=record)
    assert info.value.stage == "reeval"
    world.write_manifest(commit_it=False)                            # both files again (a valid manifest) ...
    (world.repo/H.MANIFEST_REL).write_text((world.repo/H.MANIFEST_REL).read_text()+"\n")       # ... but not byte for byte the one of the master lock
    (world.repo/H.RECORDS_REL).write_text((world.repo/H.RECORDS_REL).read_text()+"# changed with it\n")
    commit(world.repo, "the manifest changed after the master lock")
    record = reeval_record(world, 2, [world.rows[1][3]])
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(out=Path(H.EVAL_DIR)/"re2", reeval_path=record)
    assert info.value.stage == "reeval" and "same manifest" in str(info.value)


def test_reevaluation_needs_a_committed_record_at_the_pinned_place(world, tmp_path):
    admission, context = world.open()
    elsewhere = tmp_path/"holdout_B_v1_reeval_1.json"
    elsewhere.write_text("{}")
    with pytest.raises(H.HoldoutRefused, match="is splits"):
        world.open(out=Path(H.EVAL_DIR)/"re", reeval_path=elsewhere)
    path = world.repo/H.REEVAL_DIR_REL/"holdout_B_v1_reeval_1.json"
    path.write_text("{}")                                            # not committed
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(out=Path(H.EVAL_DIR)/"re", reeval_path=path)
    assert info.value.stage == "reeval"


# ---- D107: the classes of a written cell, holdout_common's records (alignment item 3) -----------------------------------------------------------------
def classify(world, name, row=0):
    return H.classify_cell(world.arm(name), world.rows[row], table_of(world), repo=world.repo)


def test_class_output_and_method_failure_and_unit_limit(world):
    cell = classify(world, "method")
    assert cell.written and cell.outcome == "output" and cell.review["value"] is True and cell.field_dir.endswith("output")
    cell = classify(world, "ants")
    assert cell.written and cell.field_dir == str(big(world, "ants", world.rows[0]).parent) and cell.field_sha256
    world.write_cell(world.arm("ants"), world.rows[0], status="method_failed", field=False)
    cell = classify(world, "ants")
    assert cell.written and (cell.outcome, cell.no_output_class) == ("no_output", "method")
    world.write_cell(world.arm("method"), world.rows[0], status="unit_limit", field=False)
    cell = classify(world, "method")
    assert cell.written and cell.no_output_class == "unit_limit"


def test_the_terminal_record_is_the_first_of_its_files_there(world):
    """XB's precedence: resume/overridden.json, resume/result.json, overridden.json, result.json (xb_core.effective_record): a resumed attempt's ok counts."""
    arm, row = world.arm("fsyn"), world.rows[0]
    world.write_cell(arm, row, status="infra_failure", field=False, overridden=True)        # the unit's reading overrode the first attempt
    assert classify(world, "fsyn").status == "infra_failure"
    world.write_cell(arm, row, status="ok", resume=True)                                     # the resume's own record
    cell = classify(world, "fsyn")
    assert cell.status == "ok" and cell.written and cell.terminal.endswith("resume/result.json") and ".resume" in cell.field_dir


def test_class_wrapper_exception_first_not_written_resumed_and_failed_again_written(world):
    """Alignment item 3 (revision 1a item 2, 1b item 5): a first wrapper exception is not written; after a committed resume whose run ended in a wrapper
    exception again it is (no output)."""
    arm, row = world.arm("fsyn"), world.rows[0]
    first = world.write_cell(arm, row, status="wrapper_error", field=False)
    cell = classify(world, "fsyn")
    assert not cell.written and "first wrapper exception" in cell.problem
    world.write_authorization(arm, row, "resume", previous_record=str(first), previous_record_sha256=file_sha256(first))
    assert not classify(world, "fsyn").written                         # the resume did not run yet: the terminal record is the first one (review2 m1)
    copy_ = world.record_dir("fsyn", row)/"resume"/"result.json"
    copy_.parent.mkdir()
    copy_.write_bytes(first.read_bytes())                              # the first record copied, not a resumed run: still not written (review2 m1)
    assert "did not run" in classify(world, "fsyn").problem
    world.write_cell(arm, row, status="wrapper_error", field=False, resume=True, extra=dict(attempt=2))
    cell = classify(world, "fsyn")
    assert cell.written and cell.no_output_class == "wrapper" and cell.authorization["kind"] == "resume" and first in cell.needed


def test_class_implementation_error_needs_a_no_resume_record_with_jia(world):
    """Alignment item 3 (C22.5-6, revision 1c item 3): hx_implementation_error is written only with a committed no_resume record whose reason states (甲)."""
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status="hx_implementation_error", field=False)
    step4 = world.record_dir("method", row)/"step4.json"
    cell = classify(world, "method")
    assert not cell.written and "(甲)" in cell.problem
    world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="C22.5-6 (a)，BZ (乙)：修好之后重跑")
    assert not classify(world, "method").written
    world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="BZ (甲)：补成恒等")     # no case of C22.5-6 (A5b review I2)
    assert not classify(world, "method").written
    world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="C22.5-6 (a)，BZ (甲)：补成恒等、其余对续跑")
    cell = classify(world, "method")
    assert cell.written and cell.no_output_class == "implementation" and cell.authorization["kind"] == "no_resume_jia" and step4 in cell.needed


def test_class_infra_failure_needs_a_resume_or_a_no_resume(world):
    arm, row = world.arm("configc"), world.rows[0]
    first = world.write_cell(arm, row, status="infra_failure", field=False)
    cell = classify(world, "configc")
    assert not cell.written and "resume" in cell.problem
    world.write_authorization(arm, row, "no_resume", record=str(first), record_sha256=file_sha256(first))
    cell = classify(world, "configc")
    assert cell.written and cell.no_output_class == "infrastructure" and cell.authorization["kind"] == "no_resume"


def test_class_infra_failure_after_a_resume(world):
    arm, row = world.arm("configc"), world.rows[0]
    first = world.write_cell(arm, row, status="infra_failure", field=False)
    world.write_authorization(arm, row, "resume", previous_record=str(first), previous_record_sha256=file_sha256(first))
    world.write_cell(arm, row, status="infra_failure", field=False, resume=True, extra=dict(attempt=2))
    cell = classify(world, "configc")
    assert cell.written and cell.authorization["kind"] == "resume"


@pytest.mark.parametrize("change", ["written_by", "words", "pair", "uncommitted", "note", "record_sha", "record_outside", "record_in_data"])
def test_class_records_that_do_not_hold(world, change):
    """holdout_common's checks (written by BZ or "lead on BZ's behalf", words, the cell, committed) and this entry's (note, the record at its sha256 inside
    authorization.records, never under the data root; review2 m2)."""
    arm, row = world.arm("configc"), world.rows[0]
    first = world.write_cell(arm, row, status="infra_failure", field=False)
    fields = dict(record=str(first), record_sha256=file_sha256(first))
    if change == "written_by":
        fields["written_by"] = "lead"
    elif change == "words":
        fields["words"] = " "
    elif change == "pair":
        fields["pair"] = world.rows[1][3]
    elif change == "note":
        fields["note"] = "其他"
    elif change == "record_sha":
        fields["record_sha256"] = "0"*64
    elif change == "record_outside":
        other = world.tmp/"elsewhere.json"
        other.write_text("{}")
        fields.update(record=str(other), record_sha256=file_sha256(other))
    elif change == "record_in_data":
        other = world.data/"x.json"
        other.write_text("{}")
        fields.update(record=str(other), record_sha256=file_sha256(other))
    path = world.write_authorization(arm, row, "no_resume", **fields)
    if change == "uncommitted":
        path.write_text(path.read_text().replace("synthetic", "changed"))
    if change in ("written_by", "words", "uncommitted"):
        with pytest.raises(H.HoldoutRefused):
            classify(world, "configc")
    else:
        assert not classify(world, "configc").written


def test_written_cells_with_records_go_through_the_evaluation(world):
    """A resume's previous record is read (hashed) in (b), so it is an input of the manifest; the cell is imputed as identity downstream."""
    arm, row = world.arm("fsyn"), world.rows[0]
    first = world.write_cell(arm, row, status="wrapper_error", field=False)
    world.write_authorization(arm, row, "resume", previous_record=str(first), previous_record_sha256=file_sha256(first))
    world.write_cell(arm, row, status="wrapper_error", field=False, resume=True, extra=dict(attempt=2))
    world.write_manifest()
    assert str(first) in world.needed()
    code, context = evaluate(world)
    assert code == 0
    entry = pair_record(context, row)["arms"]["fsyn"]
    assert entry["outcome"] == "no_output" and entry["no_output_class"] == "wrapper" and entry["authorization"]["kind"] == "resume"


def xb_stop(world, arm="cadam", **changes):
    """XB's own P19 record (xb_drive: <record root>/<arm>/arm_stopped.json, schema bayesreg.xb.arm_stop.v1) and its committed copy runs/holdout_B_v1/arm_stop_<arm>.json (prereg
    revision 2 item 11): XB's fields as they are, XB's file by path and sha256, written_by, unrun_pairs."""
    original = world.tmp/"records"/arm/H.XB_STOP_NAME
    xb = dict(schema=H.XB_STOP_SCHEMA, arm=arm, pair_set="holdout", utc=H.utc(), suspicious_pairs=["x", "y", "z"], at_pair="z", rule="P19 (synthetic)")
    original.write_text(json.dumps(xb), encoding="utf-8")
    rel = f"runs/holdout_B_v1/arm_stop_{arm}.json"
    document = dict(xb, schema=H.SCHEMA_STOP, written_by="lead", unrun_pairs=[world.rows[1][3]], arm_stopped=dict(path=str(original), sha256=file_sha256(original)))
    document.update(changes)
    (world.repo/rel).parent.mkdir(parents=True, exist_ok=True)
    (world.repo/rel).write_text(json.dumps(document), encoding="utf-8")
    commit(world.repo, "stop")
    return original, rel


def test_class_arm_stopped_needs_the_committed_stop_record(world):
    (world.record_dir("cadam", world.rows[1])/"result.json").unlink()
    cell = classify(world, "cadam", 1)
    assert not cell.written
    original, rel = xb_stop(world)
    cell = classify(world, "cadam", 1)
    assert cell.written and cell.no_output_class == "arm_stopped" and cell.stop_record["arm_stopped"]["path"] == str(original)
    assert str(original) in map(str, cell.needed)                     # XB's record is read: an input of the manifest (D108)
    world.write_manifest()
    code, context = evaluate(world)
    entry = pair_record(context, world.rows[1])["arms"]["cadam"]
    assert code == 0 and entry["no_output_class"] == "arm_stopped" and entry["stop_record"]["arm_stopped"]["sha256"] == file_sha256(original)


@pytest.mark.parametrize("how", ["sha256", "field", "no_pointer", "elsewhere", "missing"])
def test_the_stop_record_must_name_xb_s_record_at_its_sha256(world, how):
    """Prereg revision 2 item 11 (the lead's ruling (5); A5b item 4): the committed stop record names XB's arm_stopped.json by path and sha256 and copies its fields;
    any of these not holding leaves the cell not written."""
    (world.record_dir("cadam", world.rows[1])/"result.json").unlink()
    original, rel = xb_stop(world)
    document = json.loads((world.repo/rel).read_text())
    if how == "sha256":
        document["arm_stopped"]["sha256"] = "0"*64
    elif how == "field":
        document["at_pair"] = "another"
    elif how == "no_pointer":
        del document["arm_stopped"]
    elif how == "elsewhere":
        other = world.tmp/"elsewhere.json"
        other.write_bytes(original.read_bytes())
        document["arm_stopped"]["path"] = str(other)
    else:
        original.unlink()
    (world.repo/rel).write_text(json.dumps(document))
    commit(world.repo, f"stop record: {how}")
    cell = classify(world, "cadam", 1)
    assert not cell.written and ("arm_stopped" in cell.problem or "sha256" in cell.problem or "fields" in cell.problem), cell.problem


def test_fixed_statuses_cannot_be_remapped_and_excluded_arms(world):
    table = json.loads((world.repo/H.ARM_TABLE_REL).read_text())
    bad = copy.deepcopy(table)
    bad["arms"][1]["terminal"]["classes"]["method_failure"].append("needs_lead_rejudgment")
    bad["arms"][1]["terminal"]["classes"]["not_written"].remove("needs_lead_rejudgment")
    with pytest.raises(H.HoldoutRefused, match="fixed class"):
        H.validate_arm_table(bad)
    for name in ("transmorph", "vpreg_x", "synthmorph"):
        bad = copy.deepcopy(table)
        bad["arms"].append(dict(copy.deepcopy(table["arms"][1]), name=name, family="descriptive"))
        with pytest.raises(H.HoldoutRefused, match="not in the arm table"):
            H.validate_arm_table(bad)
    bad = copy.deepcopy(table)
    bad["main_family"] = bad["main_family"][:4]
    with pytest.raises(H.HoldoutRefused, match="main_family"):
        H.validate_arm_table(bad)
    good = copy.deepcopy(table)
    good["arms"].append(dict(copy.deepcopy(table["arms"][1]), name="ants_90s", family="tier_90s"))
    assert H.validate_arm_table(good)["arms"][-1]["family"] == "tier_90s"


# ---- the manifest's form --------------------------------------------------------------------------------------------------------------------------------
def test_parse_manifest_forms():
    head = f"# {H.SCHEMA_MANIFEST}\n# prereg_sha256 {'1'*64}\n# arm_table_sha256 {'2'*64}\n# unseal_sha256 {'3'*64}\n"
    header, inputs = H.parse_manifest((head+f"{'a'*64}  /x/disp.npz\n\n{'b'*64}  /x/line.json\n").encode())
    assert inputs == {"/x/disp.npz": "a"*64, "/x/line.json": "b"*64} and header["unseal_sha256"] == "3"*64
    header, _ = H.parse_manifest((head+"# pq_method_side_decision runs/holdout_B_v1/pq_method_side.json\n").encode())
    assert header["pq_method_side_decision"] == "runs/holdout_B_v1/pq_method_side.json"
    for bad in (head+f"{'a'*64}  input  /x/disp.npz\n",                        # no role column: the record-only lines are in the other file
                head+f"{'a'*64}  /x/a\n{'b'*64}  /x/a\n",                      # one line per path
                head+"# something else\n", head+f"#begin input_lock {'4'*40} p {'5'*64}\n", head+f"#annex rejected {'6'*64}  /x/old.json\n",
                head.replace("# unseal_sha256", "# x"), head+f"{'a'*64}  relative/path\n",
                head+f"# pq_manifest_sha256 {'4'*64}\n",                       # PQ's release is read from the code tree now
                head+"# pq_method_side_decision /abs/x.json\n", head.replace(H.SCHEMA_MANIFEST, "bayesreg.holdout.eval_manifest.v2")):
        with pytest.raises(H.HoldoutRefused):
            H.parse_manifest(bad.encode())


# ---- the other entries keep refusing sealed subjects; the dry run --------------------------------------------------------------------------------------
def test_other_entries_refuse_sealed_subjects_even_with_a_bound_allowlist(world, tmp_path):
    """C23.12-7: truth.admit_pair, truth.load_truth, eval_seg.admit_subjects and OasisFiles.label_path refuse a sealed subject of the (synthetic) hold-out
    even with the allowlist of bind_unseal and the OasisFiles of bind_files (SG's real functions); only this entry reads its truth."""
    from bayesreg.eval import truth as T
    from bayesreg.eval.metrics import OASIS
    from bayesreg.io.pair import _read_manifest
    from bayesreg.io.sealed import guard_sealed_lists
    from bayesreg.seg.unet.data import OasisFiles
    fixed, moving = SUBJECTS[0], SUBJECTS[1]
    sealed = guard_sealed_lists()
    data = tmp_path/"d"
    name = f"{fixed}__{moving}"
    syn.write_pair(data, name, fixed, moving)
    wrong = load_dev_allowlist(syn.write_allowlist(tmp_path/"a.json", subjects=(fixed, moving)))      # an allowlist that wrongly holds them
    unseal = U.load_unseal(world.repo/H.UNSEAL_REL)
    bound = U.bind_unseal(load_dev_allowlist(world.repo/H.ALLOWLIST_REL), unseal)
    for allow in (wrong, bound):
        with pytest.raises(SubjectSealed):
            T.admit_pair(data, syn.STAGE, name, allow, sealed)
    pair_dir = data/syn.STAGE/name
    grid, _, selected = _read_manifest(pair_dir/"manifest.json")
    handmade = dict(unsealed=False, manifest_sha256=file_sha256(pair_dir/"manifest.json"), fixed_subject=fixed, moving_subject=moving,
                    pair_sha256_manifest=selected["pair_sha256"], allowlist_path=str(tmp_path/"a.json"), allowlist_sha256=wrong.sha256)
    with pytest.raises(SubjectSealed):
        T.load_truth(pair_dir, OASIS, grid, handmade, repo=world.repo)        # the world's lists (truth reads them from the tree it is given)
    sys.path.insert(0, str(H.REPO/"experiments"/"unv2"))
    import eval_seg
    for allow in (wrong, bound):
        with pytest.raises(SubjectSealed):
            eval_seg.admit_subjects(allow, "synthetic_dev" if allow is wrong else "holdout_B_v1_synth", sealed, set())
    files = U.bind_files(OasisFiles(root=world.data), bound)
    assert files.image_path(fixed) == world.data/fixed/"aligned_norm.nii.gz"            # the bound OasisFiles answers for the image ...
    for kind in ("seg35", "seg4"):
        with pytest.raises(SubjectSealed, match="never read through OasisFiles"):
            files.label_path(fixed, kind)                                                 # ... never for a label
    with pytest.raises(SubjectSealed):
        OasisFiles(root=world.data).label_path(fixed)
    assert world.reads == []


def test_cli_refuses_a_real_hold_out_subject(tmp_path):
    """bayesreg.eval.cli on the real pinned lists (not the synthetic world): a real holdout_B id refuses with exit code 3, nothing read."""
    from bayesreg.io.sealed import load_sealed_lists
    S._reset_cache()
    real = sorted(load_sealed_lists().subjects["holdout_B_v1"])[:2]
    data = tmp_path/"d"
    name = f"{real[0]}__{real[1]}"
    syn.write_pair(data, name, real[0], real[1])
    syn.write_allowlist(tmp_path/"a.json", subjects=tuple(real))
    run = tmp_path/"run"
    run.mkdir()
    np.savez(run/"disp.npz", d=np.zeros((3,)+syn.SHAPE))
    assert cli.main([str(run), "--pairs-stage", syn.STAGE, "--pair", name, "--method", "x", "--out", "m.json", "--long", str(tmp_path/"l.csv"),
                     "--out-dir", str(tmp_path/"o"), "--data-root", str(data), "--allowlist", str(tmp_path/"a.json")]) == 3
    S._reset_cache()


def test_main_refusals_that_are_not_holdout_refused_exit_2(world, monkeypatch, capsys):
    """Review m16: a SealedListError (or an OSError / ValueError) before any truth exits 2 as a refusal, not with a traceback."""
    def broken(*args, **kwargs):
        raise S.SealedListError("sealed list holdout_B_v1: sha256 differs (synthetic)")
    monkeypatch.setattr(H, "open_holdout", broken)
    assert H.main(["evaluate", "--out-dir", str(world.results/"o")]) == 2
    assert "SealedListError" in capsys.readouterr().err


# ---- the spy (C23.6-5) --------------------------------------------------------------------------------------------------------------------------------
def test_spy_requires_an_empty_cache_of_the_sealed_lists(world):
    from bayesreg.io.sealed import guard_sealed_lists
    guard_sealed_lists()
    assert not S.sealed_cache_is_empty()
    exc = refused(world, "spy", fresh=False)
    assert "cached" in str(exc)


def test_spy_records_each_read_of_the_sealed_lists_and_every_input(world):
    code, context = evaluate(world)
    assert code == 0 and H._SPY is None
    opened = json.loads((context["out"]/"holdout_eval.json").read_text())["opened_files"]
    lists = {str((world.repo/spec["path"]).resolve()) for spec in world.specs}
    seen = [e for e in opened["events"] if e["kind"] == "sealed_list"]
    assert {e["path"] for e in seen} == lists and opened["sealed_list_reads"] == len(seen) >= len(lists)
    assert opened["refused"] == [] and {e["kind"] for e in opened["events"]} == {"sealed_list", "input", "truth"}
    inputs = {e["path"] for e in opened["events"] if e["kind"] == "input"}
    assert inputs <= {str(Path(p).resolve()) for p in context["listed"]}
    assert any("pq_big" in p for p in inputs) and any("pq_small" in p for p in inputs) and any("/big/" in p for p in inputs)
    truth = {e["path"] for e in opened["events"] if e["kind"] == "truth"}
    assert truth == {str((world.data/s/H.LABEL_FILES[k]).resolve()) for s in SUBJECTS for k in ("seg35", "seg4")}


def test_spy_stops_an_open_outside_the_manifest(world, monkeypatch):
    stray = world.record_dir("fga", world.rows[0])/"notes.txt"
    stray.write_text("not listed")
    admission, context = world.open()
    original = H.score_cell

    def nosy(info, grid, truth, **kwargs):
        stray.read_text()                                            # a read the manifest does not hold
        return original(info, grid, truth, **kwargs)
    monkeypatch.setattr(H, "score_cell", nosy)
    assert H.evaluate_holdout(admission, context) == 4
    stopped = json.loads((context["out"]/"stopped.json").read_text())
    assert "SpyRefused" in stopped["reason"] and stopped["pair"] == world.rows[0][3]
    assert json.loads((context["out"]/"holdout_eval.json").read_text())["opened_files"]["refused"] == [str(stray.resolve())]


def test_spy_stops_an_unlisted_pq_file(world, monkeypatch):
    stray = world.pq_dirs(world.rows[0])[0]/"notes.txt"
    stray.write_text("not listed")
    admission, context = world.open()
    original = H.score_cell

    def nosy(info, grid, truth, **kwargs):
        stray.read_text()                                            # a PQ directory is watched too
        return original(info, grid, truth, **kwargs)
    monkeypatch.setattr(H, "score_cell", nosy)
    assert H.evaluate_holdout(admission, context) == 4
    assert json.loads((context["out"]/"holdout_eval.json").read_text())["opened_files"]["refused"] == [str(stray.resolve())]


def test_spy_refuses_before_c_a_file_outside_the_manifest(world, monkeypatch):
    original = H.check_cells

    def nosy(table, pairs, *, repo=None, cache=None):
        big(world, "ants", world.rows[0]).read_bytes()                                                # listed: fine
        (world.data/SUBJECTS[0]/H.LABEL_FILES["seg35"]).read_bytes()                                  # a truth file before (c): never allowed
        return original(table, pairs, repo=repo, cache=cache)
    monkeypatch.setattr(H, "check_cells", nosy)
    refused(world, "spy", check_spy=False)                           # this test reads a truth file on purpose; the spy must catch it
    assert any(e["path"].startswith(str(world.data.resolve())) for e in H._LAST_SPY.events)


def test_spy_watches_a_linked_subject_directory_by_its_unresolved_path(tmp_path):
    """Review m17: a directory under the watched root that links elsewhere is still watched."""
    elsewhere = tmp_path/"elsewhere"
    elsewhere.mkdir()
    (elsewhere/"aligned_seg35.nii.gz").write_bytes(b"x")
    root = tmp_path/"data"
    root.mkdir()
    (root/"OASIS_OAS1_0999_MR1").symlink_to(elsewhere, target_is_directory=True)
    spy = H.OpenedFiles([root], [])
    spy.saw(str(root/"OASIS_OAS1_0999_MR1"/"aligned_seg35.nii.gz"), "rb")
    assert spy.events and spy.events[0]["path"] == str((elsewhere/"aligned_seg35.nii.gz").resolve())
    spy.events.clear()
    spy.allow([])
    with pytest.raises(H.SpyRefused):
        spy.saw(str(root/"OASIS_OAS1_0999_MR1"/"aligned_seg35.nii.gz"), "rb")


# ---- the Jacobian certificate of every arm in BL1's configuration ------------------------------------------------------------------------------
def test_bl1_record_is_the_bl1_certificate():
    from dataclasses import fields
    from jd_curl.three_d.grid import make_grid_3d
    from .test_cert_eval_bl1 import archived, fields_of
    original, config = archived()
    assert dict(H.BL1_CERT_CONFIG) == dict(delta=config.delta, max_depth=config.max_depth, max_boxes=config.max_boxes, tile_size=config.tile_size,
                                           policy=config.policy, prepass=config.prepass, fallback_workers=config.fallback_workers)
    grid = make_grid_3d(nx=7, ny=8, nz=9, hx=1., hy=1., hz=1., unit="mm")
    for name, u in fields_of(grid).items():
        record = H.certify_field_bl1(u, grid)
        theirs = original(u, grid)
        for f in fields(theirs):
            if f.name in cert_eval.RESULT_FIELDS:
                mine, other = record[f.name], cert_eval._value(getattr(theirs, f.name))
                assert mine == other or (isinstance(mine, float) and np.isnan(mine) and np.isnan(other)), (name, f.name)
        assert record["config"]["config_class"] == "ParallelEnclosureCertificateConfig" and record["configuration"] == "BL1"
    with pytest.raises(ValueError):
        H.score_d(np.zeros((3, 9, 8, 7)), grid, None, cert_mode="full", jacobian_config="BL1")


def test_dry_run_on_this_tree_stops_at_0_and_a_reading_nothing(tmp_path, monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(H, "_load_label_volume", lambda path: reads.append(path))
    monkeypatch.setattr(H, "PREREG_SHA256", None)                   # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, as before
    monkeypatch.setattr(H, "TRUTH_READER_DECISION", None)           # P1 filled the two (runs/20261115-p1); the state before, set back here, as PREREG_SHA256
    monkeypatch.setattr(H._experiment_module("pq", "pqh_pins"), "PQ_REPLAY_DECISION", None)
    assert H.TRUTH_READER_DECISION is None and H.PREREG_SHA256 is None
    assert H.main(["dry-run", "--out-dir", str(tmp_path/"dry"), "--pair", "OASIS_OAS1_0423_MR1__OASIS_OAS1_0099_MR1"]) == 0
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    assert report["first_refusal"] == "rule one (pinned_code.json; C24.3-4)" and "pinned_code.json" in report["conditions"]["rule one (pinned_code.json; C24.3-4)"]
    assert report["conditions"]["(0) truth-reader gate"].startswith("refused") and report["conditions"]["(0) PQ replay gate"].startswith("refused")
    assert report["conditions"]["(a) unseal object and pairs"].startswith("refused")
    assert report["conditions"]["(a) pair on the list"] == "passed" and reads == []


def test_dry_run_on_this_tree_at_p0_passes_b_and_reads_nothing(tmp_path, monkeypatch):
    """P0 (runs/20261114-p0/d1_expected.md section 7): with the pins as committed at P0 (P1 filled the two (0) constants: set back to None here, the P0 state), the dry run on this
    tree refuses at rule one, both (0) gates, the unseal record and the manifest, and passes (a) the pair and (b) the preregistration; no truth is read."""
    reads = []
    monkeypatch.setattr(H, "_load_label_volume", lambda path: reads.append(path))
    monkeypatch.setattr(H, "TRUTH_READER_DECISION", None)
    monkeypatch.setattr(H._experiment_module("pq", "pqh_pins"), "PQ_REPLAY_DECISION", None)
    assert H.main(["dry-run", "--out-dir", str(tmp_path/"dry"), "--pair", "OASIS_OAS1_0423_MR1__OASIS_OAS1_0099_MR1"]) == 0
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    conditions = report["conditions"]
    assert report["first_refusal"] == "rule one (pinned_code.json; C24.3-4)" and "pinned_code.json" in conditions["rule one (pinned_code.json; C24.3-4)"]
    assert conditions["(0) truth-reader gate"].startswith("refused") and conditions["(0) PQ replay gate"].startswith("refused")
    assert conditions["(a) pair on the list"] == "passed" and conditions["(a) unseal object and pairs"].startswith("refused")
    assert conditions["(b) preregistration"] == "passed" and conditions["(c) manifest committed"].startswith("refused")
    assert report["truth_read"] is False and reads == []


def test_dry_run_on_this_tree_at_p1_passes_both_gates_and_reads_nothing(tmp_path, monkeypatch):
    """P1 (runs/20261115-p1): with the two (0) constants as filled, the dry run on this tree passes both (0) gates against the real gate records (read only: their sha256 before and
    after), rule two asked about P0'' by both (replaced here by a recorder: the range P0''..HEAD is tested in tests/test_bayesreg/io_seg/test_code_identity.py, the working tree by the
    gates at run time); the rest as at P0 (rule one, the unseal record and the manifest refused; the pair and the preregistration passed); no truth is read."""
    before = P1.real_evidence(P1.TRUTH["result_json"], P1.TRUTH_LOCK[0], P1.PQ["result_json"], P1.PQ_LOCK[0])
    reads, gates = [], []
    monkeypatch.setattr(H, "_load_label_volume", lambda path: reads.append(path))
    monkeypatch.setattr(H, "code_identity_problem", lambda commit, repo=None: gates.append(("entry", commit)))
    from bayesreg.io import code_identity
    monkeypatch.setattr(code_identity, "gate_identity_problems", lambda commit, *, repo=None, run=None, head="HEAD": (gates.append(("pqh", commit)), (dict(stub="test"), []))[1])
    assert H.main(["dry-run", "--out-dir", str(tmp_path/"dry"), "--pair", "OASIS_OAS1_0423_MR1__OASIS_OAS1_0099_MR1"]) == 0
    report = json.loads((tmp_path/"dry"/"dry_run.json").read_text())
    conditions = report["conditions"]
    assert report["first_refusal"] == "rule one (pinned_code.json; C24.3-4)" and "pinned_code.json" in conditions["rule one (pinned_code.json; C24.3-4)"]
    assert conditions["(0) truth-reader gate"] == "passed" and conditions["(0) PQ replay gate"] == "passed"
    assert sorted(gates) == sorted([("entry", P1.P0PP), ("pqh", P1.P0PP), ("entry", P1.P0PP)])            # the truth gate's check; PQH's function and the entry's own check of the replay gate
    assert conditions["(a) pair on the list"] == "passed" and conditions["(a) unseal object and pairs"].startswith("refused")
    assert conditions["(b) preregistration"] == "passed" and conditions["(c) manifest committed"].startswith("refused")
    assert report["truth_read"] is False and reads == []
    assert tuple(P1.sha256_of(p) for p in (P1.TRUTH["result_json"], P1.TRUTH_LOCK[0], P1.PQ["result_json"], P1.PQ_LOCK[0])) == before


def test_the_truth_reader_decision_is_the_gate_that_ran_on_p0pp():
    """P1 (prereg revision 2 item 9 step 5; runs/20261107-ev/truth_gate_plan.md section 4; runs/20261115-d58/result_C4.md): TRUTH_READER_DECISION is a plain dict literal with exactly
    DECISION_KEYS and the values of the gate of 2026-11-15 on P0'' (written again in tests/test_bayesreg/_p1_decisions.py): the pinned plan, the pass reading and counts, the gate's own
    result and lock paths; holdout.py is the file of P0'' but for this constant (rule two's AST check on the file)."""
    decision = H.TRUTH_READER_DECISION
    P1.assert_values(decision, H.DECISION_KEYS, P1.TRUTH)
    assert decision["plan_sha256"] == H.TRUTH_GATE_PLAN_SHA256 and decision["reading"] == H.GATE_READING_PASS and decision["counts"] == {k: dict(v) for k, v in H.GATE_COUNTS.items()}
    assert sorted(decision["environment"]) == sorted(H.DECISION_ENVIRONMENT_KEYS)
    assert Path(decision["result_json"]) == Path(H.TRUTH_GATE_OUT)/"truth_gate.json" and Path(H.TRUTH_GATE_LOCK) == P1.TRUTH_LOCK[0]
    rel = "bayesreg/eval/holdout.py"
    now, gate = (H.REPO/rel).read_text(encoding="utf-8"), P1.blob_at(P1.P0PP, rel)
    assert P1.at_gate(now, "TRUTH_READER_DECISION") == gate
    from bayesreg.io import code_identity
    assert code_identity.decision_assignment_problem(gate, now, rel) is None


def test_the_truth_reader_decision_holds_against_the_real_gate(monkeypatch):
    """P1: check_decision() takes the constant against the real truth_gate.json and lock (read only: their sha256 before and after), whose schema, reading, counts, plan, environment and code
    (head P0'', clean; the lock's first code_head P0'') are the constant's; rule two is asked about P0'' (replaced here by a recorder; the range: tests/test_bayesreg/io_seg/test_code_identity.py)."""
    before = P1.real_evidence(P1.TRUTH["result_json"], P1.TRUTH_LOCK[0])
    assert before == (P1.TRUTH["result_sha256"], P1.TRUTH_LOCK[1])
    result = json.loads(Path(P1.TRUTH["result_json"]).read_text(encoding="utf-8"))
    assert result["schema"] == H.SCHEMA_GATE and result["reading"] == P1.TRUTH["reading"] and result["counts"] == P1.TRUTH["counts"] and result["environment"] == P1.TRUTH["environment"]
    assert result["plan"]["sha256"] == P1.TRUTH["plan_sha256"] and result["code"]["head"] == P1.P0PP and result["code"]["clean"] is True
    assert P1.lock_line(P1.TRUTH_LOCK[0])["code_head"] == P1.P0PP
    gates = []
    monkeypatch.setattr(H, "code_identity_problem", lambda commit, repo=None: gates.append(commit))
    checked = H.check_decision()
    assert gates == [P1.P0PP] and checked["reading"] == H.GATE_READING_PASS and checked["code_commit"] == P1.P0PP
    assert (P1.sha256_of(P1.TRUTH["result_json"]), P1.sha256_of(P1.TRUTH_LOCK[0])) == before


def test_the_preregistration_and_arm_table_pins_are_the_committed_files():
    """P0: PREREG_SHA256 and ARM_TABLE_SHA256 are the sha256 of the committed preregistration (revisions 1-2, adopted by BZ 11-10) and arm table; condition (b)'s two
    readers accept them (the preregistration names the table's sha256; the table validates).  Only these two code-tree files are read."""
    P0.assert_pinned(H.PREREG_SHA256, H.PREREG_REL)
    P0.assert_pinned(H.ARM_TABLE_SHA256, H.ARM_TABLE_REL)
    raw, record = H.check_prereg()
    table, table_record = H.load_arm_table(raw)
    assert record["sha256"] == H.PREREG_SHA256 and table_record["sha256"] == H.ARM_TABLE_SHA256 and H.method_arm(table)["family"] == "method"


def test_constants_match_the_codebase():
    from bayesreg.seg.unet.data import DEFAULT_DATA_ROOT, LABEL_FILES
    assert dict(H.LABEL_FILES) == LABEL_FILES and H.DATA_ROOT == DEFAULT_DATA_ROOT
    assert H.sha_bytes((H.REPO/H.PAIRS_REL).read_bytes()) == H.PAIRS_SHA256
    assert H.sha_bytes((H.REPO/H.SEG4_MAP_REL).read_bytes()) == H.SEG4_MAP_SHA256
    rows, _ = H.load_pairs()
    from bayesreg.eval.truth import load_sealed_lists
    assert {s for r in rows for s in (r[1], r[2])} == set(load_sealed_lists().subjects["holdout_B_v1"])
    _, pq_eval, pq_stats = H._pq_modules()
    assert D.PQ_BASELINES == pq_stats.JUDGED_BASELINES and D.PQ_R_MIN == pq_stats.R_MIN and H.DECISION_TILE == 65536
    hc = H._holdout_common()
    assert H.HOLDOUT_RESULTS == hc.HOLDOUT_ROOT and set(hc.RESERVED) <= set(H.RESERVED_NAMES) and hc.NO_RESUME_NOTE == "未批补跑"
    assert H.CODE_DIRS == ("bayesreg", "experiments", "jd_curl", "configs", "scripts")


def test_h1_id_constant_is_synthetic():
    assert int(H1_ID.split("_")[2]) >= 900
