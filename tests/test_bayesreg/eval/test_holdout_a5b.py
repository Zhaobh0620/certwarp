"""Work package A5b on the evaluation entry (runs/20261110-a5/任务书.md, supplements 1-4, 7; contract revision 9 draft C24.2, C24.3, C24.8; prereg revision 2 items 4, 11, 12,
14, 17 (f), 18): rule one and rule two (bayesreg.io.code_identity), the truth gate's code commit, the --out-dir, BZ's (ii) path, the method cell through the finalized and
move records, the step-5 (甲) and the withdrawal (arm review), P20 (the method's and the other arms'), the rejudgment records, XB's stop record, the repair ablation (decided
before any truth, never certified, in no family) and holdout_decide's ablation section.  The synthetic world of _holdout_world.py only; no real subject file is opened."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

import bayesreg.io.sealed as S
from bayesreg.eval import cert_eval
from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as D
from bayesreg.io import code_identity as CI
from bayesreg.io import rejudgment as RJ

from ._holdout_world import MAIN, World, commit, git, head


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "_LAST_SPY", None)
    w = World(tmp_path)
    w.patch(monkeypatch)
    yield w
    H.stop_spy()
    S._reset_cache()


def refused(world, stage, **kwargs):
    with pytest.raises(H.HoldoutRefused) as info:
        world.open(**kwargs)
    assert info.value.stage == stage, info.value
    assert world.reads == [] and not Path(H.MASTER_LOCK).exists() and H._SPY is None
    return info.value


def classify(world, name, row=0):
    table = H.validate_arm_table(json.loads((world.repo/H.ARM_TABLE_REL).read_text()))
    return H.classify_cell(world.arm(name), world.rows[row], table, repo=world.repo)


def evaluate(world, **kwargs):
    admission, context = world.open(**kwargs)
    return H.evaluate_holdout(admission, context), context


def pair_record(context, row):
    return json.loads((context["out"]/"pairs"/f"{row[3]}.json").read_text())


# ---- rule one (C24.3-4, -7): before anything; P1 from the committed pinned record --------------------------------------------------------------------
def pin(world, *, commit_=None):
    """runs/holdout_B_v1/pinned_code.json naming P1 (the world's HEAD now, or `commit_`) and its decision_log line, added in one commit (C24.3-2, -9)."""
    p1 = commit_ or head(world.repo)
    line = f"| 2026-11-10 | the pinned commit {p1} (synthetic) |"
    rel = CI.PINNED_RECORD_REL
    (world.repo/rel).parent.mkdir(parents=True, exist_ok=True)
    (world.repo/rel).write_text(json.dumps(dict(schema=CI.PINNED_RECORD_SCHEMA, commit=p1, decision_log_line=line, written_by="lead", time_local="11-10 1:00 PM CST",
                                                utc="2026-11-10T19:00:00Z", reason="synthetic")), encoding="utf-8")
    with (world.repo/CI.LATER_DECISION_LOG).open("a", encoding="utf-8") as stream:
        stream.write(line+"\n")
    commit(world.repo, "pinned code")
    return p1


def test_rule_one_comes_first_and_refuses_nothing_read(world, monkeypatch):
    calls = []
    monkeypatch.setattr(H, "check_decision", lambda *a, **k: calls.append("0"))
    monkeypatch.setattr(H, "unit_identity_problems", lambda commit=None, *, repo=None, run=None: (dict(pinned=None), ["no pinned record (synthetic)"]))
    exc = refused(world, "code_identity")
    assert "no pinned record" in str(exc) and calls == []
    monkeypatch.setattr(H, "unit_identity_problems", CI.unit_identity_problems)            # the real rule: the world has no pinned record
    exc = refused(world, "code_identity")
    assert "pinned_code.json" in str(exc) and calls == []


def test_rule_one_holds_on_a_pinned_tree_and_the_lock_line_names_p1(world, monkeypatch):
    monkeypatch.setattr(H, "unit_identity_problems", CI.unit_identity_problems)
    p1 = pin(world)
    (world.repo/"splits"/"bayesreg_v1"/"holdout_B_v1_xb_rerun_1.json").write_text("{}")       # an authorization after P1: allowed by rule one (added once)
    commit(world.repo, "an authorization after P1")
    admission, context = world.open()
    line = json.loads(Path(H.MASTER_LOCK).read_text())
    assert line["pinned_commit"] == p1 and context["record"]["rule_one"]["pinned"] == p1 and world.reads == []
    H.stop_spy()


@pytest.mark.parametrize("change", ["code", "record_changed", "tracked_dirty"])
def test_rule_one_refuses_code_after_p1_a_changed_record_or_a_dirty_tree(world, monkeypatch, change):
    monkeypatch.setattr(H, "unit_identity_problems", CI.unit_identity_problems)
    pin(world)
    if change == "code":
        (world.repo/"configs"/"bayesreg"/"other.json").write_text("{}")
        commit(world.repo, "a code change after P1")
    elif change == "record_changed":
        (world.repo/H.MANIFEST_REL).write_text((world.repo/H.MANIFEST_REL).read_text()+"\n")
        commit(world.repo, "a record changed after P1")
    else:
        (world.repo/H.ALLOWLIST_REL).write_text((world.repo/H.ALLOWLIST_REL).read_text()+" ")
    refused(world, "code_identity")


# ---- rule two at (0) (C24.3-6; supplement 7): the four decision constants as dict literals pass, anything more refuses ---------------------------------------
CONSTANT_HOMES = dict(CI.DECISION_HOMES)


def gate_tree(world, *, sneak=False):
    """P0: the four constants' files with None; a documents commit; P1: the four filled as plain dict literals (C24.3-6 (a); decision_log 2026-11-10 ~3:10 AM CST (1)),
    one line of them with a statement after a ';' when `sneak`.  Returns P0."""
    for name, rel in CONSTANT_HOMES.items():
        path = world.repo/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{rel} (synthetic)"""\nX = 1\n{name} = None\n\n\ndef f():\n    return {name}\n', encoding="utf-8")
    commit(world.repo, "P0: the gates' commit")
    p0 = head(world.repo)
    (world.repo/"docs").mkdir(exist_ok=True)
    (world.repo/"docs"/"result.md").write_text("the gates' results (synthetic)\n")
    (world.repo/"runs"/"20261110-x").mkdir(parents=True)
    (world.repo/"runs"/"20261110-x"/"review.md").write_text("a review (synthetic)\n")
    commit(world.repo, "documents after P0 (not checked by rule two)")
    for k, (name, rel) in enumerate(CONSTANT_HOMES.items()):
        path = world.repo/rel
        literal = "{'reading': 'pass', 'code_commit': '%s', 'counts': {'a': 1}}" % p0
        tail = "; X = 2" if sneak and k == 0 else ""
        path.write_text(path.read_text(encoding="utf-8").replace(f"{name} = None", f"{name} = {literal}{tail}"), encoding="utf-8")
    commit(world.repo, "P1: the four decision constants")
    return p0


def gate_decisions(world, p0):
    """The truth gate's decision and the PQ replay gate's (the world's stand-in check) with code_commit P0: the truth gate's result and lock as the gate wrote them on P0."""
    decision = world.decision()
    result_path = Path(decision["result_json"])
    result = json.loads(result_path.read_text())
    result["code"] = dict(head=p0, clean=True)
    result_path.write_text(json.dumps(result))
    (world.results/"ev"/"truth_gate.lock").write_text(json.dumps(dict(code_head=p0))+"\n")
    decision.update(code_commit=p0, result_sha256=file_sha256(result_path))
    return decision, dict(world.pq_replay(), code_commit=p0)


def test_rule_two_lets_the_four_dict_literal_constants_through_at_0(world):
    p0 = gate_tree(world)
    decision, replay = gate_decisions(world, p0)
    assert H.code_identity_problem(p0, world.repo) is None
    H.check_decision(decision, world.repo)
    H.check_pq_replay(replay, world.repo)
    world.write_manifest()
    admission, context = world.open(decision=decision, pq_replay=replay)
    assert context["record"]["gate"]["code_commit"] == p0 and context["record"]["pq_replay"]["code_commit"] == p0
    H.stop_spy()


def test_rule_two_refuses_a_constant_line_that_carries_another_statement(world):
    p0 = gate_tree(world, sneak=True)
    decision, replay = gate_decisions(world, p0)
    exc = refused(world, "gate", decision=decision, pq_replay=replay)
    assert "lines to itself" in str(exc) or "outside the decision" in str(exc)
    with pytest.raises(H.HoldoutRefused) as info:
        H.check_pq_replay(replay, world.repo)
    assert info.value.stage == "pq_replay"


def test_rule_two_refuses_any_other_code_change_after_the_gate(world):
    p0 = gate_tree(world)
    decision, replay = gate_decisions(world, p0)
    (world.repo/"experiments"/"holdout"/"helper.py").write_text("X = 1\n")
    commit(world.repo, "another code file after the gate")
    exc = refused(world, "gate", decision=decision, pq_replay=replay)
    assert "experiments/holdout/helper.py" in str(exc)


# ---- the truth gate's code commit (B4 review, 3rd plan, I1) ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["result_head", "not_clean", "lock_head", "no_lock"])
def test_the_truth_gate_decision_must_name_the_commit_the_gate_ran_on(world, how):
    decision = world.decision()
    result_path, lock = Path(decision["result_json"]), world.results/"ev"/"truth_gate.lock"
    result = json.loads(result_path.read_text())
    if how == "result_head":
        result["code"]["head"] = "a"*40
    elif how == "not_clean":
        result["code"]["clean"] = False
    elif how == "lock_head":
        lock.write_text(json.dumps(dict(code_head="b"*40))+"\n")
    else:
        lock.unlink()
    result_path.write_text(json.dumps(result))
    decision["result_sha256"] = file_sha256(result_path)
    exc = refused(world, "gate", decision=decision)
    assert decision["code_commit"] in str(exc)


# ---- the --out-dir (C24.8 item 15) and BZ's (ii) path (A5b items 9, 10) --------------------------------------------------------------------------------------
def test_the_out_dir_is_a_new_directory_under_the_eval_root(world):
    for out in (world.results/"elsewhere", Path(H.EVAL_DIR), Path(H.PAIR_LOCK_DIR)/"x", Path(H.EVAL_DIR)/"reeval_3.lock", Path(H.EVAL_DIR)/"holdout_eval.lock"):
        exc = refused(world, "output", out=out)
        assert str(H.EVAL_DIR) in str(exc) or "lock" in str(exc)
    admission, context = world.open(out=Path(H.EVAL_DIR)/"run_1")
    assert context["out"] == Path(H.EVAL_DIR)/"run_1"
    H.stop_spy()


def test_bz_s_ii_record_only_at_its_one_path(world):
    other = "runs/holdout_B_v1/pq_method_side_other.json"
    (world.repo/other).write_text(json.dumps(dict(schema=H.SCHEMA_PQ_METHOD_SIDE, written_by="BZ", words="（合成测试）", time_local="x", utc=H.utc(), reason="r", choice="ii")))
    commit(world.repo, "BZ (ii) at another path")
    world.bz_ii = other
    world.write_manifest()
    exc = refused(world, "manifest")
    assert H.PQ_METHOD_SIDE_REL in str(exc)


# ---- the method cell through the finalized and move records (A5b item 8) -------------------------------------------------------------------------------------
def move_path(world, row):
    return world.record_dir("method", row)/H.MOVE_NAME


def rewrite(path, change):
    document = json.loads(Path(path).read_text())
    change(document)
    Path(path).write_text(json.dumps(document))


@pytest.mark.parametrize("how", ["no_move", "dst_dir", "src_dir", "dst_of_a_file", "sha256_after", "no_finalized", "finalized_sha", "half_moved"])
def test_the_method_cell_must_be_the_finalized_cell_moved(world, how):
    row = world.rows[0]
    if how == "no_move":
        move_path(world, row).unlink()
    elif how == "dst_dir":
        rewrite(move_path(world, row), lambda d: d.update(dst_dir=str(world.tmp/"elsewhere")))
    elif how == "src_dir":
        rewrite(move_path(world, row), lambda d: d.update(src_dir=str(world.tmp/"elsewhere")))
    elif how == "dst_of_a_file":
        rewrite(move_path(world, row), lambda d: [f.update(dst=str(world.tmp/"x"/f["file"])) for f in d["files"] if f["file"] == "output/disp.npz"])
    elif how == "sha256_after":
        rewrite(move_path(world, row), lambda d: [f.update(sha256_after="0"*64) for f in d["files"] if f["file"] == "output/final.npz"])
    elif how == "no_finalized":
        (world.review_root/H.FINALIZED_NAME).unlink()
    elif how == "finalized_sha":
        rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]]["part"]["output"].update({"output/cert.json": "1"*64}))
    else:
        rewrite(move_path(world, row), lambda d: d.update(passed=False))
    cell = classify(world, "method")
    assert not cell.written and "finalized and move records" in cell.problem, cell.problem
    refused(world, "arms")


def test_a_half_moved_cell_counts_with_a_passed_move_resume(world):
    """The interface alignment revision 2: move.json passed false -> its ok rows plus move_resume.json's; every file in exactly one ok row."""
    row = world.rows[0]
    rows = world.move_rows(row)
    first, rest = rows[:3], rows[3:]
    record = world.record_dir("method", row)
    base = dict(schema=H.MOVE_SCHEMA, src_dir=str(world.work_cell(row)), dst_dir=str(world.cell("method", row)), utc=H.utc(), error=None, rule="synthetic")
    (record/H.MOVE_NAME).write_text(json.dumps(dict(base, files=first+[dict(rest[0], ok=False, problem="copying", sha256_after=None)], passed=False, problems=["x"])))
    assert not classify(world, "method").written                   # half moved, no move_resume.json
    (record/H.MOVE_RESUME_NAME).write_text(json.dumps(dict(base, files=rest, passed=True, problems=[])))
    cell = classify(world, "method")
    assert cell.written and cell.move["files"]["terminal.json"]["record"].endswith(H.MOVE_NAME if "terminal.json" in [r["file"] for r in first] else H.MOVE_RESUME_NAME)
    (record/H.MOVE_RESUME_NAME).write_text(json.dumps(dict(base, files=rest+first[:1], passed=True, problems=[])))
    cell = classify(world, "method")
    assert not cell.written and "two ok rows" in cell.problem
    (record/H.MOVE_RESUME_NAME).write_text(json.dumps(dict(base, files=rest, passed=False, problems=["y"])))
    assert not classify(world, "method").written
    (record/H.MOVE_RESUME_NAME).write_text(json.dumps(dict(base, files=rest, passed=True, problems=[])))
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0 and pair_record(context, row)["arms"]["method"]["move"]["files"]["output/disp.npz"]["finalized_sha256"]


def test_a_method_file_at_another_sha_than_the_finalized_record_refuses_at_c(world):
    """The move and finalized records agree, the file on D: does not: the manifest (written from the files) lists another sha256 than the finalized record."""
    row = world.rows[1]
    path = world.cell("method", row)/"output"/"line.json"
    path.write_text(json.dumps(dict(line_status="hx_certified", extra=1)))
    world.write_manifest()                                           # the world's finalized and move records still name the old sha256
    rewrite(move_path(world, row), lambda d: None)
    exc = refused(world, "manifest")
    assert "finalized record names" in str(exc)


# ---- the step-5 (甲), the withdrawal, the method's P20 (prereg revision 2 items 14, 17 (f), 18; A5b items 5, 7) ------------------------------------------------
def home_copy(world, row, name, document):
    folder = world.review_root/row[3]
    folder.mkdir(parents=True, exist_ok=True)
    (folder/name).write_text(json.dumps(document), encoding="utf-8")
    return folder/name


def test_a_step5_implementation_error_is_written_by_bz_s_jia_of_arm_review(world):
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status="hx_implementation_error", field=False)
    review = home_copy(world, row, "review.json", dict(line_status="hx_implementation_error", finalized=False, implementation_error=dict(kind="a")))
    assert not classify(world, "method").written
    world.write_authorization(arm, row, "no_resume", as_arm="review", record=str(review), record_sha256=file_sha256(review), reason="C22.5-6 (a)，BZ (甲)：补成恒等")
    cell = classify(world, "method")
    assert cell.written and cell.no_output_class == "implementation" and cell.authorization["arm"] == "review" and review in cell.needed
    assert cell.move["finalized"]["bz_decision"]["sha256"] == cell.authorization["sha256"]
    rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]].update(bz_decision=None))
    cell = classify(world, "method")
    assert not cell.written and "bz_decision" in cell.problem                     # the finalized record must name the very record


def test_a_no_resume_of_arm_review_whose_record_is_not_on_home_is_not_taken(world):
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status="hx_implementation_error", field=False)
    inside = world.cell("method", row)/"review"/"review.json"                     # the cell's own copy (it moves to D:): not the /home record
    world.write_authorization(arm, row, "no_resume", as_arm="review", record=str(inside), record_sha256=file_sha256(inside), reason="C22.5-6 (a)，BZ (甲)")
    cell = classify(world, "method")
    assert not cell.written


def test_a_withdrawal_is_written_with_its_no_resume_record(world):
    """hb_review.withdraw: terminal.json hx_implementation_error (rerun_record the authorization), the record the /home copy of the newest review_not_finalized; the entry reads it as
    written, no output (C22.5-6 (b), (甲))."""
    arm, row = world.arm("method"), world.rows[1]
    stopped = home_copy(world, row, "review_not_finalized.json", dict(seq=1, kind="rounds", reason="synthetic", code=21))
    path = world.write_authorization(arm, row, "no_resume", as_arm="review", record=str(stopped), record_sha256=file_sha256(stopped),
                                     reason="D111 (ii) 撤回；C22.5-6 (b)，(甲)")
    world.write_cell(arm, row, status="hx_implementation_error", field=False,
                     extra=dict(rerun_record=dict(path=str(path.relative_to(world.repo)), sha256=file_sha256(path)), output_dir=None))
    cell = classify(world, "method", 1)
    assert cell.written and cell.no_output_class == "implementation" and cell.authorization["record"] == str(stopped)
    world.write_manifest()
    code, context = evaluate(world)
    entry = pair_record(context, row)["arms"]["method"]
    assert code == 0 and entry["outcome"] == "no_output" and entry["authorization"]["arm"] == "review"


def test_the_method_s_p20_is_written_only_by_bz_s_no_resume_naming_p20(world):
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status="hx_provenance_rejected", field=False)
    step4 = world.record_dir("method", row)/"step4.json"
    assert not classify(world, "method").written
    world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="BZ 不批重跑")
    cell = classify(world, "method")
    assert not cell.written and "P20" in cell.problem
    world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="P20：BZ 不批重跑")
    cell = classify(world, "method")
    assert cell.written and cell.no_output_class == "bz_no_rerun" and cell.cause == "P20" and cell.rejudgment is None


@pytest.mark.parametrize("status, defective, sound", [
    ("hx_implementation_error", "BZ (甲)：补成恒等", "C22.5-6 (a)，BZ (甲)：补成恒等、其余续跑"),            # (甲) without the case of C22.5-6, then a whole one
    ("hx_provenance_rejected", "BZ 不批重跑", "P20：BZ 不批重跑")])                                      # P20 not named, then named
def test_a_defective_decision_then_a_whole_one_hb_and_the_entry_take_the_same(world, status, defective, sound):
    """A5b review I2 (the lead's ruling 11-10 ~9:22 PM CST): a cell with first a defective no_resume record (kept: records are only added) and then a whole one. HB's selection
    (hb_core.bz_decision_auth, which the world's finalized record uses) takes the whole one; the entry takes the record at the finalized bz_decision's sha256 and checks it by the same
    criterion (bayesreg.io.rejudgment), so both name the second.  Before the whole one, HB names none and the entry does not write the cell."""
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status=status, field=False)
    step4 = world.record_dir("method", row)/"step4.json"
    first = world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason=defective)
    assert world.bz_decision(row) is None and not classify(world, "method").written
    second = world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason=sound)
    decided = world.bz_decision(row)
    cell = classify(world, "method")
    assert decided["sha256"] == file_sha256(second) and decided["step"] == 4 and cell.written
    assert cell.authorization["sha256"] == decided["sha256"] == cell.move["finalized"]["bz_decision"]["sha256"] and cell.authorization["path"] != str(first)


def test_two_whole_decisions_the_entry_takes_the_one_hb_named_not_its_own_pick(world):
    """A5b review I2: the entry no longer picks among sound records itself.  Two whole (甲) records; the finalized record names the second (as if HB had taken it): the entry takes the
    second; a finalized bz_decision naming no record of the cell, or a record of another step's arm, writes nothing."""
    arm, row = world.arm("method"), world.rows[0]
    world.write_cell(arm, row, status="hx_implementation_error", field=False)
    step4 = world.record_dir("method", row)/"step4.json"
    one = world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="C22.5-6 (a)，BZ (甲)")
    two = world.write_authorization(arm, row, "no_resume", record=str(step4), record_sha256=file_sha256(step4), reason="C22.5-6 (a)，BZ (甲)：同上")
    assert world.bz_decision(row)["sha256"] == file_sha256(one)                     # HB takes the first in name order
    rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]].update(bz_decision=dict(step=4, path=str(two), sha256=file_sha256(two))))
    cell = classify(world, "method")
    assert cell.written and cell.authorization["sha256"] == file_sha256(two)
    rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]].update(bz_decision=dict(step=5, path=str(two), sha256=file_sha256(two))))
    cell = classify(world, "method")
    assert not cell.written and "step 5" in cell.problem
    rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]].update(bz_decision=dict(step=4, path="x", sha256="0"*64)))
    assert not classify(world, "method").written
    rewrite(world.review_root/H.FINALIZED_NAME, lambda d: d["pairs"][row[3]].update(bz_decision=dict(step=3, path=str(two), sha256=file_sha256(two))))
    cell = classify(world, "method")
    assert not cell.written and "names no bz_decision" in cell.problem                # a step that is neither 4 nor 5: no record is taken


def test_the_decision_criterion_is_one_for_hb_holdout_common_and_the_entry():
    """A5b review I2: bayesreg.io.rejudgment holds the criterion; hb_core.decision_reason_problem delegates to it (the cell's case known), the entry calls it with the case unknown."""
    from ._holdout_world import hb_core
    hb = hb_core()
    assert RJ.p20_reason_problem("P20：BZ 不批重跑") is None and RJ.p20_reason_problem("BZ 不批重跑") is not None
    for reason, kind, which, ok in [("C22.5-6 (a)，BZ (甲)", "a", "(甲)", True), ("C22.5-6 (a)，BZ（甲）", "a", "(甲)", True), ("C22.5-6 (a)，BZ (甲)", "b", "(甲)", False),
                                    ("C22.5-6 (a)，BZ (甲)", None, "(甲)", True), ("BZ (甲)", None, "(甲)", False), ("C22.5-6 (a) 或 C22.5-6 (b)，BZ (甲)", None, "(甲)", False),
                                    ("C22.5-6 (a)，BZ (甲)，不选 (乙)", "a", "(甲)", False), ("C22.5-6 (b)，BZ (乙)", "b", "(乙)", True)]:
        assert (RJ.decision_reason_problem(reason, kind, which) is None) is ok, (reason, kind, which)
        if kind is not None:
            assert (hb.decision_reason_problem(dict(reason=reason), kind, which) is None) is ok
    assert hb.decision_reason_problem(dict(reason="C22.5-6 (a)，BZ (甲)"), None, "(甲)") is not None          # HB's queue needs the cell's case
    assert H.P20_WORD == RJ.P20_WORD and H.JIA == RJ.CHOICE_WORDS["(甲)"]


# ---- the rejudgment records (prereg revision 2 item 18; A5b item 6) ---------------------------------------------------------------------------------------
def rejudge(world, arm_name, row, record, judgment, *, written_by="lead", words="", status="needs_lead_rejudgment", k=None, commit_it=True):
    folder = world.repo/RJ.REJUDGMENT_DIR_REL/arm_name
    folder.mkdir(parents=True, exist_ok=True)
    k = k or 1+len(list(folder.glob(f"{row[3]}_*.json")))
    path = folder/f"{row[3]}_{k}.json"
    path.write_text(json.dumps(dict(schema=RJ.REJUDGMENT_SCHEMA, arm=arm_name, pair=row[3], pair_set="holdout", status=status,
                                    record=dict(path=str(record), sha256=file_sha256(record)), evidence=[], judgment=judgment, reason="synthetic evidence",
                                    written_by=written_by, words=words, time_local="11-10 2:00 PM CST", utc=H.utc())), encoding="utf-8")
    if commit_it:
        commit(world.repo, f"rejudgment {path.name}")
    return path


@pytest.mark.parametrize("judgment, writer, words, cls", [("unit_limit", "lead", "", "unit_limit"), ("no_output", "BZ", "（合成测试）BZ：不再跑", "bz_no_rerun")])
def test_a_needs_lead_rejudgment_cell_is_written_by_its_rejudgment(world, judgment, writer, words, cls):
    row = world.rows[0]
    record = world.write_cell(world.arm("ants"), row, status="needs_lead_rejudgment", field=False)
    assert not classify(world, "ants").written
    rejudge(world, "ants", row, record, judgment, written_by=writer, words=words)
    cell = classify(world, "ants")
    assert cell.written and cell.outcome == "no_output" and cell.no_output_class == cls and cell.rejudgment["judgment"] == judgment and cell.cause == "rejudgment"
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0 and pair_record(context, row)["arms"]["ants"]["rejudgment"]["judgment"] == judgment


@pytest.mark.parametrize("how", ["lead_writes_no_output", "uncommitted", "record_changed", "other_status", "extra_key"])
def test_a_rejudgment_that_does_not_hold_writes_nothing(world, how):
    row = world.rows[0]
    record = world.write_cell(world.arm("ants"), row, status="needs_lead_rejudgment", field=False)
    if how == "lead_writes_no_output":
        rejudge(world, "ants", row, record, "no_output", written_by="lead", words="x")
    elif how == "uncommitted":
        rejudge(world, "ants", row, record, "unit_limit", commit_it=False)
    elif how == "record_changed":
        rejudge(world, "ants", row, record, "unit_limit")
        record.write_text(record.read_text()+" ")
    elif how == "other_status":
        rejudge(world, "ants", row, record, "unit_limit", status="infra_failure")
    else:
        path = rejudge(world, "ants", row, record, "unit_limit", commit_it=False)
        rewrite(path, lambda d: d.update(resume=None))
        commit(world.repo, "a rejudgment with a resume key")
    assert not classify(world, "ants").written


def test_an_infrastructure_rejudgment_writes_only_after_the_resume_ran(world):
    arm, row = world.arm("ants"), world.rows[1]
    first = world.write_cell(arm, row, status="needs_lead_rejudgment", field=False)
    rejudge(world, "ants", row, first, "infrastructure")
    cell = classify(world, "ants", 1)
    assert not cell.written and "only classifies" in cell.problem
    world.write_authorization(arm, row, "resume", previous_record=str(first), previous_record_sha256=file_sha256(first))
    second = world.write_cell(arm, row, status="needs_lead_rejudgment", field=False, resume=True, extra=dict(attempt=2))
    assert not classify(world, "ants", 1).written                       # the rejudgment binds the first record, not the resumed one
    rejudge(world, "ants", row, second, "infrastructure")
    cell = classify(world, "ants", 1)
    assert cell.written and cell.no_output_class == "infrastructure" and cell.authorization["kind"] == "resume" and cell.rejudgment["k"] == 2


def test_the_reader_is_one_for_the_entry_and_holdout_common():
    """One format, one reader (N1): bayesreg.io.rejudgment's names are holdout_common's."""
    hc = H._holdout_common()
    assert (RJ.REJUDGMENT_DIR_REL, RJ.REJUDGMENT_SCHEMA, RJ.REJUDGMENT_KEYS, RJ.JUDGMENTS) == (hc.REJUDGMENT_DIR_REL, hc.REJUDGMENT_SCHEMA, hc.REJUDGMENT_KEYS, hc.JUDGMENTS)
    assert RJ.BZ_WRITERS == hc.RESUME_WRITERS and RJ.SAFE.pattern == hc.SAFE.pattern


@pytest.mark.parametrize("hold", [True, False])
def test_holdout_common_s_rejudgment_wraps_the_core(world, monkeypatch, hold):
    """A5b item 6 (second half): holdout_common.rejudgment, the name the XB chain and HB's configuration C queue call, wraps bayesreg.io.rejudgment.rejudgment: the same record and the same
    problems; an arm that is not a plain name still refuses there (safe_name)."""
    hc = H._holdout_common()
    row = world.rows[0]
    record = world.write_cell(world.arm("ants"), row, status="needs_lead_rejudgment", field=False)
    rejudge(world, "ants", row, record, "unit_limit", commit_it=hold)
    real, calls = RJ.rejudgment, []
    monkeypatch.setattr(RJ, "rejudgment", lambda *a, **k: calls.append(k["root"]) or real(*a, **k))
    core, problems = real("ants", row[3], record=record, root=world.repo)
    wrapped, wrapped_problems = hc.rejudgment("ants", row[3], record=record, root=world.repo)
    assert calls == [world.repo] and wrapped_problems == problems and bool(problems) is not hold
    assert (wrapped is None and core is None) if not hold else (wrapped == {k: core[k] for k in ("path", "sha256", "k", "document")} and wrapped["k"] == 1)
    with pytest.raises(hc.Refused):
        hc.rejudgment("../ants", row[3], record=record, root=world.repo)
    assert calls == [world.repo]


# ---- P20 of the other arms: BZ's no_resume of the newest refusal copy (prereg revision 2 item 17 (f); A5b item 7) ----------------------------------------------
def refusal_copies(world, name, row, seqs):
    folder = world.record_dir(name, row)/H.REFUSAL_COPIES
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, seq in enumerate(seqs):
        path = folder/f"xb_holdout_2026111{k}__refusal_{k}.json"
        path.write_text(json.dumps(dict(schema="bayesreg.holdout.refusal.v1", arm=name, pair=row[3], pair_set="holdout", stage="admission", seq=seq)), encoding="utf-8")
        path.with_name(path.name+".sha256").write_text(f"{file_sha256(path)}  /batches/x/refusals/{k}.json\n")
        paths.append(path)
    return paths


def test_a_cell_refused_before_its_lock_is_written_by_bz_s_no_resume_of_the_newest_copy(world):
    arm, row = world.arm("fga"), world.rows[0]
    (world.record_dir("fga", row)/"result.json").unlink()
    older, newer = refusal_copies(world, "fga", row, (1, 2))
    assert not classify(world, "fga").written
    world.write_authorization(arm, row, "no_resume", record=str(older), record_sha256=file_sha256(older), reason="P20：BZ 不批再跑")
    cell = classify(world, "fga")
    assert not cell.written and str(newer) in cell.problem               # the older copy is not the newest refusal
    world.write_authorization(arm, row, "no_resume", record=str(newer), record_sha256=file_sha256(newer), reason="P20：BZ 不批再跑")
    cell = classify(world, "fga")
    assert cell.written and cell.no_output_class == "bz_no_rerun" and cell.cause == "P20" and cell.authorization["newest_refusal_copy"]["seq"] == 2
    assert {older, newer} <= set(cell.needed)                             # every copy is read: inputs of the manifest
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0 and pair_record(context, row)["arms"]["fga"]["cause"] == "P20"


def test_refusal_copies_without_distinct_sequence_numbers_tell_no_newest(world):
    arm, row = world.arm("fga"), world.rows[0]
    (world.record_dir("fga", row)/"result.json").unlink()
    _, newer = refusal_copies(world, "fga", row, (2, 2))
    world.write_authorization(arm, row, "no_resume", record=str(newer), record_sha256=file_sha256(newer), reason="P20")
    cell = classify(world, "fga")
    assert not cell.written and "sequence number" in cell.problem


# ---- the repair ablation (prereg revision 2 item 4; C24.2; A5b item 3) ---------------------------------------------------------------------------------------
def test_the_ablation_pairs_are_decided_before_any_truth_and_written_in_the_lock_line(world):
    good = world.write_prerepair(world.rows[0])
    world.write_prerepair(world.rows[1], passed=False, checks=dict(field_sha256=True, u_sha256=True, tapered_u_sha256=False), disp=False)    # before the manifest (C24.2-4)
    world.write_manifest()
    admission, context = world.open()
    line = json.loads(Path(H.MASTER_LOCK).read_text())
    assert line["ablation"]["pairs"] == [world.rows[0][3]] and "tapered_u_sha256" in line["ablation"]["excluded"][world.rows[1][3]]
    assert world.reads == []
    spy = context["spy"]
    record_reads = [k for k, e in enumerate(spy.events) if e["path"].endswith(H.PREREPAIR_RECORD)]
    assert record_reads and all(spy.events[k]["phase"] == "admission" for k in record_reads)          # read in (b), before (c) and any truth
    assert H.evaluate_holdout(admission, context) == 0
    record = pair_record(context, world.rows[0])
    ablation = record["ablation"]
    assert ablation["in_ablation"] and ablation["complete"] and ablation["prerepair"]["certificate"] is None and good["output"]["sha256"] == ablation["prerepair"]["field"]["file_sha256"]
    for key in H.ABLATION_KEYS:
        assert ablation["after"][key] == record["arms"]["method"]["metrics"][key]
        assert ablation["difference"][key] == pytest.approx(ablation["after"][key]-ablation["before"][key])
    assert "ablation" not in record["arms"] and pair_record(context, world.rows[1])["ablation"]["in_ablation"] is False


def test_the_ablation_is_never_certified(world, monkeypatch):
    calls = {}

    def counting(name, fn):
        def wrapped(*a, **k):
            calls[name] = calls.get(name, 0)+1
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(cert_eval, "certify_field", counting("certify_field", cert_eval.certify_field))
    monkeypatch.setattr(H, "certify_field_bl1", counting("bl1", H.certify_field_bl1))
    code, _ = evaluate(world)
    without = dict(calls)
    calls.clear()
    Path(H.MASTER_LOCK).unlink()
    for row in world.rows:
        world.write_prerepair(row)
    world.write_manifest()
    for p in Path(H.PAIR_LOCK_DIR).glob("*.lock"):
        p.unlink()
    code2, context = evaluate(world, out=Path(H.EVAL_DIR)/"second")
    assert code == code2 == 0 and calls == without and all(pair_record(context, r)["ablation"]["in_ablation"] for r in world.rows)


def test_an_ablation_that_cannot_be_read_is_recorded_only_and_pq_the_pair_and_the_decision_go_on(world, monkeypatch):
    """A5b review I1 (the lead's ruling, 11-10 ~9:22 PM CST): PQ is evaluated before the ablation; a FieldError reading a prerepair_disp.npz (another grid, a
    non-finite value, no key d) is written into record['ablation'] only -- the pair is complete, its PQ numbers are there, the run exits 0, holdout_decide writes
    its file and lists the pair outside the ablation tables."""
    from bayesreg.eval.fields import FieldError
    for row in world.rows:
        world.write_prerepair(row)
    world.write_manifest()
    real, order = H.read_disp, []

    def broken(path, shape=None):
        if Path(path).name == H.PREREPAIR_DISP:
            order.append("ablation")
            raise FieldError(f"{path}: d is not on the pair's grid (test)")
        return real(path, shape)
    real_pq = H.evaluate_pq
    monkeypatch.setattr(H, "read_disp", broken)
    monkeypatch.setattr(H, "evaluate_pq", lambda *a, **k: order.append("pq") or real_pq(*a, **k))
    code, context = evaluate(world)
    assert code == 0 and order == ["pq", "ablation"]*len(world.rows)
    summary = json.loads((context["out"]/"holdout_eval.json").read_text())
    assert summary["failed"] == [] and summary["evaluated"] == [r[3] for r in world.rows]
    for row in world.rows:
        record = pair_record(context, row)
        assert record["complete"] is True and record["pq"]["complete"] is True and record["pq"]["acceptance"] is not None
        assert record["ablation"]["in_ablation"] is True and record["ablation"]["complete"] is False and "FieldError" in record["ablation"]["error"]
    decided = D.decide(D.collect([context["out"]]), world.table, world.rows, sdlogj_guard=.02, idx=D.draw_idx(n=2, b=100))
    assert decided["ablation"]["tables"]["main"]["n_pairs"] == 0 and all("did not complete" in v for v in decided["ablation"]["excluded"].values())


@pytest.mark.parametrize("how", ["byte", "truth", "spy"])
def test_a_stop_in_the_ablation_still_stops_the_whole_evaluation(world, monkeypatch, how):
    """A5b diff review minor 2: the three exceptions that stop everything are not kept in the ablation section.  A byte of prerepair_disp.npz changed after the manifest (and after the
    admission: the entry's own verify_files at the read raises ManifestChanged), or a TruthMismatch / SpyRefused raised at the read: exit 4, stopped.json names it, the pair's record is
    kept incomplete and the next pair is not reached."""
    for row in world.rows:
        world.write_prerepair(row)
    world.write_manifest()
    admission, context = world.open()
    disp = Path(H.PREREPAIR_BIG)/world.rows[0][3]/H.PREREPAIR_DISP
    if how == "byte":
        disp.chmod(0o644)
        disp.write_bytes(disp.read_bytes()+b"\0")
        expect = "ManifestChanged"
    else:
        error = H.TruthMismatch("truth_grid", "synthetic (test)") if how == "truth" else H.SpyRefused("synthetic open refused (test)")
        expect = type(error).__name__
        real = H.read_disp
        monkeypatch.setattr(H, "read_disp", lambda path, shape=None: (_ for _ in ()).throw(error) if Path(path).name == H.PREREPAIR_DISP else real(path, shape))
    assert H.evaluate_holdout(admission, context) == 4
    stopped = json.loads((context["out"]/"stopped.json").read_text())
    summary = json.loads((context["out"]/"holdout_eval.json").read_text())
    assert stopped["pair"] == world.rows[0][3] and stopped["reason"].startswith(expect) and summary["stopped"].startswith(expect)
    assert summary["evaluated"] == [] and summary["failed"] == [] and world.rows[1][3] in summary["not_reached"]
    assert not (context["out"]/"pairs"/f"{world.rows[1][3]}.json").exists()
    first = pair_record(context, world.rows[0])
    assert first["complete"] is False and first["error"].startswith(expect) and first["pq"]["complete"] is True      # PQ ran first (I1); the stop came at the ablation


def test_a_broken_ablation_section_changes_nothing_else_in_the_decision(world):
    """A5b review I1: holdout_decide's ablation section is computed apart; a malformed one (a number that is not one, a section that is not a mapping) writes an
    error into the ablation section only and the rest of the decision file is the same byte for byte."""
    for row in world.rows:
        world.write_prerepair(row)
    world.write_manifest()
    code, context = evaluate(world)
    records = D.collect([context["out"]])
    idx = D.draw_idx(n=2, b=100)
    good = D.decide(records, world.table, world.rows, sdlogj_guard=.02, idx=idx)
    rest = lambda doc: json.dumps({k: v for k, v in doc.items() if k != "ablation"}, sort_keys=True, default=str)       # noqa: E731
    for how in ("not_a_number", "not_a_mapping"):
        bad = copy.deepcopy(records)
        for r in bad.values():
            if how == "not_a_number":
                r["ablation"]["difference"] = {k: "x" for k in H.ABLATION_KEYS}
            else:
                r["ablation"]["before"] = ["not", "a", "mapping"]
        out = D.decide(bad, world.table, world.rows, sdlogj_guard=.02, idx=idx)
        assert code == 0 and rest(out) == rest(good) and out["ablation"]["tables"] is None and out["ablation"]["error"]


def test_a_preparation_record_that_does_not_hold_leaves_the_pair_out_and_f1_f3_go_on(world):
    row = world.rows[0]
    world.write_prerepair(row, inputs={n: dict(path="x", sha256="0"*64) for n in ("t88/field_0600.npz", "t88/run.json", "repair/repair.json")})
    world.write_manifest()
    code, context = evaluate(world)
    assert code == 0 and "part.t88_repair" in pair_record(context, row)["ablation"]["reason"]


def test_a_preparation_record_not_in_the_manifest_refuses_like_every_input(world):
    world.write_prerepair(world.rows[0])
    world.write_manifest(drop=str(Path(H.PREREPAIR_ROOT)/world.rows[0][3]/H.PREREPAIR_RECORD))
    refused(world, "manifest")


def test_the_ablation_is_not_an_arm_a_table_with_one_more_arm_is_refused_before_any_truth(world):
    """C24.2-4: the pinned arm table holds no ablation arm; a table with one is another sha256 than the pin."""
    table = copy.deepcopy(world.table)
    extra = copy.deepcopy(next(a for a in table["arms"] if a["name"] == "ants"))
    extra["name"] = "ablation_prerepair"
    table["arms"].append(extra)
    pinned = world.table_sha
    (world.repo/H.ARM_TABLE_REL).write_text(json.dumps(table, indent=1))
    commit(world.repo, "a table with an ablation arm")
    assert file_sha256(world.repo/H.ARM_TABLE_REL) != pinned
    refused(world, "arms")


def test_holdout_decide_s_ablation_changes_no_decision(world):
    for row in world.rows:
        world.write_prerepair(row)
    world.write_manifest()
    code, context = evaluate(world)
    records = D.collect([context["out"]])
    idx = D.draw_idx(n=2, b=200)
    with_ablation = D.decide(records, world.table, world.rows, sdlogj_guard=.02, idx=idx)
    assert with_ablation["ablation"]["tables"]["main"]["n_pairs"] == 2 and with_ablation["ablation"]["tables"]["main"]["endpoints"]["seg35"]["interval"] is not None
    stripped = copy.deepcopy(records)
    for r in stripped.values():
        r.pop("ablation", None)
    changed = copy.deepcopy(records)
    for r in changed.values():
        r["ablation"]["difference"] = {k: 123.0 for k in H.ABLATION_KEYS}
        r["ablation"]["before"] = {k: -5.0 for k in H.ABLATION_KEYS}
    rest = lambda doc: json.dumps({k: v for k, v in doc.items() if k != "ablation"}, sort_keys=True, default=str)       # noqa: E731
    for other in (stripped, changed):
        assert rest(D.decide(other, world.table, world.rows, sdlogj_guard=.02, idx=idx)) == rest(with_ablation)
    assert D.decide(stripped, world.table, world.rows, sdlogj_guard=.02, idx=idx)["ablation"]["tables"]["main"]["n_pairs"] == 0
    assert D.ABLATION_KEYS == H.ABLATION_KEYS
