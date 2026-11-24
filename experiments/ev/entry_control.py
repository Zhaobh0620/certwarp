"""The evaluation-entry control before the unsealing (preregistration 5.1 item 8, as the lead reworded it on 11-07, change 2: "入口用的指标代码路径对 HE 冒烟
0438-0439 的 disp.npz 复算"; runs/20261107-ev/eval_control_plan.md, its sha256 pinned at P0 (11-14) in PLAN_SHA256).

    python -B experiments/ev/entry_control.py run | dry-run

bayesreg.eval.holdout refuses a development pair at (0) / (a), so the control calls only its third stage, the metric function score_cell -- the same function
object the entry calls for every hold-out cell (SCORE is holdout.score_cell; tested) -- on the HE smoke v2 output of OASIS_OAS1_0438_MR1 <- 0439 (arm a:
production launcher, C on, hx_certified), with the truth the HE evaluation unit used: the development pair's pair.npz through the bench (truth.admit_pair,
truth.load_truth; the pair and both subjects also through the second lock of experiments/unv2/eval_seg.admit_subjects).  The result must equal the metrics,
per_label and official_row of HE's evaluation.json number for number (NaN = NaN).  The certificate is not compared: HE's evaluation unit computed the review
certificate of C22.4-1 (tile 65536), the entry's cert_eval statistics use tile 8192.
Order: plan pinned -> the two HE files at their pinned sha256 -> code tree clean -> admission (no label) -> new output directory -> machine gate (own unit
tag-ev-control) -> the one-time lock (O_EXCL) -> score_cell -> HE's evaluation.json read and compared -> entry_control.json.  Development data only; the
numbers are no result (the record keeps the differing keys, not the values).
"""
import os

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402
import sys                                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/"experiments"/"ev"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ev_common as ev                                             # noqa: E402
from bayesreg.eval import holdout as H                             # noqa: E402

SCORE = H.score_cell             # the entry's third stage, the same object (revision: the lead's change 2)
PLAN_REL = "runs/20261107-ev/eval_control_plan.md"
PLAN_SHA256 = "0e70920c2f86d0585a47410522b92a2fb68e897783b10243f5fe6ce4bcd7e259"     # P0 (11-14): the final plan (PT 5ef8243); None or another plan: the control does not run
STAGE = "stage_L2R_val_v1"
FIXED, MOVING = "OASIS_OAS1_0438_MR1", "OASIS_OAS1_0439_MR1"
PAIR = H.PAIR_NAME.format(fixed=FIXED, moving=MOVING)
HE_PAIR_DIR = Path("/home/rmcfz/seq_vp_tag_results/he_smoke_v2/a/0438-0439")
HE_FILES = {str(HE_PAIR_DIR/"output"/"disp.npz"): "7e18d768a05c2f8b9ebbe440bfd9a2d64a4e28cc64ada9bf38bfec26f380fd85",
            str(HE_PAIR_DIR/"evaluation.json"): "4f719cbdefd09b0658a555e98a47b6bee58803002f92c72372261cadf3ccd015"}
OUT = H.RESULTS/"ev"/"entry_control_v1"
LOCK = H.RESULTS/"ev"/"entry_control.lock"
OWN_UNITS = ("tag-ev-control",)
READ_SAME, READ_DIFFERENT, READ_UNDECIDED = "逐数相同", "不同（交总负责人）", "不判（交总负责人）"
COMPARED = (("metrics", "metrics"), ("per_label", "per_label"), ("official", "official_row"))


def admit(data_root=None, allowlist_path=None):
    """The development pair through the bench's guard (names, manifest, the allowlist) and the second lock of eval_seg.admit_subjects on both subjects."""
    from bayesreg.eval import truth as T
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    allowlist = load_dev_allowlist(ROOT/H.ALLOWLIST_REL if allowlist_path is None else allowlist_path)
    sealed = T.load_sealed_lists()
    training = set(load_folds(ROOT/H.FOLDS_REL).training_subjects)
    source = next((s for s in allowlist.sources if s["name"] == STAGE), None)
    for subject in (FIXED, MOVING):
        if source is None or subject not in source["subjects"] or sealed.which(subject) is not None or subject in allowlist.never_for_training \
                or not allowlist.allows(subject) or subject in training:
            raise SubjectSealed(f"{subject}: not a development subject of the control")
    data = ROOT/"data"/"three_d" if data_root is None else Path(data_root)
    pair_dir, grid, metadata, guard = T.admit_pair(data, STAGE, PAIR, allowlist, sealed)
    if (guard["fixed_subject"], guard["moving_subject"]) != (FIXED, MOVING):
        raise ev.ToolRefused("admission", f"{PAIR}: the manifest's subjects are not {FIXED} <- {MOVING}")
    return pair_dir, grid, guard


def compare(record, evaluation):
    mine = ev.normalized(record)
    out = {}
    for ours, theirs in COMPARED:
        out[ours] = ev.same(mine.get(ours), evaluation.get(theirs), null_is_nan=True)
    return out


def run(*, out=None, run_unit=None, cgroup=H.PROC_CGROUP, dry=False, data_root=None, allowlist_path=None):
    plan = ev.pinned_file(PLAN_REL, PLAN_SHA256, "plan")
    ev.verify_list(HE_FILES)
    code = ev.code_state()
    if not code["clean"]:
        raise ev.ToolRefused("code", f"the code tree is not clean: {code['status'][:200]!r}")
    pair_dir, grid, guard = admit(data_root, allowlist_path)
    out = ev.new_out(OUT if out is None else out, H.RESULTS)
    if dry:
        print(f"dry run: plan {plan['sha256']}, HE files verified, {PAIR} admitted, out {out}; nothing read")
        return 0
    machine = H.machine_gate(run=run_unit, cgroup=cgroup, own_units=OWN_UNITS)
    line = dict(tool="experiments/ev/entry_control.py", plan_sha256=plan["sha256"], code_head=code["head"], utc=H.utc(), pid=os.getpid())
    H.take_lock(LOCK, line)
    out.mkdir(parents=True)
    from bayesreg.eval import fields, truth as T
    from bayesreg.eval.metrics import OASIS
    document = dict(schema="bayesreg.holdout.entry_control.v1", plan=plan, code=code, machine=machine, lock=dict(path=str(LOCK), line=line), pair=PAIR,
                    score_function=f"{SCORE.__module__}.{SCORE.__qualname__}", score_is_entry_function=SCORE is H.score_cell, utc_start=H.utc())
    try:
        truth, truth_record = T.load_truth(pair_dir, OASIS, grid, guard)
        info = fields.describe_run(HE_PAIR_DIR/"output", pair=PAIR, kind="disp")
        record = SCORE(info, grid, truth, cert_mode=None)
        ev.verify_list(HE_FILES)
        evaluation = json.loads(Path(HE_PAIR_DIR/"evaluation.json").read_text(encoding="utf-8"))
        identity = dict(pair=evaluation.get("pair") == PAIR, disp_sha256=(evaluation.get("disp") or {}).get("file_sha256") == record["field"]["disp"]["file_sha256"],
                        pair_npz_sha256=(evaluation.get("truth") or {}).get("pair_sha256") == truth_record["pair_sha256"])
        differ = compare(record, evaluation)
        same = all(identity.values()) and not any(differ.values()) and document["score_is_entry_function"]
        document.update(identity=identity, differing_keys=differ, reading=READ_SAME if same else READ_DIFFERENT)
    except Exception as exc:                                       # kept; no rerun
        import traceback
        document.update(reading=READ_UNDECIDED, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    document["utc_end"] = H.utc()
    H.write_once(out/"entry_control.json", document)
    print(f"reading: {document['reading']}")
    return 0 if document["reading"] == READ_SAME else 8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("run", "dry-run"))
    args = parser.parse_args(argv)
    try:
        return run(dry=args.command == "dry-run")
    except (ev.ToolRefused, H.HoldoutRefused, H.SubjectSealed) as exc:
        print(f"refused: {type(exc).__name__}: {exc}; nothing of a subject was read", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
