"""The truth-reader gate before the unsealing (contract C23.12-10; preregistration revision 1, item 15; runs/20261107-ev/truth_gate_plan.md, its sha256 pinned
at P0 (11-14) in bayesreg.eval.holdout.TRUTH_GATE_PLAN_SHA256).

    python -B experiments/ev/truth_gate.py run       (from a clean checkout of main; unit tag-ev-truthgate; CPU)
    python -B experiments/ev/truth_gate.py dry-run   (the checks before the lock only; nothing of a subject is read)

Two items, both must be bitwise identical:
  (i)  the entry's one reader (bayesreg.eval.holdout.read_truth) on the 20 development subjects of stage_L2R_val_v1 against the label members of the 19 stage
       pair.npz files (fixed_seg35, fixed_seg4, moving_seg35, moving_seg4; bayesreg.eval.truth.load_truth, the bench's reader): float64 on both sides (the type
       evaluate_field gets), each subject at every pair it is in; the stored dtypes recorded.  Reading 20/20 (seg35 and seg4).
  (ii) the entry's metric stage (holdout.score_cell, the same function the hold-out uses) with (i)'s arrays, against bayesreg.eval.cli with pair.npz's, on a set
       of existing development final fields: every metric, per-label value and certificate field of cli's metrics.json (timing left out) and the official
       aggregate of cli's summary.  Reading 19/19.
       P3's 19 final fields (/home/rmcfz/seq_vp_tag_results/p3_v1) cannot be given to bayesreg.eval.cli as they are (fields.detect_kind: its run directories
       have cmd.txt but neither config.json, protocol.json nor disp.npz; the final field is the tapered T88 field with repair.npz's node values, rebuilt only by
       the P3 evaluation code).  So the set is configuration C's A0 of D0 (d0_v1/runs/A0, 19 configC run directories, cli-readable as they are; re-evaluated by
       cli before, M2V1 ref_reeval), pinned file by file in runs/20261107-ev/truth_gate_fields.sha256 (FIELDS_SHA256).  Not run for this gate.
Order: plan pinned -> field list pinned and every listed file at its sha256 -> code tree clean -> admission (holdout.admit_truth_gate: the 20 subjects, the
development allowlist source, no T0, training or sealed subject; the 19 pairs through truth.admit_pair; no label) -> new output directory -> machine gate (own
unit tag-ev-truthgate) -> the one-time lock (holdout.TRUTH_GATE_LOCK, O_EXCL) -> (i) -> (ii) -> the reading -> truth_gate.json.  A failure keeps everything; no
rerun (交 BZ).  The development numbers compared here are no result: the record keeps only whether they are identical and which keys differ.
"""
import os

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402
import sys                                                         # noqa: E402

import numpy as np                                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/"experiments"/"ev"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ev_common as ev                                             # noqa: E402
from bayesreg.eval import holdout as H                             # noqa: E402

FIELDS_REL = "runs/20261107-ev/truth_gate_fields.sha256"
FIELDS_SHA256 = "93cb3f20be07fc4a267ac311b9a9bad82266d577f49336016a8e77210f581c58"     # P0 (11-14): pinned with the plan; None or another list: the gate does not run
FIELD_ROOT = Path("/home/rmcfz/seq_vp_tag_results/d0_v1/runs/A0")
METHOD, ARM = "configC_frozen_eb808be_90s", "A0"                   # cli's labels of these runs (experiments/m2v1/reeval_refs.py)
SKIP = ("seconds",)                                                # timing, at any depth of the certificate record
COMPARED = ("metrics", "per_label", "certificate", "nonfinite_metrics")


def run_dirs(rows):
    return [FIELD_ROOT/row["row"][3] for row in rows]


def check_inputs():
    """The plan and the field list at their pinned sha256; the list covers the 19 run directories of FIELD_ROOT only; every listed file at its sha256
    (ev.verify_list: B4 review, 3rd plan, B1 -- before, the list was read but no file of it hashed).  Returns (plan, fields, listed, the verify_list record)."""
    plan = ev.pinned_file(H.TRUTH_GATE_PLAN_REL, H.TRUTH_GATE_PLAN_SHA256, "plan")
    fields = ev.pinned_file(FIELDS_REL, FIELDS_SHA256, "fields")
    listed = ev.sha256_list(ROOT/FIELDS_REL)
    covered = {Path(p).relative_to(FIELD_ROOT).parts[0] for p in listed if Path(p).is_relative_to(FIELD_ROOT)}
    want = {H.PAIR_NAME.format(fixed=f, moving=m) for f, m in H.DEV_PAIRS}
    if covered != want or len(listed) != len({p for p in listed if Path(p).is_relative_to(FIELD_ROOT)}):
        raise ev.ToolRefused("fields", f"{FIELDS_REL}: the 19 run directories of {FIELD_ROOT} only (got {sorted(covered ^ want)} differing)")
    verified = ev.verify_list(listed)
    return plan, fields, listed, verified


def item_i(admission, rows):
    """(i): per pair and side, the reader's arrays against pair.npz's members (float64), bit for bit; returns (per-subject rows, reader arrays per pair)."""
    from bayesreg.eval import truth as T
    from bayesreg.eval.metrics import OASIS
    lookup = H.seg4_map()
    per_subject, arrays = {}, {}
    for entry in rows:
        _, fixed, moving, name = entry["row"]
        bench, bench_record = T.load_truth(entry["pair_dir"], OASIS, entry["grid"], entry["guard"])
        mine = {}
        for side, subject in (("fixed", fixed), ("moving", moving)):
            values, record = H.read_truth(subject, admission=admission, grid_metadata=entry["metadata"], pair_name=name)
            mine[side] = values
            for kind in ("seg35", "seg4"):
                theirs = np.asarray(bench[f"{side}_{kind}"], dtype=np.float64)
                same = theirs.shape == values[kind].shape and np.array_equal(np.ascontiguousarray(theirs).view(np.uint64), np.ascontiguousarray(values[kind]).view(np.uint64))
                row = dict(pair=name, side=side, kind=kind, bitwise=bool(same), npz_dtype=str(np.asarray(bench[f"{side}_{kind}"]).dtype),
                           nifti_stored_dtype=record[kind]["stored_dtype"], nifti_sha256=record[kind]["sha256"], shape_zyx=record[kind]["shape_zyx"],
                           differing_voxels=None if theirs.shape != values[kind].shape else int(np.count_nonzero(theirs != values[kind])))
                per_subject.setdefault(subject, []).append(row)
            per_subject[subject].append(dict(pair=name, side=side, seg4_equals_G=H.seg4_equals_g(values["seg35"], values["seg4"], lookup)))
        arrays[name] = (H.truth_of_pair(mine["fixed"], mine["moving"]), bench_record)
    identical = sorted(s for s, items in per_subject.items() if all(r.get("bitwise", True) for r in items))
    return dict(subjects=per_subject, identical=identical, n=len(per_subject), n_identical=len(identical)), arrays


def item_ii(admission, rows, arrays, out):
    """(ii): bayesreg.eval.cli on the 19 run directories (its own truth from pair.npz), then holdout.score_cell on the same directories with (i)'s arrays."""
    from bayesreg.eval import cli, fields, metrics
    cli_dir = out/"cli"
    cli_dir.mkdir()
    argv = [str(p) for p in run_dirs(rows)]+["--pairs-stage", H.DEV_STAGE, "--out", "metrics.json", "--long", str(cli_dir/"long.csv"), "--summary",
                                             str(cli_dir/"summary.json"), "--out-dir", str(cli_dir/"runs"), "--method", METHOD, "--arm", ARM,
                                             "--data-root", admission.record["data_root"], "--allowlist", admission.record["allowlist_path"], "--cert", "auto"]
    code = cli.main(argv)
    pairs, mine_rows = [], {}
    for entry in rows:
        name = entry["row"][3]
        info = fields.describe_run(FIELD_ROOT/name)
        truth, _ = arrays[name]
        cert_mode = cli.AUTO_CERT[info.kind]
        record = ev.normalized(H.score_cell(info, entry["grid"], truth, cert_mode=cert_mode))
        run_id = info.run_id_hint or (FIELD_ROOT/name).resolve().name
        theirs_path = cli_dir/"runs"/run_id/"metrics.json"
        theirs = json.loads(theirs_path.read_text(encoding="utf-8")) if theirs_path.is_file() else None
        mine = json.loads(H.dumps({k: record.get(k) for k in COMPARED}))
        differ = ["(cli wrote no metrics.json)"] if theirs is None else ev.same(mine, {k: theirs.get(k) for k in COMPARED}, skip=SKIP)
        pairs.append(dict(pair=name, run_id=run_id, kind=info.kind, field_role=info.field_role, identical=not differ, differing_keys=differ[:50]))
        mine_rows.setdefault(info.field_role, []).append(record["official"])
    summary = json.loads((cli_dir/"summary.json").read_text(encoding="utf-8")) if (cli_dir/"summary.json").is_file() else {"groups": []}
    aggregates = []
    for group in summary.get("groups", []):
        mine = json.loads(H.dumps(metrics.aggregate(mine_rows.get(group["field_role"], [])))) if mine_rows.get(group["field_role"]) else None
        aggregates.append(dict(field_role=group["field_role"], identical=mine is not None and not ev.same(mine, group.get("official_aggregate"))))
    identical = [p["pair"] for p in pairs if p["identical"]]
    all_aggregates = bool(aggregates) and all(a["identical"] for a in aggregates)
    return dict(cli_exit=code, pairs=pairs, aggregates=aggregates, n=len(pairs), n_identical=len(identical) if all_aggregates else 0,
                aggregates_identical=all_aggregates)


def reading(i, ii, complete):
    if not complete:
        return H.GATE_READING_UNDECIDED
    if i["n_identical"] == 20 and i["n"] == 20 and ii["n_identical"] == 19 and ii["n"] == 19 and ii["cli_exit"] == 0:
        return H.GATE_READING_PASS
    return H.GATE_READING_FAIL


def run(out=None, *, run_unit=None, cgroup=H.PROC_CGROUP, dry=False, data_root=None, allowlist_path=None):
    plan, field_list, listed, verified = check_inputs()                 # every listed file at its sha256 before the admission (B4 review B1)
    code = ev.code_state()
    if not code["clean"]:
        raise ev.ToolRefused("code", f"the code tree is not clean: {code['status'][:200]!r}")
    admission, rows = H.admit_truth_gate(data_root=data_root, allowlist_path=allowlist_path)
    out = ev.new_out(H.TRUTH_GATE_OUT if out is None else out, H.RESULTS)
    if Path(out) != Path(H.TRUTH_GATE_OUT).resolve():
        raise ev.ToolRefused("output", f"the gate writes {H.TRUTH_GATE_OUT} only (TRUTH_READER_DECISION points there)")
    if dry:
        print(f"dry run: plan {plan['sha256']}, fields {field_list['sha256']} ({verified['files']} files verified by sha256), admission {len(rows)} pairs, out {out}; no label or truth file read")
        return 0
    machine = H.machine_gate(run=run_unit, cgroup=cgroup, own_units=H.GATE_OWN_UNITS)
    import nibabel
    environment = dict(numpy=np.__version__, nibabel=nibabel.__version__)
    line = dict(tool="experiments/ev/truth_gate.py", plan_sha256=plan["sha256"], fields_sha256=field_list["sha256"], code_head=code["head"], utc=H.utc(), pid=os.getpid())
    lock = H.take_gate_lock(admission, line)
    out.mkdir(parents=True)
    document = dict(schema=H.SCHEMA_GATE, plan=plan, fields=field_list, fields_verified=verified, code=code, environment=environment, machine=machine, lock=lock, utc_start=H.utc())
    complete = False
    try:
        i, arrays = item_i(admission, rows)
        document["fields_verified_before_ii"] = ev.verify_list(listed)     # again after the lock, before (ii) reads them (entry_control's way; B4 review B1)
        ii = item_ii(admission, rows, arrays, out)
        complete = True
    except Exception as exc:                                       # kept; no rerun (交 BZ)
        import traceback
        i = ii = None
        document.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    text = reading(i, ii, complete)
    document.update(item_i=i, item_ii=ii, reading=text, utc_end=H.utc(),
                    counts=dict(reader=dict(subjects=20, identical=i["n_identical"] if i else 0), metrics=dict(pairs=19, identical=ii["n_identical"] if ii else 0)))
    H.write_once(out/"truth_gate.json", document)
    print(f"reading: {text}")
    return 0 if text == H.GATE_READING_PASS else 8


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
