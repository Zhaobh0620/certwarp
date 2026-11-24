"""M2V1 batch summary (runs/20261031-m2v1-dev/plan.md): summary.json / summary.md from the files only. Read only on
every run, evaluation and reference directory.

    python -B experiments/m2v1/summarize.py --root DIR [--d0-long F] [--a4-long F] [--smoke-root DIR] [--ref-reeval DIR]

Inputs:
  <root>/batch.json        written by experiments/m2v1/run_batch.py at its start: stage, pairs, the D47 pair, per arm the
                           configuration and the run_pair argument values it implies, eta and its source record
  <root>/runs/<arm>/<short>/   run_pair outputs (status.json, config.json, hyper.json, log.jsonl)
  <root>/eval/<arm>/<short>/   bayesreg.eval.cli output (done.json, <run_id>/metrics.json) or eval_failed.json
  <root>/launch/run_<arm>-<short>_*.txt   launch records (a pair without run directory and without launch record was not
                           run: deadline or batch not reached; "未运行", not a failure)
  --ref-reeval  the re-evaluation of A0 and A4' by bayesreg.eval.cli at 68721d1 (experiments/m2v1/reeval_refs.py;
             default <root>/ref_reeval): <arm>/<short>/done.json and one <run_id>/metrics.json per pair -- the reference
             numbers of the comparisons (the lead, 10-31: one evaluation entry)
  --d0-long  D0 long.csv (A0: configuration C, frozen eb808be, 90 s CPU; experiments/d0/evaluate.py), read only: the pair
             ids and run ids the re-evaluation must match, and the original numbers shown as a reference column
  --a4-long  A4' long.csv (arm A4p: configuration C + U-Net posterior colouring), read only, the same use
  --smoke-root  the GPU smoke v2 root: H2 (0438<-0439 only, eta_tmp, 2700 s) as a descriptive reference row
Per pair and arm: run state (not_run, failed_no_status, ran), run status / wording / exit code, field role, eval error,
seg35, seg4, HD95, official and brain SDlogJ, the evaluator's certificate (passed, min J lower bound), the config.json
args check against batch.json (every value the arm implies) and the eta check, and the per-level readouts of
experiments/m2v1/level_table.py.
Comparisons (dev arm only, against A0 and A4' separately; seg35 and seg4 side by side): per pair d = dev - reference over
the pairs where the dev row is comparable (field_role native_formal or disp_only, evaluator certificate passed, value
finite) and the re-evaluated reference is comparable and finite; median of d, number of d > 0, n. The long.csv files
must hold exactly the batch's pairs (pair ids checked; method and arm checked), and every pair's re-evaluation must
exist with that pair and the long.csv run id; otherwise the summary stops.
Reading labels (plan.md, preregistered; a label, not a conclusion; not an equal-time comparison): n = comparable pairs,
N = pairs of the batch; with n >= ceil(15 N / 19) (15 of 19): above = median d >= .01 and #(d > 0) >= ceil(14 n / 19);
below = median d <= -.01 and #(d > 0) <= floor(5 n / 19); otherwise middle; fewer pairs: describe only.
Convergence table: per run per level the ending (达阈 / 上限收尾 / 预算收尾 / 受阻收尾 / other) and GF_B/T_B at the end;
counts of runs per ending, blocked levels, exported fields and evaluator certificates passed (hard constraint 1: every
exported field must pass).
D47 table (0438<-0439): the D47 arm (jd_cell_estimate gauss27+corners), the dev arm on the same pair (certificate) and the
smoke H2 reference, per level, with the reading flags of plan.md (lean gauss27+corners / keep certificate / no
distinction). The lead decides; any change of D47 goes through the contract (C09.1-3a).
Written once: summary.json / summary.md; if they exist and the recomputed content differs, summary_<UTC>.json / .md are
written next to them (nothing overwritten) and the difference is reported.
"""
import argparse
import csv
from datetime import datetime, timezone
import glob
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.eval.fields import COMPARABLE_ROLES  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


levels = _load("_m2v1_level_table", "experiments/m2v1/level_table.py")

SCHEMA = "bayesreg.m2v1.summary.v1"
D0_LONG = Path("/home/rmcfz/seq_vp_tag_results/d0_v1/analysis/long.csv")
A4_LONG = Path("/home/rmcfz/seq_vp_tag_results/d0_v1_a4/analysis/long.csv")
SMOKE_ROOT = Path("/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke_v2")
REF_METHOD = "configC_frozen_eb808be_90s"
REFERENCES = (("A0", "A0", "D0 A0：配置 C（冻结 eb808be，CPU，90 s）"),
              ("A4p", "A4′", "D0 补充臂 A4′：配置 C + U-Net 后验上色（CPU，90 s）"))
MEASURES = (("seg35", "seg35"), ("seg4", "seg4"), ("hd95", "HD95"), ("sdlogj_official", "SDlogJ 官方"),
            ("sdlogj_brain", "SDlogJ 脑内"))
COMPARE = ("seg35", "seg4")
LABEL_DIFF = .01
NOTE = ("开发集（stage_L2R_val_v1）单次观测。每次运行以开发开关 --wall-seconds 的“预算收尾”或其他 C13-3 收尾结束，"
        "不是“B 线收敛”，除非状态写明。与 A0、A4′ 的比较不是等时比较：本批 {backend} {device}、每次 {wall:g} s；A0、A4′ "
        "是冻结求解器 CPU 90 s。η 来自 T0 调参对的扫描（与本批的对不重叠）。读数标签不是结论。")


class SummaryError(Exception):
    """The summary cannot be written as specified (a reference file does not match the batch)."""


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def short_name(pair):
    fixed, _, moving = pair.partition("__")
    return f"{fixed.split('_')[2]}-{moving.split('_')[2]}"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- references ----------------------------------------------------------------------------------------------------
def reference(path, arm, pairs):
    """{pair: {measure: value, run_id}} of one arm of a D0 long.csv; the pair set must be the batch's."""
    path = Path(path)
    if not path.is_file():
        raise SummaryError(f"reference {arm}: {path} missing")
    out, methods = {}, set()
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("arm") != arm:
                continue
            methods.add(row.get("method"))
            entry = out.setdefault(row["pair"], dict(run_id=row.get("run_id")))
            if row.get("run_id") != entry["run_id"]:
                raise SummaryError(f"reference {arm}: {row['pair']} has two run ids ({entry['run_id']}, {row['run_id']})")
            measure = row.get("measure")
            if measure in entry:
                raise SummaryError(f"reference {arm}: {row['pair']} {measure} listed twice in {path}")
            try:
                entry[measure] = float(row.get("value"))
            except (TypeError, ValueError):
                entry[measure] = None
    if methods != {REF_METHOD}:
        raise SummaryError(f"reference {arm}: methods {sorted(map(str, methods))} in {path}, expected {REF_METHOD}")
    if set(out) != set(pairs):
        raise SummaryError(f"reference {arm}: pairs of {path} differ from the batch (missing "
                           f"{sorted(set(pairs)-set(out))}, extra {sorted(set(out)-set(pairs))})")
    return dict(path=str(path), sha256=file_sha256(path), arm=arm, method=REF_METHOD, pairs=out)


def reeval_reference(ref_dir, arm, pairs, original):
    """{pair: {measure: value, run_id, field_role, comparable, route}} of one arm re-evaluated by
    experiments/m2v1/reeval_refs.py (bayesreg.eval.cli at 68721d1): <ref_dir>/<arm>/<short>/done.json and exactly one
    <run_id>/metrics.json per pair, whose pair and run id must be the batch's pair and the original long.csv's run id.
    A missing re-evaluation stops the summary (the lead asked for one evaluation entry)."""
    ref_dir, out = Path(ref_dir), {}
    for pair in pairs:
        ev = ref_dir/arm/short_name(pair)
        found = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        done = read_json(ev/"done.json")
        if done is None or len(found) != 1:
            raise SummaryError(f"re-evaluation {arm} {short_name(pair)}: {ev}/done.json {done is not None}, "
                               f"{len(found)} metrics.json (experiments/m2v1/reeval_refs.py)")
        record = read_json(found[0]) or {}
        rid = original["pairs"][pair]["run_id"]
        if record.get("pair") != pair or record.get("run_id") != rid:
            raise SummaryError(f"re-evaluation {found[0]}: pair {record.get('pair')!r}, run id {record.get('run_id')!r}; "
                               f"expected {pair!r}, {rid!r}")
        m, c = record.get("metrics") or {}, record.get("certificate") or {}
        out[pair] = dict(run_id=rid, field_role=record.get("field_role"), route=done.get("route"), metrics_file=found[0],
                         comparable=bool(not record.get("error") and record.get("field_role") in COMPARABLE_ROLES
                                         and c.get("passed") is True),
                         **{k: (m.get(k) if finite(m.get(k)) else None) for k, _ in MEASURES})
    compare = ref_dir/"compare.json"
    return dict(source="bayesreg.eval.cli 重评（experiments/m2v1/reeval_refs.py）", dir=str(ref_dir), arm=arm,
                compare=str(compare) if compare.is_file() else None,
                compare_sha256=file_sha256(compare) if compare.is_file() else None,
                original=dict(path=original["path"], sha256=original["sha256"], method=original["method"]), pairs=out)


def smoke_reference(root, pair):
    """The smoke v2 H2 arm (descriptive; one pair). None when absent or on another pair."""
    root = Path(root)
    run = root/"H2"
    config = read_json(run/"config.json")
    if config is None or (config.get("args") or {}).get("pair", "").split("/")[-1] != pair:
        return None
    found = sorted(glob.glob(str(root/"eval_H2"/"*"/"metrics.json")))
    record = read_json(found[0]) if len(found) == 1 else None
    m = (record or {}).get("metrics") or {}
    c = (record or {}).get("certificate") or {}
    return dict(run_dir=str(run), metrics_file=found[0] if len(found) == 1 else None,
                metrics={k: (m.get(k) if finite(m.get(k)) else None) for k, _ in MEASURES},
                field_role=(record or {}).get("field_role"), certificate=dict(passed=c.get("passed"),
                                                                              min_jd_lower_bound=c.get("min_jd_lower_bound")),
                levels=levels.run_record(run, "H2（冒烟 v2）"))


# ---- batch rows ------------------------------------------------------------------------------------------------------
def args_check(args, expected):
    if not isinstance(args, dict):
        return None
    return {k: [args.get(k), v] for k, v in expected.items() if args.get(k) != v}


def eta_check(config_eta, eta):
    if not config_eta:
        return None
    ok = (config_eta.get("alpha") == eta["alpha"] and config_eta.get("nu") == eta["nu"]
          and len(config_eta.get("tau") or []) == len(eta["tau"])
          and all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(config_eta["tau"], eta["tau"])))
    return ok


def pair_row(root, arm, pair, batch):
    root, short = Path(root), short_name(pair)
    run, ev = root/"runs"/arm/short, root/"eval"/arm/short
    launches = sorted(glob.glob(str(root/"launch"/f"run_{arm}-{short}_*.txt")))
    status, config = read_json(run/"status.json"), read_json(run/"config.json")
    if not run.exists() and not launches:
        state = "not_run"
    elif status is None:
        state = "failed_no_status"
    else:
        state = "ran"
    found = sorted(glob.glob(str(ev/"*"/"metrics.json")))
    record = read_json(found[0]) if (ev/"done.json").is_file() and len(found) == 1 else None
    m, c = (record or {}).get("metrics") or {}, (record or {}).get("certificate") or {}
    eval_state = ("done" if record is not None else "failed" if (ev/"eval_failed.json").is_file() else
                  "none")
    arm_doc = batch["arms"][arm]
    differ = args_check((config or {}).get("args"), arm_doc["expected_args"])
    # exported = an evaluated field exists (M2V1 review I1): the evaluator's field_present, or, without an evaluation,
    # disp.npz in the run directory. status.json "exported" also holds the state hash of final_certificate_rejected and
    # stopped_before_native runs, which have no disp.npz and no field.
    exported = (bool(record.get("field_present")) if record is not None else (run/"disp.npz").is_file())
    row = dict(arm=arm, pair=pair, short=short, state=state, run_dir=str(run), launches=launches,
               run_status=(status or {}).get("status"), wording=(status or {}).get("wording"),
               exit_code=(status or {}).get("exit_code"), native=(status or {}).get("native"),
               exported=exported, status_exported_hash=(status or {}).get("exported"), eval_state=eval_state,
               eval_metrics=found[0] if record is not None else None, run_id=(record or {}).get("run_id"),
               field_role=(record or {}).get("field_role"), eval_error=(record or {}).get("error"),
               metrics={k: (m.get(k) if finite(m.get(k)) else None) for k, _ in MEASURES},
               certificate=dict(mode=c.get("mode"), status=c.get("status"), passed=c.get("passed"),
                                min_jd_lower_bound=c.get("min_jd_lower_bound")) if c else None,
               args_differ=differ, eta_ok=eta_check((config or {}).get("eta"), batch["eta"]),
               levels=levels.run_record(run, f"{arm}-{short}") if run.exists() else None)
    row["comparable"] = bool(record is not None and not row["eval_error"] and row["field_role"] in COMPARABLE_ROLES
                             and (row["certificate"] or {}).get("passed") is True)
    return row


def min_pairs(n_batch):
    """The fewest comparable pairs that get a label: 15 of 19 (the lead, 10-31, before any result), ceil(15 N / 19) for
    a batch of N pairs (dry runs)."""
    return math.ceil(15*n_batch/19)


def reading_label(stats, n_batch):
    """Plan reading rule (preregistered): with n comparable pairs, n >= min_pairs(n_batch): above if median d >= .01 and
    #(d > 0) >= ceil(14 n / 19); below if median d <= -.01 and #(d > 0) <= floor(5 n / 19); middle otherwise. Fewer
    pairs: describe only."""
    n, need = stats["n"], min_pairs(n_batch)
    if n < need:
        return f"n = {n} < {need}：只描述"
    med, pos = stats["median"], stats["positive"]
    if med >= LABEL_DIFF and pos >= math.ceil(14*n/19):
        return "高于（开发集读数，不是结论）"
    if med <= -LABEL_DIFF and pos <= math.floor(5*n/19):
        return "低于（开发集读数，不是结论）"
    return "中间区域（只描述）"


def paired(rows, ref, measure, n_batch):
    diffs = []
    for row in rows:
        entry = ref["pairs"].get(row["pair"], {})
        value, other = row["metrics"].get(measure), entry.get(measure)
        if row["comparable"] and entry.get("comparable", True) and finite(value) and finite(other):
            diffs.append(dict(pair=row["pair"], ours=value, reference=other, d=value-other))
    d = [x["d"] for x in diffs]
    stats = dict(measure=measure, n=len(d), n_batch=n_batch, median=statistics.median(d) if d else None,
                 positive=sum(x > 0 for x in d), negative=sum(x < 0 for x in d), zero=sum(x == 0 for x in d),
                 mean_ours=statistics.fmean([x["ours"] for x in diffs]) if d else None,
                 mean_reference=statistics.fmean([x["reference"] for x in diffs]) if d else None, per_pair=diffs)
    stats["label"] = reading_label(stats, n_batch) if d else "无可比对"
    return stats


# ---- D47 flags -------------------------------------------------------------------------------------------------------
def level_map(record):
    return {(r["level"], r["round"]): r for r in ((record or {}).get("levels") or [])}


def d47_flags(gc, cert):
    """The plan's reading flags for the D47 control (gc: the gauss27+corners arm's row, cert: the dev arm's row on the
    same pair). Returns dict(lean_gc=[reasons], keep_certificate=[reasons], verdict)."""
    lean, keep = [], []
    ok = (lambda row: row is not None and row["state"] == "ran" and row["comparable"])
    if gc is None or cert is None or gc["levels"] is None or cert["levels"] is None:
        return dict(lean_gc=[], keep_certificate=[], verdict="缺一臂：无法对照")
    g, c = level_map(gc["levels"]), level_map(cert["levels"])
    if gc["levels"]["n_blocked"]:
        keep.append(f"gauss27+corners 臂有受阻收尾的层 {gc['levels']['blocked']}")
    if not ok(gc):
        keep.append(f"gauss27+corners 臂没有可比的导出场（{gc['state']}，{gc['run_status']}，field_role {gc['field_role']}）")
    gb, cb = (gc.get("certificate") or {}).get("min_jd_lower_bound"), (cert.get("certificate") or {}).get(
        "min_jd_lower_bound")
    if finite(gb) and finite(cb) and gb < 1e-6 <= cb:
        keep.append(f"gauss27+corners 臂终态证书下界 {gb:.3g} < 1e-6（certificate 臂 {cb:.3g}）")
    if not keep:
        for key in sorted(set(g) & set(c)):
            a, b = g[key], c[key]
            if a["ending"] == "达阈" and b["ending"] == "上限收尾":
                lean.append(f"L{key[0]}：gauss27+corners 达阈（{a['accepted']} 步），certificate 撞上限")
        native = max((k for k in set(g) & set(c)), default=None)
        if native is not None and native[0] == 3:
            a, b = g[native], c[native]
            if finite(a["GF_B_over_T_B"]) and finite(b["GF_B_over_T_B"]) and a["GF_B_over_T_B"]*10 <= b["GF_B_over_T_B"]:
                lean.append(f"原生层末 GF_B/T_B：gauss27+corners {a['GF_B_over_T_B']:.3g}，certificate "
                            f"{b['GF_B_over_T_B']:.3g}（至少低 10 倍）")
            if (isinstance(a["accepted"], int) and isinstance(b["accepted"], int) and b["accepted"] > 0
                    and a["accepted"] >= 1.5*b["accepted"]):
                lean.append(f"原生层接受步：gauss27+corners {a['accepted']}，certificate {b['accepted']}（≥ 1.5 倍）")
        if not ok(cert):
            lean.append(f"certificate 臂没有可比的导出场（{cert['state']}，{cert['run_status']}）")
        if cert["levels"]["n_blocked"]:
            lean.append(f"certificate 臂有受阻收尾的层 {cert['levels']['blocked']}")
    verdict = ("倾向重新考虑 D47（交总负责人）" if lean else "支持维持 D47" if keep else "无法区分（只描述）")
    return dict(lean_gc=lean, keep_certificate=keep, verdict=verdict)


# ---- document --------------------------------------------------------------------------------------------------------
def summary(root, *, d0_long=D0_LONG, a4_long=A4_LONG, smoke_root=SMOKE_ROOT, ref_reeval=None):
    root = Path(root)
    batch = read_json(root/"batch.json")
    if batch is None:
        raise SummaryError(f"{root}/batch.json missing (the queue writes it at its start)")
    pairs, n = batch["pairs"], len(batch["pairs"])
    rows = {arm: [pair_row(root, arm, p, batch) for p in batch["arms"][arm]["pairs"]] for arm in batch["order"]}
    originals = {name: reference(path, name, pairs) for (name, _, _), path in zip(REFERENCES, (d0_long, a4_long))}
    ref_dir = Path(ref_reeval) if ref_reeval is not None else root/"ref_reeval"
    refs = {name: reeval_reference(ref_dir, name, pairs, originals[name]) for name, _, _ in REFERENCES}
    comparisons = {name: {m: paired(rows["dev"], refs[name], m, n) for m in COMPARE} for name, _, _ in REFERENCES}
    for name in comparisons:
        for m in COMPARE:
            comparisons[name][m]["label_rule"] = (
                f"参照取 bayesreg.eval.cli 重评；可比对数 n ≥ {min_pairs(n)} 时：高于 = 中位差 ≥ {LABEL_DIFF:g} 且为正 ≥ "
                f"⌈14n/19⌉；低于 = 中位差 ≤ −{LABEL_DIFF:g} 且为正 ≤ ⌊5n/19⌋；其余为中间区域；n < {min_pairs(n)} 只描述")
    all_rows = [r for arm in batch["order"] for r in rows[arm]]
    exported = [r for r in all_rows if r["exported"]]
    cert_pass = [r for r in exported if (r["certificate"] or {}).get("passed") is True]
    counts = dict(runs=len(all_rows), not_run=sum(r["state"] == "not_run" for r in all_rows),
                  failed_no_status=sum(r["state"] == "failed_no_status" for r in all_rows),
                  ran=sum(r["state"] == "ran" for r in all_rows),
                  exit_nonzero=sum(r["state"] == "ran" and r["exit_code"] != 0 for r in all_rows),
                  ran_without_export=sum(r["state"] == "ran" and not r["exported"] for r in all_rows),
                  exported=len(exported), evaluator_certificate_passed=len(cert_pass),
                  eval_failed=sum(r["eval_state"] == "failed" for r in all_rows),
                  blocked_runs=sum(bool(r["levels"] and r["levels"]["n_blocked"]) for r in all_rows),
                  args_mismatch=sum(bool(r["args_differ"]) for r in all_rows),
                  eta_mismatch=sum(r["eta_ok"] is False for r in all_rows),
                  run_status={})
    for r in all_rows:
        if r["state"] == "ran":
            counts["run_status"][r["run_status"]] = counts["run_status"].get(r["run_status"], 0)+1
    counts["hard_constraint_1"] = ("通过：每个导出场的评测证书都通过" if len(cert_pass) == len(exported) else
                                   f"**违反**：{len(exported)-len(cert_pass)} 个导出场的评测证书未通过")
    d47_pair = batch["d47_pair"]
    gc = next((r for r in rows.get("d47", []) if r["pair"] == d47_pair), None)
    cert = next((r for r in rows["dev"] if r["pair"] == d47_pair), None)
    smoke = smoke_reference(smoke_root, d47_pair) if smoke_root else None
    dev = batch["arms"]["dev"]["configuration"]
    note = NOTE.format(wall=batch["wall_seconds"], backend=dev.get("backend"), device=dev.get("device"))
    return dict(schema=SCHEMA, written_utc=utc(), note=note, root=str(root),
                batch=dict(path=str(root/"batch.json"), sha256=file_sha256(root/"batch.json"), code_sha=batch["code_sha"],
                           eta=batch["eta"], eta_source=batch["eta_source"]),
                rows=rows, counts=counts,
                references={k: dict(v, pairs=None) for k, v in refs.items()},
                reference_values={k: v["pairs"] for k, v in refs.items()},
                original_reference_values={k: v["pairs"] for k, v in originals.items()},
                comparisons=comparisons,
                d47=dict(pair=d47_pair, gauss27_corners=gc, certificate=cert, smoke_h2=smoke, flags=d47_flags(gc, cert)))


def fmt(value, nd=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{nd}g}" if abs(value) < 1e-3 and value != 0 else f"{value:.{nd}f}"
    return str(value)


STATE_TEXT = dict(not_run="未运行（截止或未轮到；不算失败）", failed_no_status="失败：单元结束但没有 status.json（被杀？）",
                  ran="已运行")


def level_cell(rec, level):
    rows = [r for r in (rec or {}).get("levels") or [] if r["level"] == level]
    if not rows:
        return "—"
    return "；".join(f"{r['ending']} {r['accepted']} 步 · {levels.fmt(r['GF_B_over_T_B'])}" for r in rows)


def markdown(doc):
    out = [f"# M2V1 开发批汇总（{doc['root']}）", "", doc["note"], "",
           f"代码 {doc['batch']['code_sha']}；η：α {doc['batch']['eta']['alpha']:.6g}，ν {doc['batch']['eta']['nu']:.6g}，"
           f"τ {doc['batch']['eta']['tau']}（{doc['batch']['eta_source']['source']}，sha256 "
           f"{doc['batch']['eta_source']['source_sha256'][:12]}…）", ""]
    c = doc["counts"]
    out += ["## 运行情况", "",
            f"- 计划 {c['runs']} 次：已运行 {c['ran']}，未运行 {c['not_run']}，单元结束无 status.json {c['failed_no_status']}，"
            f"退出码非 0 {c['exit_nonzero']}，评测失败 {c['eval_failed']}。",
            f"- 运行状态：{json.dumps(c['run_status'], ensure_ascii=False)}；有受阻收尾层的运行 {c['blocked_runs']}。",
            f"- 导出（评测器 field_present）{c['exported']}，评测证书通过 {c['evaluator_certificate_passed']}："
            f"{c['hard_constraint_1']}。运行结束但没有导出场的 {c['ran_without_export']} 次计为失败（没有可比导出，"
            "例如 final_certificate_rejected、stopped_before_native），不计入硬约束 1。",
            f"- config.json 参数与计划不符 {c['args_mismatch']}；η 不符 {c['eta_mismatch']}。", ""]
    head = ["臂", "对", "状态", "收尾", "field_role", *(t for _, t in MEASURES), "证书（评测）", "A0 seg35", "A0 seg4",
            "A4′ seg35", "A4′ seg4"]
    out += ["## 逐对指标", "", "| "+" | ".join(head)+" |", "|"+"---|"*len(head)]
    refs = doc["reference_values"]
    for arm, rows in doc["rows"].items():
        for r in rows:
            cert = r["certificate"] or {}
            a0, a4 = refs["A0"].get(r["pair"], {}), refs["A4p"].get(r["pair"], {})
            cells = [arm, r["short"], STATE_TEXT[r["state"]], f"{r['run_status']}（{r['wording']}，exit {r['exit_code']}）"
                     if r["state"] == "ran" else "—", r["field_role"] or "—",
                     *(fmt(r["metrics"].get(k), 3 if k == "hd95" else 4) for k, _ in MEASURES),
                     f"{cert.get('passed')} / {fmt(cert.get('min_jd_lower_bound'), 3)}" if cert else "—",
                     fmt(a0.get("seg35")), fmt(a0.get("seg4")), fmt(a4.get("seg35")), fmt(a4.get("seg4"))]
            out.append("| "+" | ".join(str(x).replace("|", "/") for x in cells)+" |")
    out += ["", "## 配对比较（dev 臂 − 参照；seg35 与 seg4 并列；不是等时比较）", "",
            "| 参照 | 指标 | n | 中位差 | 为正 | 为负 | 本批均值 | 参照均值 | 读数标签 |", "|---|---|---|---|---|---|---|---|---|"]
    for (name, label, _), comp in zip(REFERENCES, doc["comparisons"].values()):
        for m in COMPARE:
            s = comp[m]
            out.append(f"| {label} | {m} | {s['n']}/{s['n_batch']} | {fmt(s['median'])} | {s['positive']} | {s['negative']} | "
                       f"{fmt(s['mean_ours'])} | {fmt(s['mean_reference'])} | {s['label']} |")
    out += ["", f"读数规则：{doc['comparisons']['A0']['seg35']['label_rule']}。", "",
            "参照数字取 bayesreg.eval.cli 重评（" + "；".join(
                f"{v['arm']}：{v['dir']}，对照 {v['compare'] or '—'}" for v in doc["references"].values())
            + "）。原数字（experiments/d0/evaluate.py）只作对照列：" + "；".join(
                f"{v['original']['path']}（sha256 {v['original']['sha256'][:12]}…）"
                for v in doc["references"].values()), ""]
    head = ["对", *(f"{label} {m} 重评 / 原" for _, label, _ in REFERENCES for m in COMPARE), "A4′ 重评路线"]
    out += ["### 参照：重评与原数字", "", "| "+" | ".join(head)+" |", "|"+"---|"*len(head)]
    orig = doc["original_reference_values"]
    for pair in [r["pair"] for r in doc["rows"]["dev"]]:
        cells = [short_name(pair)]
        for name, _, _ in REFERENCES:
            for m in COMPARE:
                a, b = refs[name].get(pair, {}).get(m), orig[name].get(pair, {}).get(m)
                cells.append(f"{fmt(a)}（逐位相同）" if a is not None and a == b else f"{fmt(a)} / {fmt(b)}")
        cells.append(str(refs["A4p"].get(pair, {}).get("route")))
        out.append("| "+" | ".join(cells)+" |")
    out.append("")
    out += ["## 收敛表（每层：收尾、接受步、末 GF_B/T_B）", "", "| 臂 | 对 | L0 | L1 | L2 | L3（原生） | 运行收尾 |",
            "|---|---|---|---|---|---|---|"]
    for arm, rows in doc["rows"].items():
        for r in rows:
            rec = r["levels"]
            out.append(f"| {arm} | {r['short']} | " + " | ".join(level_cell(rec, lv) for lv in range(4))
                       + f" | {r['wording'] or STATE_TEXT[r['state']]} |")
    d = doc["d47"]
    out += ["", f"## D47 对照臂（{d['pair']}）", "",
            "同一对、同一 η、同一预算；只差 --jd-cell-estimate。H2 是冒烟 v2（η_tmp，2700 s），只作描述。", ""]
    recs = [x for x in ((d["gauss27_corners"] or {}).get("levels"), (d["certificate"] or {}).get("levels"),
                        (d["smoke_h2"] or {}).get("levels")) if x]
    out.append(levels.markdown(recs, title="D47 各层读数"))
    out += ["| 臂 | seg35 | seg4 | HD95 | SDlogJ 官方 | SDlogJ 脑内 | 证书（评测） |", "|---|---|---|---|---|---|---|"]
    for name, row in (("gauss27+corners（D47 臂）", d["gauss27_corners"]), ("certificate（开发批）", d["certificate"]),
                      ("H2 冒烟 v2（参照）", d["smoke_h2"])):
        if row is None:
            out.append(f"| {name} | 缺 |" + " — |"*5)
            continue
        cert = row.get("certificate") or {}
        out.append(f"| {name} | " + " | ".join(fmt(row["metrics"].get(k), 3 if k == "hd95" else 4) for k, _ in MEASURES)
                   + f" | {cert.get('passed')} / {fmt(cert.get('min_jd_lower_bound'), 3)} |")
    f = d["flags"]
    out += ["", f"读数：**{f['verdict']}**。", ""]
    out += [f"- 倾向 gauss27+corners：{x}" for x in f["lean_gc"]]
    out += [f"- 支持 certificate：{x}" for x in f["keep_certificate"]]
    return "\n".join(out)+"\n"


def strip_volatile(doc):
    return {k: v for k, v in doc.items() if k != "written_utc"}


def write(root, **kw):
    """summary.json / summary.md once; a different recomputation goes to summary_<UTC>.*; returns the paths."""
    root = Path(root)
    doc = summary(root, **kw)
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
    stored = read_json(root/"summary.json")
    if stored is None:
        target_json, target_md = root/"summary.json", root/"summary.md"
    elif json.loads(json.dumps(strip_volatile(doc))) == strip_volatile(stored):
        return root/"summary.json", root/"summary.md", "unchanged"
    else:
        s = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_json, target_md = root/f"summary_{s}.json", root/f"summary_{s}.md"
    with open(target_json, "x", encoding="utf-8") as stream:
        stream.write(text)
    with open(target_md, "x", encoding="utf-8") as stream:
        stream.write(markdown(doc))
    return target_json, target_md, "new" if stored is None else "differs (new file; the earlier one kept)"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/m2v1/summarize.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--d0-long", type=Path, default=D0_LONG)
    parser.add_argument("--a4-long", type=Path, default=A4_LONG)
    parser.add_argument("--smoke-root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--ref-reeval", type=Path, default=None,
                        help="the re-evaluation of A0 / A4' (experiments/m2v1/reeval_refs.py; default <root>/ref_reeval)")
    args = parser.parse_args(argv)
    try:
        js, md, state = write(args.root, d0_long=args.d0_long, a4_long=args.a4_long, smoke_root=args.smoke_root,
                              ref_reeval=args.ref_reeval)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 8
    print(f"summary: {js} {md} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
