"""M2 GPU smoke: one table of the arms next to the D0 A0 run (read only; experiments/m2/gpu_smoke.sh calls it last).

    python -B experiments/m2/compare_gpu_smoke.py --root DIR --arms G1,G2,G3 --a0 A0_METRICS [--a0-eval DIR]
        [--ref-root DIR --ref-arms G2,G3] --out-json F --out-md F

Per arm (the run directory <root>/<arm> and its evaluation <root>/eval_<arm>/<run_id>/metrics.json written by
bayesreg.eval.cli): run status and wording, exit code, field role; seg35, seg4, HD95, official SDlogJ, brain SDlogJ;
the evaluator's certificate of the exported field (mode, status, passed, min J lower bound) and the run's own cert.json;
per level the round status, accepted steps, commits, GF_B/T_B and seconds (hyper.json rounds); the CPU fallbacks of
the GPU-path comparison (hyper.json gpu_comparison: comparisons, cpu_fallback, fallback_ratio); the wall-clock budget
record; wall clocks (timing.json total, solve, export, certificate) and peak RSS.
Second batch (SMK2; runs/20261030-m2-gpu-smoke-v2/plan.md): --ref-root / --ref-arms add arms of an earlier root (the
v1 G2, G3; read only) as rows marked "(v1)" in the same tables, and every row carries
  - witness types (contract C05-2b "记录与 CLI", C16; status.json witness_summary, bayesreg.solve.witness): the
    JACOBIAN_VIOLATION outcomes of every certificate call of the run by the exact class of the witness point --
    on a sample point (one of the 53: corner, edge_mid, face_center, body_center), edge interior, face interior,
    cell interior, or no witness point -- and the W2 count (the witness cell's 53 sample points all J > 0) and the
    J <= 0 counts by point type (edge_mid: K* / W1 type). A run without witness_summary (code before IMPL27, e.g.
    the v1 arms) shows "未记录";
  - rejected candidates per level: the sum over the level's rounds of hyper.json rounds[].counters.rejections (all
    reasons, and JACOBIAN_VIOLATION apart; line-search candidates of the PG rounds only, not the transfer or the
    level-start certificates);
  - the levels that ended blocked (受阻收尾: a round status whose run wording is 受阻收尾 in
    bayesreg.solve.stopping.RUN_WORDING -- line_search_failed, trial_budget_exhausted, outer_no_progress,
    outer_inner_failed) and the levels whose round ended otherwise abnormally (exception, stagnated, ...); for a
    blocked round, the witness types of the JACOBIAN_VIOLATION candidates of its last line-search iteration (log.jsonl
    candidate rows, C16) -- the W2 count there is what D48 hands to BZ;
  - in the JSON also the witness types per level from log.jsonl (witness_by_level; PG line-search candidates only).
A0 (D0, configuration C frozen eb808be, 90 s; GS review item 8): when gpu_smoke.sh re-evaluated the A0 run directory with
bayesreg.eval.cli (--a0-eval, default <root>/eval_A0: done.json and exactly one <run_id>/metrics.json with a field and
without an error), the A0 metrics, evaluator certificate and field role come from that record -- the same evaluation
entry as the arms -- and the experiments/d0/evaluate.py numbers stay in the JSON (a0.d0_evaluate). Otherwise the A0 row
uses experiments/d0/evaluate.py (--a0, d0.metrics.v1: the same definitions, the run's own certificate; field role from
its metrics_role) and the table says so, with the reason.
Field role (GS review item 4): a column next to the status. Only rows whose field_role is comparable
(bayesreg.eval.fields.COMPARABLE_ROLES: native_formal, disp_only) and whose pair is A0's sit in the table next to A0;
handover_only (no native commit: the exported field is the certified native initializer), no_native_result, unknown,
arms without an evaluation and rows of another pair are listed apart in a second table, marked.
Every number is a development-set single observation; the GPU may be shared with the segmenter training, so the wall
clocks are records, not timing results. Missing files are reported as missing (a failed arm is listed, not dropped).
"""
import argparse
import glob
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.eval.fields import (COMPARABLE_ROLES, CONFIGC_METRICS_ROLES, ROLE_HANDOVER,  # noqa: E402
                                  ROLE_UNKNOWN)
from bayesreg.solve.stopping import (CAP_REACHED, CONVERGED_GRADIENT, MAX_ITERATIONS, RUN_WORDING,  # noqa: E402
                                     BUDGET_EXHAUSTED)
from bayesreg.solve.witness import OFF_SAMPLE_TYPES, SAMPLE_TYPES  # noqa: E402

SCHEMA = "bayesreg.m2_gpu_smoke.comparison.v3"     # v3 (SMK2): reference arms, witness types, rejections, blocked levels
MEASURES = (("seg35", "seg35"), ("seg4", "seg4"), ("hd95", "HD95"), ("sdlogj_official", "SDlogJ (official)"),
            ("sdlogj_brain", "SDlogJ (brain)"))
NOTE = ("开发集单次观测（一对：{pair}）；GPU 可能与分割器训练共用，墙钟只作记录，不作计时结论；"
        "臂的停止是开发开关 --wall-seconds 的“预算收尾”或其他 C13-3 收尾，不是“B 线收敛”，除非状态写明。")
HANDOVER_TEXT = "handover_only（原生层零提交：导出的是认证过的原生初值，不是原生正式状态）"
BLOCKED_STATUSES = tuple(sorted(s for s, w in RUN_WORDING.items() if w == "受阻收尾"))
NORMAL_ROUND_STATUSES = (CONVERGED_GRADIENT, CAP_REACHED, MAX_ITERATIONS, BUDGET_EXHAUSTED)
JV = "JACOBIAN_VIOLATION"
NO_WITNESS = "None"                                 # bayesreg.solve.witness.WitnessSummary key of "no witness point"
WITNESS_NOTE = ("见证点类型：status.json witness_summary（bayesreg.solve.witness）对本次运行全部证书调用（线搜索与转层候选、"
                "层起点、导出）中 JACOBIAN_VIOLATION 的计数，按见证点的精确类别分：采样点（53 点之一：角点、棱中点、面心、体心）"
                "/ 棱内 / 面内 / 体内 / 无见证点；棱上 = 棱中点 + 棱内（见证点在单元棱上，G3 诊断中 K* 的类型）；"
                "W2 = 见证单元 53 个采样点上 J 全正（证书在采样点之间找到违反）；"
                "“53 点 J ≤ 0”按点类型计数，其中 edge_mid 为 K* / W1 型。未记录 = 该运行的代码早于见证记录（IMPL27）。")
REJECT_NOTE = ("各层被拒候选：hyper.json 各轮 counters.rejections 按层求和，写作“全部（其中 JACOBIAN_VIOLATION）”；"
               "只含 PG 线搜索候选，不含转层与层起点的证书。受阻收尾的层：该层某轮的状态在 C13 的“受阻收尾”集合 "
               f"{', '.join(BLOCKED_STATUSES)} 中，括号里是该轮最后一次线搜索迭代中 JACOBIAN_VIOLATION 候选的见证点类型"
               "（log.jsonl）；其中 W2 型（53 点 J 全正）受阻按 D48 交 BZ。其他非常规轮状态（exception、stagnated 等）另列。")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def eval_metrics(root, arm):
    found = sorted(glob.glob(str(Path(root)/f"eval_{arm}"/"*"/"metrics.json")))
    if len(found) > 1:
        raise SystemExit(f"eval_{arm}: {len(found)} metrics.json files; expected one")
    return (found[0], read_json(found[0])) if found else (None, None)


def level_rows(hyper):
    out = []
    for r in (hyper or {}).get("rounds") or []:
        out.append(dict(level=r.get("level"), round=r.get("round"), status=r.get("status"), accepted=r.get("accepted"),
                        commits=r.get("commits"), GF_B_over_T_B=r.get("GF_B_over_T_B"), seconds=r.get("seconds"),
                        min_jd_lower_bound=r.get("min_jd_lower_bound"),
                        rejections=dict((r.get("counters") or {}).get("rejections") or {}),
                        candidates=(r.get("counters") or {}).get("candidates"),
                        jd_scaling_refresh=((r.get("metric") or {}).get("jd_scaling") or {}).get("jd_scaling_refresh")))
    return out


def rejections_by_level(levels):
    """{level: dict(total, jacobian_violation, by_reason)} summed over the level's rounds."""
    out = {}
    for r in levels:
        entry = out.setdefault(r["level"], dict(total=0, jacobian_violation=0, by_reason={}))
        for reason, count in (r.get("rejections") or {}).items():
            entry["total"] += count
            entry["by_reason"][reason] = entry["by_reason"].get(reason, 0)+count
            if reason == JV:
                entry["jacobian_violation"] += count
    return out


def level_endings(levels):
    """(blocked, other): the levels with a round status in BLOCKED_STATUSES, and those with any other status that is
    not a normal round ending (NORMAL_ROUND_STATUSES), each as [dict(level, round, status)]."""
    blocked, other = [], []
    for r in levels:
        entry = dict(level=r["level"], round=r["round"], status=r["status"])
        if r["status"] in BLOCKED_STATUSES:
            blocked.append(entry)
        elif r["status"] not in NORMAL_ROUND_STATUSES:
            other.append(entry)
    return blocked, other


def witness_record(summary):
    """The witness columns from status.json witness_summary (None: not recorded by the run's code)."""
    if summary is None:
        return None
    counts = dict(summary.get("witness_type_counts") or {})
    sample = {k: counts.get(k, 0) for k in SAMPLE_TYPES}
    known = set(SAMPLE_TYPES) | set(OFF_SAMPLE_TYPES) | {NO_WITNESS}
    return dict(certificates=summary.get("certificates"), jacobian_violations=summary.get("jacobian_violations"),
                sample_point=sum(sample.values()), sample_point_by_type=sample,
                on_edge=sample["edge_mid"]+counts.get("edge_interior", 0),     # witness point on a cell edge (K*-like)
                edge_interior=counts.get("edge_interior", 0), face_interior=counts.get("face_interior", 0),
                cell_interior=counts.get("cell_interior", 0), no_witness=counts.get(NO_WITNESS, 0),
                other_types={k: v for k, v in counts.items() if k not in known},
                w2_min53_positive=summary.get("min_j53_positive"),
                min53_nonpositive_by_point_type=dict(summary.get("min_j53_nonpositive_point_type_counts") or {}),
                witness_errors=summary.get("witness_errors"))


def _witness_counter():
    return dict(jacobian_violations=0, recorded=0, types={}, w2_min53_positive=0, min53_nonpositive_by_point_type={})


def _witness_count(entry, row):
    entry["jacobian_violations"] += 1
    if "witness_type" not in row:                   # a log written before the witness record (IMPL27)
        return
    entry["recorded"] += 1
    kind = str(row.get("witness_type"))
    entry["types"][kind] = entry["types"].get(kind, 0)+1
    min53 = row.get("witness_cell_min_j53")
    if min53 is None:
        return
    if min53.get("J") is not None and min53["J"] > 0:
        entry["w2_min53_positive"] += 1
    else:
        point = str(min53.get("point_type"))
        entry["min53_nonpositive_by_point_type"][point] = entry["min53_nonpositive_by_point_type"].get(point, 0)+1


def log_witness(path):
    """From log.jsonl (C16): the JACOBIAN_VIOLATION candidate rows per level (witness types, W2, J <= 0 by point type)
    and, per (level, round), those of the round's last line-search iteration -- the candidates that a blocked round
    could not get past. None without log.jsonl."""
    path = Path(path)
    if not path.is_file():
        return None
    by_level, last = {}, {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if '"candidate"' not in line or "JACOBIAN_VIOLATION" not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") != "candidate" or row.get("reason") != JV:
                continue
            level, key = row.get("level"), (row.get("level"), row.get("round"))
            _witness_count(by_level.setdefault(level, _witness_counter()), row)
            step = (row.get("outer") or 0, row.get("iteration") or 0)
            if key not in last or step > last[key][0]:
                last[key] = (step, _witness_counter())
            if step == last[key][0]:
                _witness_count(last[key][1], row)
    return dict(by_level=by_level,
                last_iteration={f"{lv}/{r}": dict(level=lv, round=r, outer_iteration=list(step), **entry)
                                for (lv, r), (step, entry) in last.items()})


def certificate_of(c):
    return dict(mode=c.get("mode"), status=c.get("status"), passed=c.get("passed"),
                min_jd_lower_bound=c.get("min_jd_lower_bound")) if c else None


def arm_record(root, arm, *, label=None, reference=False):
    run = Path(root)/arm
    status, hyper, timing = read_json(run/"status.json"), read_json(run/"hyper.json"), read_json(run/"timing.json")
    config, cert = read_json(run/"config.json"), read_json(run/"cert.json")
    path, metrics = eval_metrics(root, arm)
    m, c = (metrics or {}).get("metrics") or {}, (metrics or {}).get("certificate") or {}
    comparison = (hyper or {}).get("gpu_comparison") or {}
    args = (config or {}).get("args") or {}
    levels = level_rows(hyper)
    blocked, other = level_endings(levels)
    logged = log_witness(run/"log.jsonl")
    for entry in blocked:            # the witness types of the candidates the blocked round could not get past
        entry["last_iteration_witness"] = (None if logged is None else
                                           logged["last_iteration"].get(f"{entry['level']}/{entry['round']}"))
    registry = (status or {}).get("structure_registry")
    closed27 = args.get("logvol_points") == "gauss27+closed27"       # share / weights are argparse defaults otherwise
    return dict(
        arm=arm, label=label or arm, reference=reference, root=str(root), run_dir=str(run), eval_metrics=path,
        run_id=(metrics or {}).get("run_id"),
        options=dict(pair=args.get("pair"), channel_source=args.get("channel_source"), logvol_points=args.get("logvol_points"),
                     logvol_closed27_share=args.get("logvol_closed27_share") if closed27 else None,
                     closed27_weights=args.get("closed27_weights") if closed27 else None,
                     jd_cell_estimate=args.get("jd_cell_estimate"),
                     backend=args.get("backend"), device=args.get("device"), metric_jd_scaling=args.get("metric_jd_scaling"),
                     outer=args.get("outer"), rounds=args.get("rounds"), wall_seconds=args.get("wall_seconds"),
                     jacobian_cache=args.get("jacobian_cache"), jacobian_cache_log=args.get("jacobian_cache_log"),
                     speculative_gradient=args.get("speculative_gradient")),
        structure_registry=None if registry is None else dict(status=registry.get("status"), path=registry.get("path"),
                                                              git=registry.get("git")),
        status=(status or {}).get("status"), wording=(status or {}).get("wording"), exit_code=(status or {}).get("exit_code"),
        reason=(status or {}).get("reason"), native=(status or {}).get("native"),
        field_role=(metrics or {}).get("field_role"), field_present=(metrics or {}).get("field_present"),
        eval_error=(metrics or {}).get("error"),
        metrics={k: m.get(k) for k, _ in MEASURES},
        certificate=certificate_of(c),
        run_certificate=None if cert is None else dict(status=cert.get("status"), min_jd_lower_bound=cert.get(
            "min_jd_lower_bound"), state_hash=cert.get("state_hash"), witness_type=cert.get("witness_type")),
        levels=levels, rejections_by_level=rejections_by_level(levels), blocked_levels=blocked,
        other_abnormal_levels=other, witness=witness_record((status or {}).get("witness_summary")),
        witness_by_level=None if logged is None else logged["by_level"],
        fallback=dict(comparisons=comparison.get("comparisons"), cpu_fallback=comparison.get("cpu_fallback"),
                      fallback_ratio=comparison.get("fallback_ratio"), seconds=comparison.get("seconds"),
                      gf_rechecks=(comparison.get("gf_rechecks") or {}).get("count")),
        wall_budget=(status or {}).get("wall_budget"),
        wall=dict(total_seconds=(timing or {}).get("total_seconds"), solve=(timing or {}).get("solve"),
                  export=(timing or {}).get("export"), certificate_seconds=(timing or {}).get("certificate_seconds"),
                  witness_seconds=(timing or {}).get("witness_seconds"),
                  peak_rss_mb=(timing or {}).get("peak_rss_mb")),
        missing=[name for name, value in (("status.json", status), ("hyper.json", hyper), ("timing.json", timing),
                                          ("evaluation", metrics)) if value is None])


def a0_reevaluation(eval_dir):
    """((metrics path, record), None) of the bayesreg.eval.cli re-evaluation of A0, or (None, why not)."""
    if eval_dir is None:
        return None, "no --a0-eval directory given"
    eval_dir = Path(eval_dir)
    if not (eval_dir/"done.json").is_file():
        return None, f"{eval_dir}/done.json missing (not run, failed or skipped; see smoke.log)"
    found = sorted(glob.glob(str(eval_dir/"*"/"metrics.json")))
    if len(found) != 1:
        return None, f"{eval_dir}: {len(found)} metrics.json files, expected one"
    record = read_json(found[0])
    if record.get("error"):
        return None, f"{found[0]}: evaluation error {record['error']}"
    if not record.get("field_present") or not record.get("metrics"):
        return None, f"{found[0]}: no field evaluated (field_role {record.get('field_role')})"
    return (found[0], record), None


def a0_record(path, eval_dir=None):
    a0 = read_json(path)
    if a0 is None:
        raise SystemExit(f"A0 metrics {path} missing")
    m, c = a0.get("metrics") or {}, a0.get("certificate") or {}
    d0 = dict(path=str(path), schema=a0.get("schema"), evaluator=(a0.get("evaluator") or {}).get("script"),
              metrics_role=a0.get("metrics_role"),
              field_role=CONFIGC_METRICS_ROLES.get(a0.get("metrics_role"), ROLE_UNKNOWN),
              metrics={k: m.get(k) for k, _ in MEASURES},
              certificate=dict(mode="run certificate (d0 evaluate.py)", status=c.get("status"), passed=c.get("passed"),
                               min_jd_lower_bound=c.get("min_jd_lower_bound")))
    record = dict(arm="A0", label="A0", reference=False, run_dir=a0.get("run_dir"), run_id=a0.get("run_id"),
                  options=dict(method=a0.get("method"), schema=a0.get("schema")), status=a0.get("status"),
                  wording=None, classification=a0.get("classification"), native_commits=a0.get("native_commits"),
                  wall=dict(total_seconds=a0.get("completion_seconds"), e2e_seconds=a0.get("e2e_seconds")),
                  pair=a0.get("pair"), d0_evaluate=d0)
    found, reason = a0_reevaluation(eval_dir)
    if found is not None:
        mpath, rec = found
        rm = rec.get("metrics") or {}
        record.update(source="bayesreg.eval.cli", eval_metrics=mpath, eval_run_id=rec.get("run_id"),
                      field_role=rec.get("field_role"), run_status=rec.get("run_status"),
                      metrics={k: rm.get(k) for k, _ in MEASURES}, certificate=certificate_of(rec.get("certificate")),
                      source_note=(f"A0 由 bayesreg.eval.cli 重评（{mpath}），与各臂同一评测入口；"
                                   "experiments/d0/evaluate.py 的原值见 comparison.json 的 a0.d0_evaluate。"))
    else:
        record.update(source="experiments/d0/evaluate.py", eval_metrics=str(path), field_role=d0["field_role"],
                      metrics=d0["metrics"], certificate=d0["certificate"], fallback_reason=reason,
                      source_note=(f"A0 来自 experiments/d0/evaluate.py（{path}；bayesreg.eval.cli 重评不可用：{reason}），"
                                   "与各臂不是同一评测入口；指标定义相同（见各自 metrics.json 的 definitions），"
                                   "证书列是该运行自己的证书。"))
    return record


def fmt(value, nd=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{nd}g}" if abs(value) < 1e-3 and value != 0 else f"{value:.{nd}f}"
    return str(value)


def levels_text(levels):
    return "; ".join(f"L{r['level']}r{r['round']} {r['status']} {r['accepted']}步" for r in levels) or "—"


def row_pair(row):
    if row["arm"] == "A0":
        return row.get("pair")
    pair = (row.get("options") or {}).get("pair")
    return None if pair is None else str(pair).split("/")[-1]


def comparable(row, pair=None):
    """Next to A0: a comparable field role and (when both are known) A0's pair."""
    if row.get("field_role") not in COMPARABLE_ROLES:
        return False
    own = row_pair(row)
    return pair is None or own is None or own == pair


def role_text(row):
    role = row.get("field_role")
    if role is None:
        return "**未评测**" if row.get("arm") != "A0" else "—"
    if role == ROLE_HANDOVER:
        return f"**{HANDOVER_TEXT}**"
    return role if role in COMPARABLE_ROLES else f"**{role}**"


def witness_text(w):
    if w is None:
        return "未记录"
    s = w["sample_point_by_type"]
    text = (f"{w['sample_point']}（角 {s['corner']}·棱中 {s['edge_mid']}·面心 {s['face_center']}·体心 {s['body_center']}）"
            f" / {w['edge_interior']} / {w['face_interior']} / {w['cell_interior']}（无 {w['no_witness']}；"
            f"棱上 {w['on_edge']}）")
    if w.get("other_types"):
        text += f"；其他 {w['other_types']}"
    return text


def w2_text(w):
    if w is None:
        return "未记录"
    nonpos = w.get("min53_nonpositive_by_point_type") or {}
    parts = ", ".join(f"{k} {v}" for k, v in sorted(nonpos.items())) or "无"
    return f"W2 {fmt(w.get('w2_min53_positive'))}；53 点 J ≤ 0：{parts}"


def rejections_text(by_level):
    if not by_level:
        return "—"
    return "; ".join(f"L{level} {e['total']}（{e['jacobian_violation']}）"
                     for level, e in sorted(by_level.items(), key=lambda kv: (kv[0] is None, kv[0])))


def last_witness_text(w):
    """The witness types of a blocked round's last iteration (log.jsonl)."""
    if w is None:
        return "末次迭代无 JV 记录"
    if w["recorded"] == 0:
        return f"末次迭代 JV {w['jacobian_violations']}，见证点未记录"
    types = ", ".join(f"{k} {v}" for k, v in sorted(w["types"].items()))
    nonpos = ", ".join(f"{k} {v}" for k, v in sorted(w["min53_nonpositive_by_point_type"].items())) or "无"
    return f"末次迭代 JV {w['jacobian_violations']}：{types}；W2 {w['w2_min53_positive']}；53 点 J ≤ 0：{nonpos}"


def endings_text(row):
    blocked, other = row.get("blocked_levels") or [], row.get("other_abnormal_levels") or []
    text = "、".join(f"L{e['level']}（{e['status']}；{last_witness_text(e.get('last_iteration_witness'))}）"
                    for e in blocked) or "无"
    if other:
        text += "；另：" + "、".join(f"L{e['level']}（{e['status']}）" for e in other)
    return text


def options_text(o):
    text = f"{o.get('channel_source')}, {o.get('logvol_points')}"
    if o.get("logvol_points") == "gauss27+closed27":
        share = o.get("logvol_closed27_share")
        text += f" β={share:g}" if isinstance(share, (int, float)) else f" β={fmt(share)}"
        text += f" {o.get('closed27_weights')}"
    # a record without jd_cell_estimate predates JDC, where gauss27+corners was the only estimate (DEF: no longer
    # "the default", so the label says why)
    text += (f", JD 估计 {o.get('jd_cell_estimate') or 'gauss27+corners（记录无此项：JDC 之前的唯一取法）'}, "
             f"{o.get('backend')}/{o.get('device')}")
    return text


HEAD = ["臂", "选项", "状态", "field_role", "seg35", "seg4", "HD95", "SDlogJ 官方", "SDlogJ 脑内", "证书（评测）",
        "各层状态与接受步数", "受阻收尾的层", "各层被拒候选（JV）", "见证点 采样点/棱内/面内/体内", "W2 与 53 点 J ≤ 0",
        "CPU 回落", "墙钟 s"]


def table_line(r):
    o = r.get("options") or {}
    if r["arm"] == "A0":
        option, status = "配置 C（冻结 eb808be，90 s）", f"{r.get('status')}（{(r.get('classification') or {}).get('class')}）"
        levels, fallback = f"原生提交 {r.get('native_commits')}", "—"
        blocked = rejected = witness = w2 = "—"
    else:
        option = options_text(o)
        registry = (r.get("structure_registry") or {}).get("status")
        if registry:
            option += f", 结构层登记 {registry}"
        status = f"{r.get('status')}（{r.get('wording')}，exit {r.get('exit_code')}）"
        levels = levels_text(r.get("levels") or [])
        f = r.get("fallback") or {}
        fallback = (f"{f.get('cpu_fallback')}/{f.get('comparisons')}（{fmt(f.get('fallback_ratio'), 3)}）"
                    if f.get("comparisons") is not None else "—")
        blocked, rejected = endings_text(r), rejections_text(r.get("rejections_by_level"))
        witness, w2 = witness_text(r.get("witness")), w2_text(r.get("witness"))
    c = r.get("certificate") or {}
    cert = f"{c.get('status')} / {c.get('passed')} / {fmt(c.get('min_jd_lower_bound'), 3)}" if c else "—"
    m = r.get("metrics") or {}
    cells = [r.get("label") or r["arm"], option, status, role_text(r),
             *(fmt(m.get(k), 3 if k == "hd95" else 4) for k, _ in MEASURES), cert, levels, blocked, rejected, witness, w2,
             fallback, fmt((r.get("wall") or {}).get("total_seconds"), 1)]
    return "| "+" | ".join(str(x).replace("|", "/") for x in cells)+" |"


def table(rows):
    return ["| "+" | ".join(HEAD)+" |", "|"+"---|"*len(HEAD), *(table_line(r) for r in rows)]


def markdown(a0, arms, pair, refs=()):
    rows = [a0, *refs, *arms]
    near = [r for r in rows if r["comparable"]]
    apart = [r for r in rows if not r["comparable"]]
    lines = [f"# M2 GPU 冒烟对照（{pair}）", "", NOTE.format(pair=pair), "", a0["source_note"], ""]
    if refs:
        lines += [f"参照臂（标“(v1)”）读自 {refs[0]['root']}（只读）；代码版本以各运行目录的 git.txt 为准，可能与本批不同，"
                  "见证点列显示“未记录”的是代码早于见证记录（IMPL27）的运行。", ""]
    lines += [f"## 可比行（field_role ∈ {' / '.join(COMPARABLE_ROLES)}，与 A0 同一对）", ""]
    lines += table(near) if near else ["（无）"]
    if apart:
        lines += ["", "## 不可比行（不与 A0 并列比较）", "",
                  f"{HANDOVER_TEXT}、no_native_result、unknown、未评测与配对不同的行只单列，不进上表。", ""]
        lines += table(apart)
    if not a0["comparable"]:
        lines += ["", f"**警告：A0 的 field_role 是 {a0.get('field_role')}，不是原生正式状态；上表没有参照行。**"]
    cert_note = ("各臂与 A0 均为 bayesreg.eval 对导出场的全证书" if a0.get("source") == "bayesreg.eval.cli"
                 else "各臂为 bayesreg.eval 对导出场的全证书；A0 为该运行自己的证书（experiments/d0/evaluate.py）")
    lines += ["", f"证书列：状态 / 是否通过 / 最小 J 下界（{cert_note}）。", "", REJECT_NOTE, "", WITNESS_NOTE, "",
              "缺失的文件：" + ("; ".join(f"{r['label']}: {', '.join(r['missing'])}" for r in [*refs, *arms]
                                    if r.get("missing")) or "无"), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", default="G1,G2,G3")
    parser.add_argument("--a0", required=True, help="D0 A0 metrics.json of the same pair (experiments/d0/evaluate.py)")
    parser.add_argument("--a0-eval", help="the bayesreg.eval.cli re-evaluation of A0 (default <root>/eval_A0)")
    parser.add_argument("--ref-root", help="an earlier smoke root (read only) whose --ref-arms are listed as (v1) rows")
    parser.add_argument("--ref-arms", default="", help="arms of --ref-root, e.g. G2,G3")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)
    for path in (args.out_json, args.out_md):
        if Path(path).exists():
            raise SystemExit(f"{path} exists; nothing is overwritten")
    ref_arms = [a for a in args.ref_arms.split(",") if a]
    if ref_arms and not args.ref_root:
        raise SystemExit("--ref-arms needs --ref-root")
    a0 = a0_record(args.a0, Path(args.root)/"eval_A0" if args.a0_eval is None else args.a0_eval)
    pair = a0.get("pair")
    refs = [arm_record(args.ref_root, arm, label=f"{arm} (v1)", reference=True) for arm in ref_arms]
    arms = [arm_record(args.root, arm) for arm in args.arms.split(",")]
    for row in [a0, *refs, *arms]:
        row["comparable"] = comparable(row, pair)
    arm_pairs = sorted({row_pair(r) for r in [*refs, *arms] if row_pair(r)})
    mismatch = [p for p in arm_pairs if p != pair]
    document = dict(schema=SCHEMA, pair=pair, note=NOTE.format(pair=pair), a0_source=a0["source"],
                    a0_source_note=a0["source_note"], comparable_roles=list(COMPARABLE_ROLES),
                    measures=[k for k, _ in MEASURES], blocked_statuses=list(BLOCKED_STATUSES),
                    witness_note=WITNESS_NOTE, rejection_note=REJECT_NOTE, pair_mismatch=mismatch,
                    ref_root=args.ref_root, a0=a0, reference_arms=refs, arms=arms)
    with open(args.out_json, "x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    text = markdown(a0, arms, pair, refs)
    if mismatch:
        text += f"\n**警告：臂的配对 {mismatch} 与 A0 的配对 {pair} 不同，这些行不可比（只可能是 dry run 或换对后的参照）。**\n"
    with open(args.out_md, "x", encoding="utf-8") as stream:
        stream.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
