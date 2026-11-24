"""Per-level readouts of bayesreg.cli.run_pair run directories (work package M2V1; CVG result, suggestion 1). Read only.

    python -B experiments/m2v1/level_table.py RUN_DIR... [--sweep-root DIR] [--out-json F] [--out-md F]

For every run directory (any run_pair output: the M2V1 batch, the eta sweep's runs/<stage>/<label>/<pair>/, the GPU smoke
arms) and every (level, EM round) of it, from hyper.json rounds (or, when the run ended without hyper.json -- killed --,
from the "round" rows of log.jsonl, marked as such):
  status of the round and its ending type (达阈 = converged_gradient, i.e. GF_B <= T_B of that round; 上限收尾 =
  cap_reached / max_iterations; 预算收尾 = budget_exhausted; 受阻收尾 = a status whose run wording is 受阻收尾 in
  bayesreg.solve.stopping.RUN_WORDING; anything else is shown as its status), accepted steps, commits, GF_B, T_B and
  GF_B/T_B at the end of the round, the rejected line-search candidates by reason (counters.rejections), the number of
  JD-scaling refreshes of S~ (metric.jd_scaling.jd_scaling_refresh; the round-start computation is not a refresh), min
  J_a and the largest fraction of interior nodes with r_a < 1 over the round's S~ computations (metric.jd_scaling), the
  certified minimum J lower bound of the round's last accepted state (min_jd_lower_bound), and the seconds.
A level that had started but has no round record (the run was killed inside it) is listed from log.jsonl as
"unfinished": the accepted rows after the last round record, the last accepted state's certified lower bound and the
JD-scaling rows of that level.
Per run also: run status, wording and exit code (status.json), the cell estimate and the metric (config.json args), eta
(config.json eta), the wall budget, and whether the native level was reached.
--sweep-root DIR adds every DIR/runs/<stage>/<label>/<pair>/ (the eta sweep layout; read only). Nothing is written
into a run directory: --out-json / --out-md must be new files outside every listed run directory. Without them the
Markdown goes to stdout.
"""
import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.solve.stopping import (BUDGET_EXHAUSTED, CAP_REACHED, CONVERGED_GRADIENT, MAX_ITERATIONS,  # noqa: E402
                                     RUN_WORDING)

SCHEMA = "bayesreg.m2v1.level_table.v1"
BLOCKED_STATUSES = tuple(sorted(s for s, w in RUN_WORDING.items() if w == "受阻收尾"))
ENDING = {CONVERGED_GRADIENT: "达阈", CAP_REACHED: "上限收尾", MAX_ITERATIONS: "上限收尾", BUDGET_EXHAUSTED: "预算收尾"}
NOTE = ("每层（每个 EM 轮）一行，取自 hyper.json 的 rounds（运行被杀、没有 hyper.json 时取 log.jsonl 的 round 行，并注明）。"
        "达阈 = 该轮 GF_B ≤ T_B（round 状态 converged_gradient）；粗层达阈只是转层，不是“B 线收敛”。"
        "S̃ 刷新 = jd_scaling_refresh（不含轮首的计算）；min J_a 与 r_a < 1 比例取该轮全部 S̃ 计算的最小 / 最大值；"
        "证书下界 = 该轮最后一个接受态的认证最小 J 下界。被拒候选只含 PG 线搜索候选。")


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ending_of(status):
    if status in ENDING:
        return ENDING[status]
    if status in BLOCKED_STATUSES:
        return "受阻收尾"
    return str(status)


def round_record(r, source):
    """One (level, round) row from a hyper.json round (or a log.jsonl round row: the same fields)."""
    js = ((r.get("metric") or {}).get("jd_scaling") or {})
    computations = js.get("computations") or []
    min_ja = js.get("min_J_a")
    if min_ja is None and computations:
        min_ja = min(c.get("min_J_a") for c in computations if finite(c.get("min_J_a")))
    max_frac = js.get("max_fraction_r_lt_1")
    if max_frac is None and computations:
        max_frac = max(c.get("fraction_r_lt_1") for c in computations if finite(c.get("fraction_r_lt_1")))
    rejections = dict(((r.get("counters") or {}).get("rejections") or {}))
    status = r.get("status")
    gf, tb = r.get("GF_B"), r.get("T_B")
    return dict(level=r.get("level"), round=r.get("round"), source=source, status=status, ending=ending_of(status),
                reached_threshold=status == CONVERGED_GRADIENT, accepted=r.get("accepted"), commits=r.get("commits"),
                cap=r.get("cap"), GF_B=gf, T_B=tb,
                GF_B_over_T_B=r.get("GF_B_over_T_B") if r.get("GF_B_over_T_B") is not None else
                (gf/tb if finite(gf) and finite(tb) and tb > 0 else None),
                rejections=rejections, rejected=sum(rejections.values()),
                candidates=(r.get("counters") or {}).get("candidates"),
                jd_scaling_mode=js.get("mode"), cell_estimate=js.get("cell_estimate"),
                jd_scaling_refresh=js.get("jd_scaling_refresh"), jd_scaling_computations=len(computations) or None,
                min_J_a=min_ja, max_fraction_r_lt_1=max_frac, min_jd_lower_bound=r.get("min_jd_lower_bound"),
                seconds=r.get("seconds"))


def log_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                          # a line cut by a kill
            if isinstance(row, dict) and row.get("kind") in ("round", "accept", "jd_scaling"):
                rows.append(row)
    return rows


def unfinished_level(rows, finished):
    """The level that a killed run was inside: accepted rows after the last round row (None when every started level
    has a round record)."""
    started = [r for r in rows if r.get("kind") in ("accept", "jd_scaling")]
    if not started:
        return None
    level, rnd = started[-1].get("level"), started[-1].get("round")
    if (level, rnd) in finished:
        return None
    accepts = [r for r in rows if r.get("kind") == "accept" and (r.get("level"), r.get("round")) == (level, rnd)]
    scaling = [r for r in rows if r.get("kind") == "jd_scaling" and (r.get("level"), r.get("round")) == (level, rnd)]
    last = accepts[-1] if accepts else {}
    ja = [r.get("min_J_a") for r in scaling if finite(r.get("min_J_a"))]
    frac = [r.get("fraction_r_lt_1") for r in scaling if finite(r.get("fraction_r_lt_1"))]
    return dict(level=level, round=rnd, source="log.jsonl (unfinished: no round record)", status="unfinished",
                ending="未结束（运行中断）", reached_threshold=False, accepted=len(accepts), commits=None, cap=None,
                GF_B=None, T_B=None, GF_B_over_T_B=None, rejections={}, rejected=None, candidates=None,
                jd_scaling_mode=None, cell_estimate=(scaling[-1].get("cell_estimate") if scaling else None),
                jd_scaling_refresh=max(len(scaling)-1, 0) if scaling else None,
                jd_scaling_computations=len(scaling) or None, min_J_a=min(ja) if ja else None,
                max_fraction_r_lt_1=max(frac) if frac else None, min_jd_lower_bound=last.get("min_jd_lower_bound"),
                seconds=None)


def run_record(run_dir, label=None):
    """Every readout of one run directory (module docstring)."""
    run_dir = Path(run_dir)
    hyper, status, config = (read_json(run_dir/"hyper.json"), read_json(run_dir/"status.json"),
                             read_json(run_dir/"config.json"))
    rows = log_rows(run_dir/"log.jsonl")
    if hyper is not None and hyper.get("rounds") is not None:
        levels = [round_record(r, "hyper.json") for r in hyper["rounds"]]
    else:
        levels = [round_record(r, "log.jsonl (no hyper.json)") for r in rows if r.get("kind") == "round"]
    extra = unfinished_level(rows, {(r["level"], r["round"]) for r in levels})
    if extra is not None:
        levels.append(extra)
    args = (config or {}).get("args") or {}
    eta = (config or {}).get("eta") or {}
    blocked = [dict(level=r["level"], round=r["round"], status=r["status"]) for r in levels
               if r["status"] in BLOCKED_STATUSES]
    return dict(run_dir=str(run_dir), label=label or run_dir.name, pair=args.get("pair"),
                run_status=(status or {}).get("status"), wording=(status or {}).get("wording"),
                exit_code=(status or {}).get("exit_code"), native=(status or {}).get("native"),
                status_json=status is not None, hyper_json=hyper is not None,
                jd_cell_estimate=args.get("jd_cell_estimate"), metric_jd_scaling=args.get("metric_jd_scaling"),
                channel_source=args.get("channel_source"), logvol_points=args.get("logvol_points"),
                eta=dict(alpha=eta.get("alpha"), nu=eta.get("nu"), tau=eta.get("tau")),
                wall_seconds=((status or {}).get("wall_budget") or {}).get("wall_seconds", args.get("wall_seconds")),
                levels=levels, blocked=blocked, n_blocked=len(blocked))


def fmt(value, nd=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-2:
            return f"{value:.{nd}g}"
        if abs(value) >= 1e4:
            return f"{value:.{nd}e}"
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.{nd}g}" if abs(value) >= 1 else f"{value:.{nd}f}"
    return str(value)


def rejections_text(r):
    if r.get("rejected") is None:
        return "—"
    if not r["rejections"]:
        return "0"
    return f"{r['rejected']}（" + "，".join(f"{k} {v}" for k, v in sorted(r["rejections"].items())) + "）"


HEAD = ["运行", "层", "轮", "收尾", "接受步", "GF_B/T_B（末）", "被拒候选", "S̃ 刷新", "min J_a", "r_a<1 比例（最大）",
        "证书下界", "秒", "来源"]


def markdown(records, title="各层读数"):
    lines = [f"## {title}", "", NOTE, "", "| "+" | ".join(HEAD)+" |", "|"+"---|"*len(HEAD)]
    for rec in records:
        if not rec["levels"]:
            lines.append(f"| {rec['label']} | — | — | 无层记录（{rec['run_status']}） |" + " — |"*(len(HEAD)-4))
        for r in rec["levels"]:
            cells = [rec["label"], r["level"], r["round"], r["ending"]+("" if r["ending"] == r["status"] else
                                                                       f"（{r['status']}）"),
                     r["accepted"], fmt(r["GF_B_over_T_B"]), rejections_text(r), fmt(r["jd_scaling_refresh"]),
                     fmt(r["min_J_a"]), fmt(r["max_fraction_r_lt_1"]), fmt(r["min_jd_lower_bound"]), fmt(r["seconds"]),
                     r["source"]]
            lines.append("| "+" | ".join(str(c).replace("|", "/") for c in cells)+" |")
    lines += ["", "| 运行 | 对 | 状态 | 措辞 | exit | 到原生层 | 单元估计 | ν | 预算 s | 受阻收尾的层 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for rec in records:
        nu = rec["eta"].get("nu")
        blocked = ", ".join("L{} {}".format(b["level"], b["status"]) for b in rec["blocked"]) or "无"
        lines.append(f"| {rec['label']} | {rec['pair']} | {rec['run_status']} | {rec['wording']} | {rec['exit_code']} | "
                     f"{fmt(rec['native'])} | {rec['jd_cell_estimate']} | {fmt(nu, 6) if nu is not None else '—'} | "
                     f"{fmt(rec['wall_seconds'])} | {blocked} |")
    return "\n".join(lines)+"\n"


def sweep_runs(root):
    """runs/<stage>/<label>/<pair>/ of an eta sweep root, sorted; labelled <stage>-<label>-<pair>."""
    out = []
    for directory in sorted(Path(root).glob("runs/*/*/*")):
        if directory.is_dir():
            out.append((directory, "-".join(directory.parts[-3:])))
    return out


def write_new(path, text, run_dirs):
    path = Path(path).resolve()
    for run in run_dirs:
        run = Path(run).resolve()
        if path == run or run in path.parents:
            raise SystemExit(f"{path}: inside the run directory {run} (read only)")
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/m2v1/level_table.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args(argv)
    targets = [(Path(r), None) for r in args.runs]
    if args.sweep_root is not None:
        targets += sweep_runs(args.sweep_root)
    if not targets:
        parser.error("no run directory (give RUN_DIR... or --sweep-root)")
    missing = [str(t) for t, _ in targets if not t.is_dir()]
    if missing:
        parser.error(f"not a directory: {missing}")
    records = [run_record(t, label) for t, label in targets]
    text = markdown(records)
    dirs = [t for t, _ in targets]
    if args.out_json:
        write_new(args.out_json, json.dumps(dict(schema=SCHEMA, note=NOTE, runs=records), indent=1, ensure_ascii=False)
                  + "\n", dirs)
    if args.out_md:
        write_new(args.out_md, text, dirs)
    if not (args.out_json or args.out_md):
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
