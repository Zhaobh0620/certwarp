"""Summary of the path pilot (experiments/pilot/run_pilot.py; runs/20261031-path-pilot/plan.md). Development set,
single observations, exploratory: every reading is descriptive; a verdict string is a reading for the lead, never a
method conclusion.

    python -B experiments/pilot/summarize.py --root DIR [--no-j53]

Reads $ROOT/batch.json, $ROOT/p0.json, the pilot's runs/<arm>/<short>/ and eval/<arm>/<short>/, and (read only) the
M2V1 root named in batch.json (its batch.json must still have the recorded sha256; P0 = its dev runs of the pilot's
pairs). Rows come from experiments/m2v1/summarize.py pair_row (status, evaluator metrics and certificate, config.json
args and eta checks, experiments/m2v1/level_table.py run_record), plus the path readouts below. Nothing is written into
a run directory or into the M2V1 root.

Per run: seg35, seg4, HD95, official and brain SDlogJ, evaluator certificate (status, min J lower bound), run status;
per level (level_table): ending, accepted steps, cap, GF_B/T_B, rejections by reason, S~ refreshes, min J_a, largest
r_a < 1 fraction, certified lower bound of the last accepted state, seconds; derived: native accepted steps, native
seconds, native final GF_B/T_B and ending, coarse levels that reached the threshold, blocked levels (受阻收尾),
JACOBIAN_VIOLATION rejections; certificate_bernstein readouts from log.jsonl jd_scaling rows (n_bernstein_j53_violations
summed, n_cells_raised / n_cells_bernstein maxima) and any bernstein_j53_violations_total in status.json, hyper.json,
timing.json (the run-level total of C09.1-3a / D53; absent -> null, "not found"); the count the reading uses is the
larger of the two that are recorded (null when neither is).
53-point minimum J of the final state (contract C09.1-3a N3 record; the lead 10-31 ~6:15 pm CDT): for every run with a
final_native.npz (P0's in the M2V1 root included; read only, CPU), u is read by bayesreg.eval.fields.read_final_native
(state_hash checked), the grid rebuilt from the file's own grid record, and bayesreg.solve.jd_scaling
.closed27_cell_estimate gives per cell the minimum of J over Gauss27 and closed27 (53 points) and over Gauss27 and the
corners; recorded: both minima, the cell of the 53-point minimum, and whether it is >= the evaluator's certified lower
bound (it must be: a sampled minimum is >= the true minimum >= a certified lower bound). The file must carry a
state_hash equal to sha256(u), the rebuilt grid's grid_id must equal the file's, and u's sha256 must equal status.json
"exported" and the evaluator certificate's state_hash where those are recorded; otherwise "待核对（…）" (PILOT review
minor 3). --no-j53 (or a failure: memory, a non-finite value, an unreadable file) records "审核时另算" instead; nothing
else depends on it.
Caps: the accepted-step cap of every (level, round) from hyper.json cap_accepted (SCH runs), else config.json args
max_accepted (PILOT review minor 4).
certificate_bernstein audit (C09.1-3a check (b), D53; PILOT review I1): every run with a violation count > 0 is listed
in counts.bernstein_violation_runs and at the top of summary.md, whatever its arm's label; its numbers enter no
conclusion until the lead's audit (reported to the lead on its own; final state recertified without the prescreen).
Baseline (the lead, 10-31 ~6:15 pm CDT, before any pilot result): P0n -- the current path on the pilot's code in the
same period -- when the batch ran it, else P0. Paired differences arm - baseline (pairs where both are comparable:
evaluated, field_role native_formal / disp_only, evaluator certificate passed; path readouts: pairs where both have
native-level records): n, median, #(d > 0), #(d < 0), per pair; native accepted steps also as the ratio arm /
baseline. Secondary reference columns: every arm (P0n included) - P0, the reused M2V1 run (other code, other period).
Also P3 - P1 and P3 - P2, and P4 - P3: "the effect of the sigma estimation domain" (support_shared against brain).
P4 (revision 3, BZ 10-31 ~9:55 pm CDT): exploratory, described only -- its row in the readings carries the checks (S1,
S2, S3) but never a candidate label, and the path suggestion ignores it; the preregistered direction hypothesis (if the
sigma weights are the main cause of the seg4 gap, the median P4 - P3 seg4 difference is clearly positive, >= +.02, and
seg35 does not clearly drop, median >= -.005) is read descriptively. Support records (hyper.json support_levels, non-brain
runs only): per level N_c/N_q, the empty-support channels, N_S, sigma_S^2 at the level start; per round sigma2_S_used /
sigma2_S_post.
Preregistered readings (plan.md "判定"; thresholds below, confirmed by the lead 10-31): per arm other than the baseline,
against the baseline, a verdict 倾向列入候选 / 不列入候选 / 待审核 / 待核对 /
无法区分 / 只描述, with the reasons; then the path suggested for the eta re-sweep (the lead decides).
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = _load("_pilot_m2v1_summarize", "experiments/m2v1/summarize.py")
lt = ms.levels                                          # experiments/m2v1/level_table.py (imported, not modified)

SCHEMA = "bayesreg.path_pilot.summary.v1"
MEASURES = ms.MEASURES                                  # seg35, seg4, HD95, SDlogJ official, SDlogJ brain
PATH_MEASURES = (("native_accepted", "原生接受步"), ("native_seconds", "原生层秒"),
                 ("native_gf_over_tb", "原生末 GF_B/T_B"), ("coarse_reached", "粗层达阈数"),
                 ("n_blocked", "受阻收尾层数"), ("jv_rejections", "JACOBIAN_VIOLATION 拒绝"))
BERNSTEIN = "certificate_bernstein"
NATIVE = 3
# preregistered thresholds (plan.md "判定"; written before any pilot run; confirmed by the lead 10-31 ~6:15 pm CDT)
MIN_N = 4                     # comparable pairs (of 5) needed for a verdict; fewer: describe only
DICE_NOT_DOWN = -.005         # median paired d of seg35 and of seg4 >= this: "Dice 不降"
DICE_DOWN = -.01              # median paired d of seg35 or of seg4 <= this: "Dice 降低"
STEPS_CAPS = 2.               # P1 / P3: median native-steps ratio arm / baseline >= 2 ("原生步数至少翻倍")
STEPS_CELL = 1.5              # P2: median native-steps ratio >= 1.5, or
GF_CELL = .5                  # P2: median native final GF_B/T_B ratio arm / baseline <= .5
MAJORITY = .8                 # and in the same direction on >= ceil(.8 n) pairs
P4_SEG4_DIRECTION = .02       # P4 - P3 direction hypothesis (revision 3; descriptive, not a conclusion)
NOTE = ("开发集（stage_L2R_val_v1 前 5 对），单次观测，探索性。每次运行以开发开关 --wall-seconds 的“预算收尾”或其他 C13-3 "
        "收尾结束，不是“B 线收敛”，除非状态写明。主要对照是 P0n（现行路径，本试验代码、同一时段）；P0 是 M2V1 开发批"
        "同一对的运行（代码与时段都不同），复用、不重跑，只作次要参照。墙钟预算的截断点依赖机器负载。读数标签不是结论。")


class SummaryError(Exception):
    """The summary cannot be written as specified."""


read_json, finite, short_name, utc = ms.read_json, ms.finite, ms.short_name, ms.utc


def find_key(doc, key):
    """Every value of `key` anywhere in a JSON document."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == key:
                out.append(v)
            out += find_key(v, key)
    elif isinstance(doc, list):
        for v in doc:
            out += find_key(v, key)
    return out


def bernstein_readouts(run):
    """certificate_bernstein records of one run directory (None values when absent)."""
    run = Path(run)
    rows = [r for r in lt.log_rows(run/"log.jsonl") if r.get("kind") == "jd_scaling"]
    mine = [r for r in rows if r.get("cell_estimate") == BERNSTEIN]
    per_level = {}
    for r in mine:
        entry = per_level.setdefault(str(r.get("level")), dict(computations=0, violations=0, max_cells_raised=None,
                                                               max_cells_bernstein=None))
        entry["computations"] += 1
        v = r.get("n_bernstein_j53_violations")
        entry["violations"] += v if isinstance(v, int) else 0
        for key, field in (("max_cells_raised", "n_cells_raised"), ("max_cells_bernstein", "n_cells_bernstein")):
            value = r.get(field)
            if isinstance(value, int):
                entry[key] = value if entry[key] is None else max(entry[key], value)
    totals = []
    for name in ("status.json", "hyper.json", "timing.json"):
        totals += [v for v in find_key(read_json(run/name), "bernstein_j53_violations_total") if isinstance(v, int)]
    rows_field = bool(mine) and all(isinstance(r.get("n_bernstein_j53_violations"), int) for r in mine)
    in_rows = sum(e["violations"] for e in per_level.values()) if mine else None
    # the reading's count (S3): the larger of the run-level total and the sum over the S~ rows, whichever are recorded
    known = ([max(totals)] if totals else [])+([in_rows] if rows_field else [])
    return dict(computations=len(mine), per_level=per_level, violations_in_rows=in_rows, rows_field=rows_field,
                violations_total=max(totals) if totals else None, total_found=bool(totals),
                violations=max(known) if known else None)


def j53_final(run_dir, compute=True):
    """The 53-point minimum J of the run's final_native.npz (module docstring); read only."""
    path = Path(run_dir)/"final_native.npz"
    if not path.is_file():
        return dict(state="无 final_native.npz", min_j53=None)
    if not compute:
        return dict(state="审核时另算（--no-j53）", min_j53=None)
    try:
        import numpy as np
        from jd_curl.three_d.grid import make_grid_3d
        from bayesreg.eval.fields import read_final_native
        from bayesreg.solve.jd_scaling import closed27_cell_estimate
        u, record = read_final_native(path)
        if record.get("state_hash_matches") is not True:
            return dict(state="待核对（final_native.npz 没有 state_hash，不能确认是哪个状态）", min_j53=None,
                        u_sha256=record.get("u_sha256"))
        g = record["grid"]
        (nz, ny, nx), (hx, hy, hz), (x0, y0, z0) = g["shape_zyx"], g["spacing_xyz"], g["origin_xyz"]
        grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit=g["unit"])
        if g.get("grid_id") is not None and grid.grid_id != g["grid_id"]:
            return dict(state=f"待核对（重建网格的 grid_id {grid.grid_id} 与文件的 {g['grid_id']} 不同）", min_j53=None,
                        u_sha256=record["u_sha256"])
        j53, default = closed27_cell_estimate(u, grid)
        cell = [int(i) for i in np.unravel_index(int(np.argmin(j53)), j53.shape)]
        out = dict(state="computed", min_j53=float(j53.min()), min_gauss27_corners=float(default.min()),
                   argmin_cell_zyx=cell, u_sha256=record["u_sha256"], file_status=record.get("status"),
                   grid_id=grid.grid_id,
                   points="Gauss27 ∪ closed27（53 点；bayesreg.solve.jd_scaling.closed27_cell_estimate）")
        del u, j53, default
        return out
    except Exception as exc:                              # optional step: any failure is recorded, never fatal
        return dict(state=f"审核时另算（{type(exc).__name__}: {str(exc)[:200]}）", min_j53=None)


def derive(row, j53=True):
    """Path readouts of one pair_row (module docstring)."""
    rec = row.get("levels")
    lv = (rec or {}).get("levels") or []
    native = [r for r in lv if r["level"] == NATIVE]
    coarse = [r for r in lv if r["level"] != NATIVE]
    accepted = [r["accepted"] for r in native]
    seconds = [r["seconds"] for r in native]
    row.update(
        native_reached=bool(native),
        native_accepted=sum(accepted) if native and all(isinstance(a, int) for a in accepted) else None,
        native_seconds=sum(seconds) if native and all(finite(s) for s in seconds) else None,
        native_gf_over_tb=native[-1]["GF_B_over_T_B"] if native else None,
        native_ending=native[-1]["ending"] if native else None,
        native_lower_bound=native[-1]["min_jd_lower_bound"] if native else None,
        coarse_reached=sum(bool(r["reached_threshold"]) for r in coarse) if lv else None,
        coarse=[dict(level=r["level"], round=r["round"], ending=r["ending"], accepted=r["accepted"], cap=r.get("cap"),
                     GF_B_over_T_B=r["GF_B_over_T_B"], seconds=r["seconds"],
                     jv=(r.get("rejections") or {}).get("JACOBIAN_VIOLATION", 0)) for r in coarse],
        n_blocked=(rec or {}).get("n_blocked") if rec else None,
        jv_rejections=sum((r.get("rejections") or {}).get("JACOBIAN_VIOLATION", 0) for r in lv) if lv else None,
        bernstein=bernstein_readouts(row["run_dir"]) if Path(row["run_dir"]).is_dir() else None)
    row["caps"] = caps_of(row["run_dir"], lv)
    row["support"] = support_readouts(row["run_dir"])
    final = j53_final(row["run_dir"], j53) if Path(row["run_dir"]).is_dir() else dict(state="无运行目录", min_j53=None)
    if final["state"] == "computed":
        evaluated = ((read_json(row["eval_metrics"]) or {}).get("certificate") or {}).get("state_hash") \
            if row.get("eval_metrics") else None
        exported = row.get("status_exported_hash")
        final.update(exported_matches=None if exported is None else exported == final["u_sha256"],
                     evaluated_matches=None if evaluated is None else evaluated == final["u_sha256"])
        if final["exported_matches"] is False or final["evaluated_matches"] is False:
            final["state"] = "待核对（final_native 的 u 与 status.json exported 或评测证书的 state_hash 不同）"
    bound = (row.get("certificate") or {}).get("min_jd_lower_bound")
    final["ge_certified_bound"] = (final["min_j53"] >= bound if final["state"] == "computed"
                                   and finite(final.get("min_j53")) and finite(bound) else None)
    row["j53_final"] = final
    return row


def support_readouts(run_dir):
    """hyper.json support_levels of a non-brain run (C04-6 记录; None under brain or without hyper.json): per level
    N_c/N_q, the empty-support channels, N_S and sigma_S^2 at the level start, with the rounds' sigma2_S_used / _post."""
    hyper = read_json(Path(run_dir)/"hyper.json")
    found = find_key(hyper, "support_levels") if hyper is not None else []
    levels = next((x for x in found if isinstance(x, list)), None)
    if not levels:
        return None
    rounds = hyper.get("rounds") or []
    out = []
    for rec in levels:
        mine = [r for r in rounds if r.get("level") == rec.get("level")]
        out.append(dict(level=rec.get("level"), N_q=rec.get("N_q"), N_c_over_N_q=rec.get("N_c_over_N_q"),
                        n_empty=len(rec.get("empty_support_channels") or []),
                        empty_support_channels=rec.get("empty_support_channels"), N_S=rec.get("N_S"),
                        sigma2_S_start=rec.get("sigma2_S_start"),
                        sigma2_S_used=[r.get("sigma2_S_used") for r in mine],
                        sigma2_S_post=[r.get("sigma2_S_post") for r in mine]))
    return out


def p4_direction(comp):
    """The preregistered direction hypothesis on P4 - P3 (revision 3), read descriptively."""
    n, d4, d35 = comp["seg4"]["n"], comp["seg4"]["median"], comp["seg35"]["median"]
    if n < MIN_N:
        reading = f"可比对数 {n} < {MIN_N}：只描述"
    elif d4 is not None and d35 is not None and d4 >= P4_SEG4_DIRECTION and d35 >= DICE_NOT_DOWN:
        reading = "与方向假设一致（描述，不是结论）"
    else:
        reading = "与方向假设不一致（描述，不是结论）"
    return dict(hypothesis=("若 σ 权重是 seg4 落后的主因，P4 − P3 的 seg4 中位差应明显为正（≥ +.02），seg35 不明显下降"
                            "（中位差 ≥ −.005）"), n=n, median_seg4=d4, median_seg35=d35, reading=reading)


def caps_of(run_dir, levels):
    """The accepted-step cap of each (level, round) in level order: hyper.json round cap_accepted (SCH runs), else
    config.json args max_accepted; None where neither is recorded (PILOT review minor 4)."""
    rounds = (read_json(Path(run_dir)/"hyper.json") or {}).get("rounds") or []
    by_key = {(r.get("level"), r.get("round")): r.get("cap_accepted") for r in rounds}
    default = ((read_json(Path(run_dir)/"config.json") or {}).get("args") or {}).get("max_accepted")
    out = []
    for r in levels:
        value = by_key.get((r["level"], r["round"]))
        out.append(value if isinstance(value, int) else default if isinstance(default, int) else None)
    return out


def stats_of(per_pair, ratio=False):
    d = [x["d"] for x in per_pair]
    out = dict(n=len(d), median=statistics.median(d) if d else None, positive=sum(x > 0 for x in d),
               negative=sum(x < 0 for x in d), zero=sum(x == 0 for x in d), per_pair=per_pair)
    if ratio:
        r = [x["ratio"] for x in per_pair if finite(x.get("ratio"))]
        out["median_ratio"] = statistics.median(r) if r else None
    return out


def paired(rows_x, rows_y, key, *, dice):
    """d = x - y per pair (x, y two rows lists in the pilot's pair order)."""
    per_pair = []
    for x, y in zip(rows_x, rows_y):
        if dice:
            a, b = x["metrics"].get(key), y["metrics"].get(key)
            ok = x["comparable"] and y["comparable"]
        else:
            a, b = x.get(key), y.get(key)
            ok = x.get("native_reached") and y.get("native_reached")
        if ok and finite(a) and finite(b):
            entry = dict(pair=x["pair"], x=a, y=b, d=a-b)
            if not dice and b > 0:
                entry["ratio"] = a/b
            per_pair.append(entry)
    return stats_of(per_pair, ratio=not dice)


def compare(rows_x, rows_y):
    out = {m: paired(rows_x, rows_y, m, dice=True) for m, _ in MEASURES}
    out.update({m: paired(rows_x, rows_y, m, dice=False) for m, _ in PATH_MEASURES})
    return out


def need(n):
    return math.ceil(MAJORITY*n)


def arm_flags(aid, rows_x, rows_0, comp, cell, caps):
    """Preregistered reading of one arm against the baseline rows rows_0 (P0n, else P0; plan.md "判定"; cell its
    jd_cell_estimate, caps its per-level caps or None); returns dict(verdict, reasons, checks)."""
    checks, reasons = {}, []
    ran = [i for i, r in enumerate(rows_x) if r["state"] != "not_run"]
    bad = [i for i in ran if not rows_x[i]["comparable"]]
    checks["S1_all_runs_certified_export"] = not bad
    if bad:                       # PILOT review minor 2: no export because of the budget vs a safety failure
        budget = [rows_x[i]["short"] for i in bad if not rows_x[i].get("exported")
                  or rows_x[i].get("field_role") == "handover_only"]
        other = [rows_x[i]["short"] for i in bad if rows_x[i]["short"] not in budget]
        if budget:
            reasons.append(f"S1 不满足：{budget} 没有可比的导出场（无导出或原生层 0 步：预算或运行失败，不是证书不通过）")
        if other:
            reasons.append(f"S1 不满足：{other} 导出场不可比（证书未通过或评测失败）")
    both = [i for i in ran if rows_0[i]["state"] == "ran" and rows_x[i]["n_blocked"] is not None
            and rows_0[i]["n_blocked"] is not None]
    # S2 = contract N3's "blocked levels not more than certificate's", pair by pair (PILOT review minor 1); the sums only
    # describe
    worse = [rows_x[i]["short"] for i in both if rows_x[i]["n_blocked"] > rows_0[i]["n_blocked"]]
    bx, b0 = sum(rows_x[i]["n_blocked"] for i in both), sum(rows_0[i]["n_blocked"] for i in both)
    checks["S2_blocked_not_more"] = not worse
    checks["blocked"] = dict(pairs_more_than_baseline=worse, arm_sum=bx, baseline_sum=b0,
                             pairs=[rows_x[i]["short"] for i in both])
    if worse:
        reasons.append(f"S2 不满足：{worse} 上受阻收尾层数多于基线（合计 {bx} 对 {b0}，合计只作描述）")
    s3 = None
    if cell == BERNSTEIN:
        totals = [(rows_x[i]["bernstein"] or {}).get("violations") for i in ran]
        # PILOT review I1 (a): one violation count > 0 makes S3 False whatever is missing elsewhere; then missing
        if any(isinstance(t, int) and t > 0 for t in totals):
            s3 = False
        elif not totals or any(t is None for t in totals):
            s3 = None
        else:
            s3 = True
        checks["S3_bernstein_violations_zero"] = s3
        if s3 is False:
            reasons.append(f"S3：核对 (b) 违反累计 {totals}（D53：单独报总负责人，用不含预筛的证书配置重新认证终态；"
                           "审核完成前其数字不进结论）")
        elif s3 is None:
            reasons.append(f"S3：违反计数未找到（运行级 bernstein_j53_violations_total 与 S̃ 行的 "
                           f"n_bernstein_j53_violations 都没有；{totals}），待核对")
        # contract C09.1-3a N3 / D53: blocked levels pair by pair not more than certificate's and every final state
        # certified (S1 and S2), whatever the benefit B; under the audit rule (PILOT review I1 (b))
        checks["N3_criterion"] = ("待审核" if s3 is False else "待核对" if s3 is None
                                  else checks["S1_all_runs_certified_export"] and checks["S2_blocked_not_more"])
    n_dice, n_path = comp["seg35"]["n"], comp["native_accepted"]["n"]
    d35, d4 = comp["seg35"]["median"], comp["seg4"]["median"]
    dice_ok = n_dice > 0 and d35 is not None and d4 is not None and d35 >= DICE_NOT_DOWN and d4 >= DICE_NOT_DOWN
    dice_bad = n_dice > 0 and ((d35 is not None and d35 <= DICE_DOWN) or (d4 is not None and d4 <= DICE_DOWN))
    checks.update(D_dice_not_down=dice_ok, D_dice_down=dice_bad, n_dice=n_dice, n_path=n_path)
    steps = comp["native_accepted"]
    gf = comp["native_gf_over_tb"]
    if cell == BERNSTEIN and caps is None:                          # P2
        b_steps = (steps["median_ratio"] is not None and steps["median_ratio"] >= STEPS_CELL
                   and steps["positive"] >= need(n_path))
        gf_ratio = statistics.median([x["ratio"] for x in gf["per_pair"] if finite(x.get("ratio"))]) \
            if any(finite(x.get("ratio")) for x in gf["per_pair"]) else None
        b_gf = gf_ratio is not None and gf_ratio <= GF_CELL and gf["negative"] >= need(gf["n"])
        benefit = b_steps or b_gf
        checks.update(B_native_steps_ratio=steps["median_ratio"], B_native_gf_ratio=gf_ratio, B_steps=b_steps, B_gf=b_gf)
    else:                                                           # P1, P3
        benefit = (steps["median_ratio"] is not None and steps["median_ratio"] >= STEPS_CAPS
                   and steps["positive"] >= need(n_path))
        checks.update(B_native_steps_ratio=steps["median_ratio"])
    checks["B_benefit"] = benefit
    if n_dice < MIN_N or n_path < MIN_N:
        verdict = f"只描述（可比对数 Dice {n_dice}、路径 {n_path}，少于 {MIN_N}）"
    elif not checks["S1_all_runs_certified_export"] or not checks["S2_blocked_not_more"]:
        verdict = "不列入候选（安全条件不满足）"
    elif s3 is False:
        verdict = "待审核（核对 (b) 违反 > 0；审核前不进结论）"
    elif dice_bad:
        verdict = "不列入候选（Dice 降低）"
        reasons.append(f"Dice 中位差 seg35 {d35:+.4f}，seg4 {d4:+.4f}（≤ {DICE_DOWN}）")
    elif benefit and dice_ok:
        verdict = "待核对（其余条件满足；违反累计未找到）" if s3 is None and cell == BERNSTEIN else "倾向列入候选（交总负责人）"
    else:
        verdict = "无法区分（只描述）"
    return dict(verdict=verdict, reasons=reasons, checks=checks)


def recommend(verdicts, p3_vs_p1):
    """The path suggested for the eta re-sweep (plan.md; the lead decides)."""
    lean = {a for a, v in verdicts.items() if v.startswith("倾向")}
    if "p3" in lean:
        if "p1" in lean and p3_vs_p1 is not None:
            s, d35, d4 = (p3_vs_p1["native_accepted"]["median_ratio"], p3_vs_p1["seg35"]["median"],
                          p3_vs_p1["seg4"]["median"])
            if not (s is not None and s >= 1 and d35 is not None and d35 >= DICE_NOT_DOWN and d4 is not None
                    and d4 >= DICE_NOT_DOWN):
                return "p1", "P3 与 P1 都倾向列入，但 P3 对 P1 的原生步数比 < 1 或 Dice 中位差 < −.005：取改动少的 P1"
        return "p3", "P3 倾向列入候选" + ("（且对 P1 不差）" if "p1" in lean else "")
    if "p1" in lean:
        return "p1", "P1 倾向列入候选，P3 不满足"
    if "p2" in lean:
        return "p2", "只有 P2 倾向列入候选"
    return "p0", "没有臂倾向列入候选：维持现行路径，交总负责人"


def summary(root, j53=True):
    root = Path(root)
    batch, p0rec = read_json(root/"batch.json"), read_json(root/"p0.json")
    if batch is None or p0rec is None:
        raise SummaryError(f"{root}/batch.json or p0.json missing (the driver writes them at its start)")
    m2root = Path(batch["m2v1"]["root"])
    m2batch = read_json(m2root/"batch.json")
    if m2batch is None or ms.file_sha256(m2root/"batch.json") != batch["m2v1"]["batch_sha256"]:
        raise SummaryError(f"{m2root}/batch.json missing or changed since the pilot recorded it")
    pairs = batch["pairs"]
    rows = {"p0": [derive(ms.pair_row(m2root, "dev", p, m2batch), j53) for p in pairs]}
    for row, pair in zip(rows["p0"], pairs):
        recorded = p0rec["pairs"].get(short_name(pair), {})
        row.update(arm="p0", p0_state=recorded.get("state"))
        if recorded.get("state") != "evaluated":
            row["comparable"] = False
    for aid in batch["arm_order"]:
        rows[aid] = [derive(ms.pair_row(root, aid, p, batch), j53) for p in pairs]
    base = "p0n" if "p0n" in batch["arm_order"] else "p0"
    others = [aid for aid in batch["arm_order"] if aid != base]
    comparisons = {aid: compare(rows[aid], rows[base]) for aid in others}
    secondary = {aid: compare(rows[aid], rows["p0"]) for aid in batch["arm_order"]} if base != "p0" else {}
    extra = {}
    for a, b in (("p3", "p1"), ("p3", "p2"), ("p4", "p3")):
        if a in rows and b in rows:
            extra[f"{a}_vs_{b}"] = compare(rows[a], rows[b])
    flags = {aid: arm_flags(aid, rows[aid], rows[base], comparisons[aid],
                            (batch["arms"][aid]["configuration"] or {}).get("jd_cell_estimate"),
                            (batch["arms"][aid]["changes"] or {}).get("max_accepted_per_level"))
             for aid in others}
    if "p4" in flags:             # exploratory (revision 3): checks kept, no candidate label, no N3 record
        flags["p4"]["checks"].pop("N3_criterion", None)
        flags["p4"].update(verdict="只描述（探索臂 P4：σ 估计域 support_shared；不贴候选标签）",
                           direction=p4_direction(extra["p4_vs_p3"]) if "p4_vs_p3" in extra else None)
    path, why = recommend({a: f["verdict"] for a, f in flags.items()}, extra.get("p3_vs_p1"))
    counts = dict(ran=sum(r["state"] == "ran" for a in batch["arm_order"] for r in rows[a]),
                  not_run=sum(r["state"] == "not_run" for a in batch["arm_order"] for r in rows[a]),
                  failed_no_status=sum(r["state"] == "failed_no_status" for a in batch["arm_order"] for r in rows[a]),
                  evaluator_certificate_passed=sum((r["certificate"] or {}).get("passed") is True
                                                   for a in batch["arm_order"] for r in rows[a]),
                  args_mismatch=sum(bool(r["args_differ"]) for a in batch["arm_order"] for r in rows[a]),
                  eta_mismatch=sum(r["eta_ok"] is False for a in batch["arm_order"] for r in rows[a]),
                  p0_comparable=sum(r["comparable"] for r in rows["p0"]),
                  j53_computed=sum(r["j53_final"]["state"] == "computed" for a in ["p0", *batch["arm_order"]]
                                   for r in rows[a]),
                  j53_below_certified_bound=sum(r["j53_final"]["ge_certified_bound"] is False
                                                for a in ["p0", *batch["arm_order"]] for r in rows[a]),
                  bernstein_violation_runs=[dict(arm=a, pair=r["pair"], short=r["short"],
                                                 violations=(r["bernstein"] or {}).get("violations"))
                                            for a in batch["arm_order"] for r in rows[a]
                                            if isinstance((r.get("bernstein") or {}).get("violations"), int)
                                            and r["bernstein"]["violations"] > 0])
    return dict(schema=SCHEMA, note=NOTE, written_utc=utc(), root=str(root), stage=batch["stage"], pairs=pairs,
                arm_order=batch["arm_order"], arms={a: dict(label=batch["arms"][a]["label"],
                                                            changes=batch["arms"][a]["changes"]) for a in batch["arm_order"]},
                code_sha=batch["code_sha"], m2v1=batch["m2v1"], eta=batch["eta"], wall_seconds=batch["wall_seconds"],
                thresholds=dict(MIN_N=MIN_N, DICE_NOT_DOWN=DICE_NOT_DOWN, DICE_DOWN=DICE_DOWN, STEPS_CAPS=STEPS_CAPS,
                                STEPS_CELL=STEPS_CELL, GF_CELL=GF_CELL, MAJORITY=MAJORITY),
                baseline=base, counts=counts, rows=rows, comparisons=comparisons, secondary_comparisons=secondary,
                extra_comparisons=extra, flags=flags, recommendation=dict(path=path, reason=why))


# ---- document --------------------------------------------------------------------------------------------------------
def fmt(value, nd=4):
    return ms.fmt(value, nd)


def label_of(doc, aid):
    return "P0（M2V1）" if aid == "p0" else doc["arms"][aid]["label"]


def stats_line(name, s):
    if not s["n"]:
        return f"| {name} | 0 | — | — | — | — | — |"
    per = "；".join(f"{short_name(x['pair'])} {x['d']:+.4g}" for x in s["per_pair"])
    ratio = fmt(s.get("median_ratio"), 3) if "median_ratio" in s else "—"
    return f"| {name} | {s['n']} | {s['median']:+.4g} | {ratio} | {s['positive']} | {s['negative']} | {per} |"


def markdown(doc):
    out = [f"# 路径改进试验汇总（{doc['root']}）", ""]
    bad = doc["counts"]["bernstein_violation_runs"]
    if bad:
        out += ["**certificate_bernstein 核对 (b) 有违反的运行（D53）：" + "；".join(
            f"{doc['arms'][b['arm']]['label']} {b['short']}：{b['violations']}" for b in bad)
                + "。这些运行要单独报总负责人，并用不含预筛的证书配置重新认证终态；审核完成之前，它们的数字不进入任何"
                "结论（下面的读数标签也一样）。**", ""]
    else:
        out += ["certificate_bernstein 核对 (b)：没有违反计数 > 0 的运行。", ""]
    out += [f"**{doc['note']}**", "",
           f"代码 {doc['code_sha']}；P0 来自 {doc['m2v1']['root']}（代码 {doc['m2v1']['code_sha']}）；η = α "
           f"{doc['eta']['alpha']:g}，ν {doc['eta']['nu']:.6g}，τ {doc['eta']['tau']}；每次 {doc['wall_seconds']:g} s。", ""]
    out += [f"主要对照（基线）：**{label_of(doc, doc['baseline'])}**；P0（M2V1）只作次要参照。", ""]
    out += ["## 各臂", "", "| 臂 | 相对 M2V1 dev 臂的改动 |", "|---|---|", "| P0（M2V1） | 无（M2V1 运行，复用；次要参照） |"]
    out += [f"| {label_of(doc, a)} | {json.dumps(doc['arms'][a]['changes'], ensure_ascii=False)} |" for a in doc["arm_order"]]
    c = doc["counts"]
    out += ["", f"运行 {c['ran']}，未运行 {c['not_run']}，失败（无 status.json）{c['failed_no_status']}，评测证书通过 "
            f"{c['evaluator_certificate_passed']}，args 不符 {c['args_mismatch']}，η 不符 {c['eta_mismatch']}；P0 可比 "
            f"{c['p0_comparable']}；53 点最小 J 算出 {c['j53_computed']}，低于证书下界 {c['j53_below_certified_bound']}。", ""]
    out += ["## 逐对逐臂", "", "| 对 | 臂 | 状态 | seg35 | seg4 | HD95 | SDlogJ 官方 | SDlogJ 脑内 | 证书 | 原生接受步 | "
            "原生秒 | 原生末 GF_B/T_B | 粗层达阈 | 受阻层 | JV 拒绝 | 各层上限 | 53 点最小 J（末态） |", "|"+"---|"*17]
    for k, pair in enumerate(doc["pairs"]):
        for aid in ["p0", *doc["arm_order"]]:
            r = doc["rows"][aid][k]
            m, cert = r["metrics"], r["certificate"] or {}
            out.append(f"| {short_name(pair)} | {label_of(doc, aid)} | {r['state']} / {r['run_status']} | "
                       f"{fmt(m.get('seg35'))} | {fmt(m.get('seg4'))} | {fmt(m.get('hd95'), 3)} | "
                       f"{fmt(m.get('sdlogj_official'), 3)} | {fmt(m.get('sdlogj_brain'), 3)} | "
                       f"{cert.get('passed')} / {fmt(cert.get('min_jd_lower_bound'), 3)} | {fmt(r['native_accepted'])} | "
                       f"{fmt(r['native_seconds'], 3)} | {fmt(r['native_gf_over_tb'], 3)} | {fmt(r['coarse_reached'])} | "
                       f"{fmt(r['n_blocked'])} | {fmt(r['jv_rejections'])} | "
                       f"{'/'.join('—' if c is None else str(c) for c in r.get('caps') or []) or '—'} | "
                       + (fmt(r['j53_final']['min_j53'], 3) if r['j53_final']['state'] == 'computed'
                          else r['j53_final']['state']) + " |")
    head = ["", "| 量 | n | 中位差 | 中位比 | 正 | 负 | 逐对 |", "|---|---|---|---|---|---|---|"]
    base = label_of(doc, doc["baseline"])
    for aid, comp in doc["comparisons"].items():
        out += ["", f"## {label_of(doc, aid)} − {base} 的配对差（主要对照；只描述）", *head]
        for m, name in (*MEASURES, *PATH_MEASURES):
            out.append(stats_line(name, comp[m]))
    for aid, comp in doc["secondary_comparisons"].items():
        out += ["", f"## {label_of(doc, aid)} − P0（M2V1）的配对差（次要参照：代码与时段不同；只描述）", *head]
        for m, name in (*MEASURES, *PATH_MEASURES):
            out.append(stats_line(name, comp[m]))
    for key, comp in doc["extra_comparisons"].items():
        a, b = key.split("_vs_")
        title = ("σ 估计域的效应：support_shared 对 brain，其余同 P3；探索，只描述" if key == "p4_vs_p3" else "只描述")
        out += ["", f"## {label_of(doc, a)} − {label_of(doc, b)} 的配对差（{title}）", *head]
        for m, name in (*MEASURES, *PATH_MEASURES):
            out.append(stats_line(name, comp[m]))
    out += ["", f"## 读数（预登记的判定，对 {base}；不是方法结论）", ""]
    for aid, f in doc["flags"].items():
        out.append(f"- **{label_of(doc, aid)}**：{f['verdict']}。检查：{json.dumps(f['checks'], ensure_ascii=False)}")
        out += [f"  - {x}" for x in f["reasons"]]
        if f.get("direction"):
            d = f["direction"]
            out.append(f"  - 方向假设（预登记，描述用）：{d['hypothesis']}。P4 − P3：n {d['n']}，seg4 中位差 "
                       f"{fmt(d['median_seg4'])}，seg35 中位差 {fmt(d['median_seg35'])}：{d['reading']}。")
    rec = doc["recommendation"]
    out += ["", f"η 重扫路径的建议（交总负责人定）：**{label_of(doc, rec['path']) if rec['path'] != 'p0' else 'P0（现行）'}**"
            f"——{rec['reason']}。", ""]
    out += ["## certificate_bernstein 记录", "", "| 对 | 臂 | S̃ 计算 | 行内违反和 | 运行级累计 | 各层（计算 / 违反 / 抬高单元最大 / |S| 最大） |",
            "|---|---|---|---|---|---|"]
    for k, pair in enumerate(doc["pairs"]):
        for aid in doc["arm_order"]:
            b = doc["rows"][aid][k].get("bernstein")
            if not b or not b["computations"]:
                continue
            levels = "；".join(f"L{lv} {e['computations']} / {e['violations']} / {e['max_cells_raised']} / "
                              f"{e['max_cells_bernstein']}" for lv, e in sorted(b["per_level"].items()))
            out.append(f"| {short_name(pair)} | {label_of(doc, aid)} | {b['computations']} | {b['violations_in_rows']} | "
                       f"{b['violations_total'] if b['total_found'] else '未找到'} | {levels} |")
    sup = [(label_of(doc, aid), r) for aid in doc["arm_order"] for r in doc["rows"][aid] if r.get("support")]
    if sup:
        out += ["", "## σ 估计域的支撑记录（非 brain 的运行；hyper.json support_levels）", "",
                "| 运行 | 层 | N_c/N_q | 空支撑通道数 | N_S | σ_S²（层首） | σ_S²（轮用 / 轮后） |", "|---|---|---|---|---|---|---|"]
        for lab, r in sup:
            for e in r["support"]:
                ratio = "，".join(fmt(x, 3) for x in e["N_c_over_N_q"] or [])
                rounds = "；".join(f"{fmt(a, 3)} / {fmt(b, 3)}" for a, b in zip(e["sigma2_S_used"], e["sigma2_S_post"]))
                out.append(f"| {lab}-{r['short']} | {e['level']} | {ratio} | {e['n_empty']} | {fmt(e['N_S'], 4)} | "
                           f"{fmt(e['sigma2_S_start'], 3)} | {rounds or '—'} |")
    records = [dict(r["levels"], label=f"{label_of(doc, aid)}-{r['short']}") for aid in ["p0", *doc["arm_order"]]
               for r in doc["rows"][aid] if r.get("levels")]
    out += ["", lt.markdown(records, title="各层读数（level_table）")]
    return "\n".join(out)+"\n"


def strip_volatile(doc):
    return {k: v for k, v in doc.items() if k != "written_utc"}


def write(root, j53=True):
    """summary.json / summary.md once; a different recomputation goes to summary_<UTC>.*; returns the paths."""
    root = Path(root)
    doc = summary(root, j53)
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
    parser = argparse.ArgumentParser(prog="python -B experiments/pilot/summarize.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--no-j53", action="store_true", help="do not compute the 53-point minimum J of the final states "
                        "(recorded as to be computed at the audit)")
    args = parser.parse_args(argv)
    try:
        js, md, state = write(args.root, not args.no_j53)
    except SummaryError as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 8
    print(f"summary: {js} {md} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
