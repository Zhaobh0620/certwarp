"""SIG: tables from sigma_options.json (reads the JSON only; writes sigma_options.md next to it).

    python -B experiments/sig/summarize.py [/home/rmcfz/seq_vp_tag_results/sig_v1/sigma_options.json]
"""
import json
from pathlib import Path
import sys

import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/rmcfz/seq_vp_tag_results/sig_v1/sigma_options.json")
MAIN = ["O0", "O1-union0-drop@0.05", "O1-fixed-drop@0.05", "O1-fixeddil1-drop@0.05", "O1-soft-max",
        "O1-union0-plugin@0.05", "O1-twovar@0.05",
        "O2-prior-n0=50", "O2-prior-sigI-k0.1", "O2-prior-sigI-k1", "O2-prior-popO0-k10", "O2-floor-0.3sigI",
        "O2-floor-1sigI", "O3-t-round1", "O3-t-fixedpoint", "O4-K6-O0", "O4-K4-O0", "O4-K6-O1drop@0.05",
        "O5-shared", "O5-shared-support", "O5-hellinger1", "O5-tied-lambda1"]
PER_CHANNEL = ["O0", "O1-union0-drop@0.05", "O1-fixeddil1-drop@0.05", "O1-soft-max", "O2-prior-sigI-k1",
               "O2-floor-0.3sigI", "O3-t-round1", "O3-t-fixedpoint", "O5-shared", "O5-shared-support", "O5-hellinger1"]
SMALL = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13]       # the 10 small structure groups of the CVG report


def g(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}g}"


def main():
    doc = json.loads(SRC.read_text())
    lines = [f"# SIG 实算表（由 {SRC.name} 生成；开发集与 T0 各一对，L0，单次，诊断）", "",
             f"- 快照 {doc['snapshot']}；脚本 sha256 {doc['script_sha256'][:16]}…；峰值 RSS {doc['peak_rss_mb']:.0f} MB；"
             f"{doc['seconds']:.0f} s", ""]
    for name, p in doc["pairs"].items():
        names = p["class_names"]
        lines += [f"## {name}（{p['pair']}）", "",
                  f"N_q = {p['N_q']:.6g}，掩膜节点 {p['n_mask_nodes']}，b_P = {p['b_P']:.6g}，Λ_min = {p['Lambda_min']:.4g}；"
                  f"参照运行 {p['reference']['run']}：G0 {p['reference']['G0']:.10g}，c_dat {p['reference']['c_dat']:.10g}", ""]
        u0 = p["states"]["u0"]["options"]
        o0 = u0["O0"]
        lines += [f"O0 复算核对：σ² 与参照运行 sigma2_used 的最大相对差 "
                  f"{max(abs(a-b)/b for a, b in zip(o0['sigma2'], p['reference']['sigma2_used'])):.2e}；"
                  f"c_dat {o0['c_dat']:.10g}；G0 {o0['data_grad_norm']:.10g}"]
        if "spectrum" in o0:
            lines += [f"；λ_max(V0, u = 0) {o0['spectrum']['lambda_max']:.6g}"]
        lines += ["", "### 表 A（u = 0）：各选项的标量", "",
                  "| 选项 | c_dat | ÷O0 | G0（数据梯度） | ÷O0 | λ_max(M⁻¹H) | κ(V0) 上界 | 结构层能量/强度能量 | 数据对角 max/均值 | 小组通道占数据曲率迹 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for on in MAIN:
            if on not in u0:
                continue
            o = u0[on]
            sp = o.get("spectrum") or {}
            kb = (sp.get("kappa_bound") or {}).get("kappa_upper")
            e = o["energy"]
            small = sum(o["share_trace"][c] for c in SMALL) if len(o["share_trace"]) == 14 else None
            lines.append(f"| {on} | {g(o['c_dat'])} | {g(o['c_dat']/o0['c_dat'])} | {g(o['data_grad_norm'], 4)} | "
                         f"{g(o['data_grad_norm']/o0['data_grad_norm'])} | {g(sp.get('lambda_max'))} | {g(kb)} | "
                         f"{g(sum(e[1:])/e[0])} | {g(o['diag_max_over_mean'])} | {g(small)} |")
        lines += ["", "### 表 B（u = 0）：各通道相对强度通道的权重 σ_I²/σ_c²（O3 为曲率加权的 w/σ² 之比）", "",
                  "| 通道 | " + " | ".join(PER_CHANNEL) + " | 支撑比例（union, .05） |",
                  "|---|" + "---|"*(len(PER_CHANNEL)+1)]
        supp = u0["O1-union0-drop@0.05"].get("support_fraction")
        for c in range(len(names)):
            row = [f"{c} {names[c]}"]
            for on in PER_CHANNEL:
                o = u0[on]
                key = "ratio_effective" if on.startswith("O3") or on.startswith("O1-twovar") else "ratio_sigmaI_over_sigma"
                row.append(g(o[key][c]))
            row.append(g(supp[c]) if supp else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "### 表 C（u = 0）：O1 的支撑阈值 θ（两侧并集、层首冻结、支撑外丢弃）", "",
                  "| θ | c_dat | G0 | 小组通道权重比 min–max | 空支撑的通道 |", "|---|---|---|---|---|"]
        for th in ("0.01", "0.02", "0.05", "0.1", "0.2"):
            o = u0[f"O1-union0-drop@{th}"]
            r = [o["ratio_sigmaI_over_sigma"][c] for c in SMALL]
            rr = [x for x in r if x is not None]
            empty = [c for c in range(1, len(names)) if o["ratio_sigmaI_over_sigma"][c] is None]
            lines.append(f"| {th} | {g(o['c_dat'])} | {g(o['data_grad_norm'], 4)} | {g(min(rr))}–{g(max(rr))} | "
                         f"{empty if empty else '无'} |")
        lines += ["", "### 表 D：O4 合并分组（u = 0）", ""]
        for key in ("O4-K6-O0", "O4-K4-O0", "O4-K6-O1drop@0.05", "O4-K4-O1drop@0.05"):
            o = u0[key]
            lines.append(f"- {key}：组 {o['groups']}；σ_I²/σ_c² = {[g(x) for x in o['ratio_sigmaI_over_sigma']]}；"
                         f"c_dat {g(o['c_dat'])}")
        t = u0["O3-t-fixedpoint"]
        hist = t["iterations"]["history"]
        lines += ["", "### 表 E：Student-t（ν_t = 4）在 u = 0 固定时 E/M 迭代的 σ_c²（选列）", "",
                  f"迭代 {t['iterations']['iterations']} 次，收敛 {t['iterations']['converged']}（相对变化 < 1e-10）", "",
                  "| 迭代 | " + " | ".join(f"{c} {names[c]}" for c in (0, 1, 4, 7, 10)) + " |",
                  "|---|" + "---|"*5]
        lines.append("| 0（O0） | " + " | ".join(g(o0["sigma2"][c]) for c in (0, 1, 4, 7, 10)) + " |")
        for it, vals in hist.items():
            lines.append(f"| {it} | " + " | ".join(g(vals[c]) for c in (0, 1, 4, 7, 10)) + " |")
        lines += ["", "### 表 F：其他状态（σ 由该状态的残差算；不是新的运行）", ""]
        for sname, srec in p["states"].items():
            if sname == "u0":
                continue
            ops = srec["options"]
            r0 = [a/b for a, b in zip(ops["O0"]["sigma2"], u0["O0"]["sigma2"])]
            r1 = [(a/b if (a and b) else None) for a, b in zip(ops["O1-union0-drop@0.05"]["sigma2"],
                                                            u0["O1-union0-drop@0.05"]["sigma2"])]
            d = srec["support_drift_theta_0.05"]
            lines += [f"**{sname}**：{srec['note']}", "",
                      "| 通道 | O0 σ²（本状态）/ O0 σ²（u = 0） | O1 σ² 同比 | O0 σ_I²/σ_c² | O1 σ_I²/σ_c² | O3 不动点 σ_I²/σ_c²（曲率加权） | 支撑（u 相关并集）进 / 出 / 层首大小 |",
                      "|---|---|---|---|---|---|---|"]
            for c in range(len(names)):
                lines.append(f"| {c} {names[c]} | {g(r0[c])} | {g(r1[c])} | {g(ops['O0']['ratio_sigmaI_over_sigma'][c])} | "
                             f"{g(ops['O1-union0-drop@0.05']['ratio_sigmaI_over_sigma'][c])} | "
                             f"{g(ops['O3-t-fixedpoint']['ratio_effective'][c])} | "
                             f"{d['entered'][c]} / {d['left'][c]} / {d['size_u0'][c]} |")
            lines += ["", f"本状态 c_dat：O0 {g(ops['O0']['c_dat'])}，O1-union0-drop@.05 {g(ops['O1-union0-drop@0.05']['c_dat'])}，"
                      f"O1-unionU-drop@.05 {g(ops['O1-unionU-drop@0.05']['c_dat'])}，O5-tied {g(ops['O5-tied-lambda1']['c_dat'])}，"
                      f"O3 不动点 {g(ops['O3-t-fixedpoint']['c_dat'])}", ""]
    out = SRC.with_suffix(".md")
    if out.exists():
        raise FileExistsError(out)
    out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
