"""JDT: summary table of the L0 diagnostic outputs (jdt_v1/<label>.json) -> jdt_v1/summary.md, summary.json."""
import json
from pathlib import Path
import sys

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/rmcfz/seq_vp_tag_results/jdt_v1")
LABELS = ("R1-k500", "R1-k1000", "R1-k2000", "R1-final", "R2-k50", "R2-k200", "R2-k500", "R2-k1000", "R2-k2000",
          "R2-final", "R7-k1000", "R7-final")
EST = (("gauss27+corners", "g27c"), ("certificate", "cert"), ("certificate_bernstein", "cert_b"),
       ("closed27_53pt", "J53"), ("dense_min", "dense"))
CHECKS_TRUE = ("ell_subset_equals_all_bitwise", "ell_equals_b_on_prescreen_cells_bitwise", "fallback_ell_not_positive",
               "closed27_default_equals_cell_jd_estimate", "r_certificate_equals_library_bitwise",
               "r_restricted_equals_unrestricted_bitwise", "r_new_ge_r_cert_everywhere", "r_new_le_r_53_everywhere",
               "all_bt_positive")
CHECKS_ZERO = ("n_cells_bt_above_J53", "n_cells_bt_full_above_J53", "n_low_cells_bt_above_dense",
               "n_low_cells_b_above_dense")


def main():
    docs = {lab: json.loads((OUT/f"{lab}.json").read_text()) for lab in LABELS if (OUT/f"{lab}.json").exists()}
    lines = ["# JDT L0 诊断汇总（自动生成；诊断，不是方法结果）", ""]
    lines += ["## 冻结节点（r_a < .01）：节点数 / ‖g‖² 份额", "",
              "| 状态 | " + " | ".join(s for _, s in EST) + " | cert 冻结而 dense r ≥ .1 | cert_b 冻结而 dense r ≥ .1 |",
              "|---" * (len(EST)+3) + "|"]
    for lab, d in docs.items():
        c = d["classes"]
        cells = [f"{c[f'{n}_r_lt_0.01']['nodes']} / {c[f'{n}_r_lt_0.01']['share_g2']:.3f}" for n, _ in EST]
        a = c["certificate_r_lt_0.01_but_dense_r_ge_0.1"]
        b = c["certificate_bernstein_r_lt_0.01_but_dense_r_ge_0.1"]
        lines.append(f"| {lab} | " + " | ".join(cells) + f" | {a['nodes']} / {a['share_g2']:.3f} | {b['nodes']} / {b['share_g2']:.3f} |")
    for lim in ("0.1", "1"):
        lines += ["", f"## r_a < {lim}：节点数 / ‖g‖² 份额", "",
                  "| 状态 | " + " | ".join(s for _, s in EST) + " |", "|---" * (len(EST)+1) + "|"]
        for lab, d in docs.items():
            c = d["classes"]
            lines.append(f"| {lab} | " + " | ".join(
                f"{c[f'{n}_r_lt_{lim}']['nodes']} / {c[f'{n}_r_lt_{lim}']['share_g2']:.3f}" for n, _ in EST) + " |")
    lines += ["", "## 单元与下界", "",
              "| 状态 | hull / 预筛 / 回退 | S（hull 且 b < .25） | 抬高 | S 上 b/dense 中位、最小 | S 上 b̃/dense 中位、最小 | "
              "min b | min b̃ | min J53 | min dense | n_nodes_r_lowered（cert / cert_b） | r 抬高的节点（cert_b 对 cert） | "
              "hull 全体 ℓ < b |", "|---" * 13 + "|"]
    for lab, d in docs.items():
        x, c = d["cells"], d["classes"]
        rb, rt = x["ratio_b_over_dense_on_S"], x["ratio_bt_over_dense_on_S"]
        lines.append(f"| {lab} | {x['n_hull']} / {x['n_prescreen']} / {x['n_fallback']} | {x['n_hull_below_act']} | "
                     f"{x['n_raised']} | {rb['median']:.3g}, {rb['min']:.3g} | {rt['median']:.10f}, {rt['min']:.4f} | "
                     f"{x['min_b']:.3g} | {x['min_bt']:.3g} | {x['min_J53']:.3g} | {x['min_dense']:.3g} | "
                     f"{c['certificate_n_nodes_r_lowered_vs_default']} / {c['certificate_bernstein_n_nodes_r_lowered_vs_default']} | "
                     f"{c['n_nodes_r_raised_new_vs_certificate']} | {x['n_hull_all_ell_below_b']} |")
    lines += ["", "## 核对", ""]
    bad = []
    for lab, d in docs.items():
        ch = d["checks"]
        for k in CHECKS_TRUE:
            if ch[k] is not True:
                bad.append(f"{lab}: {k} = {ch[k]}")
        for k in CHECKS_ZERO:
            if ch[k] != 0:
                bad.append(f"{lab}: {k} = {ch[k]}")
        if not (d["check"]["GF_B_rel_diff"] <= 1e-10 and d["check"]["F_B_rel_diff"] <= 1e-10
                and d["check"]["hash_matches_accept_row"]):
            bad.append(f"{lab}: state check {d['check']}")
    lines.append("全部通过：" + ", ".join(CHECKS_TRUE + CHECKS_ZERO) if not bad else "未通过：\n" + "\n".join(bad))
    margins = {lab: d["checks"]["min_J53_minus_bt_over_S"] for lab, d in docs.items()}
    lines.append(f"\nS 上 min(J53 − b̃)：{min(margins.values()):.3g}（各状态 {', '.join(f'{k} {v:.2g}' for k, v in margins.items())}）")
    lines += ["", "## 耗时（L0，13440 单元，numba 1 线程，nice 19，机器有别的负载）", "",
              "| 状态 | cell_pass 中位 s | 全部单元预筛中位 s | 每单元 µs | S 上预筛中位 s | S 每单元 µs | 秒（整个脚本） | 峰值 RSS MB | 负载（前） |",
              "|---" * 9 + "|"]
    for lab, d in docs.items():
        t = d["timing"]
        lines.append(f"| {lab} | {t['cell_pass']['median']:.4f} | {t['prescreen_all_cells']['median']:.4f} | "
                     f"{t['per_cell_prescreen_all_us']:.3f} | {t['prescreen_S']['median']:.2e} | "
                     f"{t['per_cell_prescreen_S_us']:.3f} | {d['seconds']:.1f} | {d['peak_rss_mb']:.0f} | "
                     f"{d['loadavg_before'][0]:.2f} |")
    (OUT/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(dict(labels=list(docs), bad=bad, margins=margins), indent=1)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
