"""CVG: Markdown tables from stiffness.json files (no computation on the model).

    python -B experiments/cvg/summarize.py STIFF_DIR [STIFF_DIR ...]
"""
import json
from pathlib import Path
import sys


def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    return f"{x:.{digits}g}"


def main(dirs):
    for d in dirs:
        doc = json.loads((Path(d)/"stiffness.json").read_text())
        print(f"\n### {doc['label']} ({d})")
        print(f"- state {doc['state']} hash {doc['state_hash'][:12]}; F_B {doc['F_B']:.10g}; cert min J lb "
              f"{fmt(doc['certificate']['min_jd_lower_bound'])}; metric c_dat {fmt(doc['metric']['c_dat'])} b_P "
              f"{fmt(doc['metric']['b_P'])} shift {fmt(doc['metric']['shift'])}; M spectrum "
              f"[{fmt(doc['metric']['M_spectrum']['min'])}, {fmt(doc['metric']['M_spectrum']['max'])}]")
        g = doc["gradient"]
        print(f"- GF_B {fmt(g['GF_B'], 5)} (data {fmt(g['GF_B_data'])}, prior {fmt(g['GF_B_prior'])}); sum check "
              f"{fmt(g['sum_check'])}")
        rows = sorted(g["blocks"].items(), key=lambda kv: -kv[1]["GF"])
        print("| block | GF | share of ‖g‖² | cos(g_b, g) |\n|---|---|---|---|")
        for k, v in rows:
            print(f"| {k} | {fmt(v['GF'])} | {fmt(v['share_of_total_sq'])} | {fmt(v['cos_with_total'])} |")
        print("- gradient location:", {k: (v["nodes"], round(v["share_g2"], 4)) for k, v in doc["gradient_location"].items()})
        print("- JD at state:", {k: doc["jd_scaling_at_state"].get(k) for k in ("min_J_a", "fraction_r_lt_1", "n_r_lt_1",
                                                                            "min_r")})
        print("- data diagonal:", doc["data_diagonal"])
        if "channel_support" in doc:
            print("- channel support:", {k: (round(v["fraction_of_brain_nodes"], 3), f"{v['sigma2']:.3g}",
                                         f"{v['sigma2_over_support']:.3g}")
                                        for k, v in doc["channel_support"]["channels"].items()})
        if "V5_scaling" in doc:
            print("- V5 scaling:", doc["V5_scaling"])
        sp = doc.get("spectra", {})
        if sp:
            print("| variant | λ_max | λ_min (SA) | κ | LA conv/matvecs | SA conv/matvecs |\n|---|---|---|---|---|---|")
            for v, e in sp.items():
                la, sa = e["LA"], e["SA"]
                print(f"| {v} | {fmt(la['values'][-1])} | {fmt(sa['values'][0])} | {fmt(e.get('kappa'))} | "
                      f"{la['converged']}/{la['matvecs']} | {sa['converged']}/{sa['matvecs']} |")
            for v, e in sp.items():
                for which in ("LA", "SA"):
                    ray = e[which].get("rayleigh_extreme")
                    if ray:
                        tot = ray["total"]
                        top = sorted(ray["blocks"].items(), key=lambda kv: -kv[1])[:5]
                        print(f"- {v} {which} Ritz vector: total {fmt(tot)}; blocks "
                              + ", ".join(f"{k} {fmt(val/tot if tot else None)}" for k, val in top)
                              + "; location " + ", ".join(f"{k} {round(val, 3)}" for k, val in ray["location"].items()))
        per = doc.get("per_block_lambda_max_V0")
        if per:
            print("- per-block λ_max(M⁻¹H_b):", ", ".join(f"{k} {fmt(v['values'][-1])}" for k, v in
                                                        sorted(per.items(), key=lambda kv: -kv[1]['values'][-1])))


if __name__ == "__main__":
    main(sys.argv[1:])
