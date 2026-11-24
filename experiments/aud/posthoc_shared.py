"""AUD post hoc (added AFTER seeing summary_r1.md, section 6: the sign of the fixed-sigma comparison depends on how the
structure channels share sigma; not in the plan, descriptive only).  Reads raw/ only; no model, no data/.

The profile objective with one brain-wide variance shared by the structure channels (a what-if of the audit: no support
weight, so NOT the contract's support_shared of C04-6):
    L_shared(u) = (N_q/2) [log sigma~_I^2(u) + K log sigma~_S^2(u)] + Pi_D + Pi_J + Pi_C,
    sigma~_S^2(u) = sum_{c in S} <rho_c^2>_B / (K (N_q + 2)),  K = number of structure channels kept.
(The joint objective with sigma_S shared has K (N_q/2) log sigma_S^2, hence the factor K.)  Also the fixed-sigma F with the
shared sigma~_S of the M2V1 final field and the critical nu there.

The denominator K (N_q + 2) is not the K N_q + 2 of the shared model's M step at a = b = 0 (C11-8; review
AUD_复核_20261102, minor 5): the differences of L_shared between fields are unaffected, the fixed-sigma rows differ by about
1e-6 relative.  Kept, so that the outputs stay fb_audit_v1/posthoc_shared.*, which were reproduced byte for byte from the
snapshot 6f7970e (runs/20261102-fb-audit/result.md); this comment was added after that snapshot and changes no output.

    python -B experiments/aud/posthoc_shared.py --root /home/rmcfz/seq_vp_tag_results/fb_audit_v1
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

GROUPS = (("13 组共享", None), ("白质 + 皮层共享", ("cerebral_wm", "cerebral_cortex")))
FIELDS = ("A0", "A1", "A4p", "P3", "P4")


def shared_terms(field, n_q, indices):
    """(N_q/2) log sigma~_I^2 and K (N_q/2) log sigma~_S^2 over the structure channels `indices` (sigma~_S^2 with the
    denominator K (N_q + 2); module docstring)."""
    rho2 = np.asarray(field["rho2_B"], dtype=np.float64)
    k = len(indices)
    return .5*n_q*np.log(rho2[0]/(n_q+2.)), k*.5*n_q*np.log(rho2[indices].sum()/(k*(n_q+2.)))


def shared_difference(x, r, n_q, indices, nu0):
    """L_shared(X) - L_shared(R) on NU_GRID x TAU_GRID and the critical nu per tau."""
    i_x, s_x = shared_terms(x, n_q, indices)
    i_r, s_r = shared_terms(r, n_q, indices)
    D = (i_x-i_r)+(s_x-s_r)
    grid = x["prior_pass"]["s_grid"]
    d_vol_s = A.vol_table(x)-A.vol_table(r)
    d_vol = {tau: (0. if s is None else float(d_vol_s[grid.index(s)])) for tau, s in A.TAU_GRID}
    d_diff, d_curl = x["F_diff"]-r["F_diff"], x["F_curl"]-r["F_curl"]
    return dict(D=float(D), D_intensity=float(i_x-i_r), D_structure=float(s_x-s_r),
                cells=[dict(nu=nu, tau=tau, M=float(D)+d_diff+nu/nu0*d_curl+d_vol[tau]) for nu in A.NU_GRID for tau, _ in A.TAU_GRID],
                nu_of_tau=[dict(tau=tau, **A.linear_crossing(float(D)+d_diff+d_vol[tau], d_curl, nu0)) for tau, _ in A.TAU_GRID])


def fixed_shared_difference(x, r, sigma2, nu0):
    """F(X) - F(R) at one common sigma: at nu0, tau .5 and the critical nu per tau."""
    data = float(sum(A.data_at_sigma(x["rho2_B"], sigma2))-sum(A.data_at_sigma(r["rho2_B"], sigma2)))
    grid = x["prior_pass"]["s_grid"]
    d_vol_s = A.vol_table(x)-A.vol_table(r)
    d_vol = {tau: (0. if s is None else float(d_vol_s[grid.index(s)])) for tau, s in A.TAU_GRID}
    d_diff, d_curl = x["F_diff"]-r["F_diff"], x["F_curl"]-r["F_curl"]
    return dict(data=data, M=data+d_diff+d_curl+d_vol[.5],
                nu_of_tau=[dict(tau=tau, **A.linear_crossing(data+d_diff+d_vol[tau], d_curl, nu0)) for tau, _ in A.TAU_GRID])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True)
    ns = parser.parse_args(argv)
    root = Path(ns.root)
    pairs, raw = A.load_raw(root)
    shorts = sorted(raw)
    names = pairs[shorts[0]]["channel_names"]
    nu0 = pairs[shorts[0]]["eta"]["nu"]
    doc = dict(schema="bayesreg.aud.posthoc_shared.v1",
               note="added after seeing summary_r1.md section 6; not in the plan; descriptive; no support weight (not C04-6)",
               pairs=shorts, profile={}, fixed={})
    for label, chosen in GROUPS:
        indices = list(range(1, len(names))) if chosen is None else [names.index(n) for n in chosen]
        doc["profile"][label] = {x: {s: shared_difference(raw[s][x], raw[s]["M2V1"], pairs[s]["N_q"], indices, nu0) for s in shorts}
                                 for x in FIELDS}
    for x in FIELDS:
        doc["fixed"][x] = {s: fixed_shared_difference(raw[s][x], raw[s]["M2V1"], A.shared_structure_sigma(raw[s]["M2V1"], pairs[s]["N_q"]), nu0)
                           for s in shorts}
    out = ["# 计划外的只读对照：结构层共用一个 σ_S（看过 summary_r1.md 第 6 节之后加的，只描述）", "",
           "全脑共享，没有合同 C04-6 的支撑权，不是 support_shared。能量单位 1e6。L_shared(X) − L_shared(M2V1)，括号里是 X 更低的对数。", ""]
    for label, _ in GROUPS:
        out += [f"## {label}（强度保留自己的 σ）", "", "| X | ν | τ = .5 | τ = 1 | τ = ∞ |", "|---|---|---|---|---|"]
        for x in FIELDS:
            rows = doc["profile"][label][x]
            for j, nu in enumerate(A.NU_GRID):
                cells = []
                for k in range(len(A.TAU_GRID)):
                    values = [rows[s]["cells"][j*len(A.TAU_GRID)+k]["M"] for s in shorts]
                    cells.append(f"{A.mean(values)/1e6:+.3f}（{sum(v < 0. for v in values)}/5）")
                out.append(f"| {x} | {nu:.3g} | "+" | ".join(cells)+" |")
        out += ["", "数据项的差（5 对均值）与临界 ν（逐对）：", "", "| X | 强度 | 结构层 | 临界 ν（τ = .5） | 临界 ν（τ = 1） | 临界 ν（τ = ∞） |", "|---|---|---|---|---|---|"]
        for x in FIELDS:
            rows = doc["profile"][label][x]
            crit = [", ".join(A.crossing_text(rows[s]["nu_of_tau"][k], "nu") for s in shorts) for k in range(len(A.TAU_GRID))]
            out.append(f"| {x} | {A.mean([rows[s]['D_intensity'] for s in shorts])/1e6:+.3f} | {A.mean([rows[s]['D_structure'] for s in shorts])/1e6:+.3f} | "
                       +" | ".join(crit)+" |")
        out.append("")
    out += ["## 固定 σ 的 F：结构层共享 σ̃_S 取 M2V1 终态（强度 σ̃_I 同样取 M2V1 终态）", "", "| X | 5 对的 F 差（ν 21.855，τ .5） | 临界 ν（τ = .5，逐对） |", "|---|---|---|"]
    for x in FIELDS:
        rows = doc["fixed"][x]
        out.append(f"| {x} | "+", ".join(A.mil(rows[s]["M"]) for s in shorts)+" | "+", ".join(A.crossing_text(rows[s]["nu_of_tau"][0], "nu") for s in shorts)+" |")
    for name, text in (("posthoc_shared.json", json.dumps(doc, indent=1, default=str)+"\n"), ("posthoc_shared.md", "\n".join(out)+"\n")):
        path = root/name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
