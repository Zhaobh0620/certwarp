"""LMI L0 demo: the readings of both arms side by side (开发对 L0 诊断).

    python -B experiments/lmi/summarize.py <root>          # root with pg/ and lm/ of l0_demo.py; writes <root>/summary.json

Per arm: status, L0 round status, accepted steps, G0, T_B, the final GF_B and GF_B/T_B, F_B at the start and the end,
F_B and GF_B/T_B at the last accepted step before 60 / 150 / 300 / 600 / 900 s and at accepted steps 1 / 5 / 10 / 50 /
100 / 500, candidates, rejection counts, certificate rejections, the minimum certified J lower bound over the accepted
steps and at the end, peak RSS; LM also the PCG iterations (total, median, max), the largest true residual, the gamma
range of the accepted steps and how many accepted steps had each gamma decade.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

TIMES = (60., 150., 300., 600., 900.)
STEPS = (1, 5, 10, 50, 100, 500)


def read_trace(path):
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for r in rows:
        out.append({k: (None if v == "" else float(v)) for k, v in r.items()})
    return out


def arm(directory):
    directory = Path(directory)
    result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
    trace = read_trace(directory/"trace.csv")
    row = result["round"]
    target = row.get("T_B")
    at_time = {}
    for t in TIMES:
        before = [r for r in trace if r["seconds"] <= t]
        at_time[str(int(t))] = (None if not before else dict(k=int(before[-1]["k"]), F_B=before[-1]["F_B"],
                                                              GF_B_over_T_B=before[-1]["GF_B_over_T_B"]))
    at_step = {str(k): next((dict(seconds=r["seconds"], F_B=r["F_B"], GF_B_over_T_B=r["GF_B_over_T_B"])
                             for r in trace if int(r["k"]) == k), None) for k in STEPS}
    bounds = [r["min_jd_lower_bound"] for r in trace]
    out = dict(status=result["status"], round_status=row.get("status"), round_reason=row.get("reason"),
               accepted=row.get("accepted"), G0=row.get("G0"), T_B=target, GF_B=row.get("GF_B"),
               GF_B_over_T_B=row.get("GF_B_over_T_B"), F_B_start=row.get("F_B_start"), F_B_end=row.get("F_B_end"),
               at_time=at_time, at_step=at_step, candidates=result["candidates"], rejections=result["rejections"],
               certificate_rejections=result["certificate_rejections"],
               min_jd_lower_bound_min=min(bounds) if bounds else None,
               min_jd_lower_bound_end=bounds[-1] if bounds else None, solve_seconds=result["solve_seconds"],
               peak_rss_mb=result["peak_rss_mb"], counters=row.get("counters"))
    if result["pcg"]["directions"]:
        gammas = [r["gamma"] for r in trace if r["gamma"] is not None]
        decades = {}
        for g in gammas:
            key = f"1e{int(np.floor(np.log10(g)))}"
            decades[key] = decades.get(key, 0)+1
        its = [int(r["pcg_iterations"]) for r in trace if r["pcg_iterations"] is not None]
        out.update(pcg=result["pcg"], accepted_pcg_iterations_median=float(np.median(its)) if its else None,
                   gamma_accepted_min=min(gammas) if gammas else None, gamma_accepted_max=max(gammas) if gammas else None,
                   gamma_accepted_decades=dict(sorted(decades.items())),
                   attempts_per_accepted_median=float(np.median([r["attempts"] for r in trace])) if trace else None,
                   lm=row.get("metric", {}).get("lm"))
    return out


R1_TRACE = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1/runs/R1-struct/trace.csv")


def compare_r1(pg_trace_path, r1_path=R1_TRACE):
    """The PG arm against CVG R1 (same configuration, code be844cd): over the common accepted steps, how many have F_B,
    GF_B and the certified bound bit for bit equal, and the first step that differs (None: all equal)."""
    if not Path(r1_path).is_file():
        return None
    with open(r1_path, encoding="utf-8") as handle:
        r1 = {int(r["k"]): r for r in csv.DictReader(handle)}
    with open(pg_trace_path, encoding="utf-8") as handle:
        pg = list(csv.DictReader(handle))
    equal, first = 0, None
    for r in pg:
        k = int(r["k"])
        ref = r1.get(k)
        if ref is None:
            break
        same = all(float(r[a]) == float(ref[b]) for a, b in (("F_B", "F_B"), ("GF_B", "GF_B"),
                                                               ("min_jd_lower_bound", "min_jd_lower_bound")))
        if same and first is None:
            equal += 1
        elif first is None:
            first = k
    return dict(r1_trace=str(r1_path), common_steps=min(len(pg), len(r1)), bitwise_equal_prefix=equal,
                first_differing_step=first)


def main(root):
    root = Path(root)
    summary = {name: arm(root/name) for name in ("pg", "lm") if (root/name/"result.json").is_file()}
    if "pg" in summary:
        summary["pg"]["vs_cvg_r1"] = compare_r1(root/"pg"/"trace.csv")
    summary["note"] = "开发对 L0 诊断 (single development pair, level 0, one run per arm, 900 s wall clock each)"
    (root/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
