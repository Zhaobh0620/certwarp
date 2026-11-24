"""CVG: GF_B curve analysis of l0_run outputs (no computation on the model).

    python -B experiments/cvg/analyze.py --out <analysis.json> RUN_DIR [RUN_DIR ...]

Per run and level: status, accepted steps, G0, T_B, the first step k with GF_B / T_B <= 1e4, 1e3, 1e2, 10, 1; least-
squares slopes of log10 GF_B and of its running minimum against k on fixed windows (steps per decade = -1/slope); the
median BB step tau and the mean number of candidates per iteration on the same windows; F_B at the window ends; the
JD-scaling records (min J_a, refreshes) and the rejection counts.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

WINDOWS = ((0, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, 20000))
MULTIPLES = (1e4, 1e3, 1e2, 10., 1.)


def slope(k, y):
    if len(k) < 3:
        return None
    a = np.polyfit(k, y, 1)[0]
    return float(a)


def analyse(run):
    run = Path(run)
    result = json.loads((run/"result.json").read_text())
    config = json.loads((run/"config.json").read_text())
    rows = list(csv.DictReader(open(run/"trace.csv")))
    out = dict(run=str(run), args=config["args"], status=result["status"], reason=result["reason"],
               candidates=result["candidates"], rejections=result["rejections"], levels=[])
    for rnd in result["rounds"]:
        ell = rnd["level"]
        T = rnd["T_B"]
        gf = np.array([g for g in rnd["GF_B_trace"]], dtype=float)
        fb = np.array(rnd["F_B_trace"], dtype=float)
        k = np.arange(len(gf))
        acc = [r for r in rows if r["level"] == str(ell)]
        tau = np.array([float(r["tau"]) for r in acc])
        cand = np.array([int(r["candidates"]) for r in acc])
        level = dict(level=ell, status=rnd["status"], accepted=rnd["accepted"], G0=rnd["G0"], T_B=T,
                     GF_B_end=rnd["GF_B"], ratio_end=rnd["GF_B"]/T if rnd["GF_B"] is not None else None,
                     sigma2_used=rnd["sigma2_used"], metric={x: rnd["metric"].get(x) for x in ("c_dat", "b_P", "shift")},
                     jd=dict(refresh=rnd["metric"]["jd_scaling"].get("jd_scaling_refresh"),
                             min_J_a=rnd["metric"]["jd_scaling"].get("min_J_a"),
                             max_fraction_r_lt_1=rnd["metric"]["jd_scaling"].get("max_fraction_r_lt_1")),
                     seconds=rnd["seconds"], seconds_per_step=rnd["seconds"]/max(rnd["accepted"], 1),
                     min_jd_lower_bound_end=rnd["min_jd_lower_bound"])
        first = {}
        ratio = gf/T
        for m in MULTIPLES:
            hit = np.nonzero(ratio <= m)[0]
            first[f"{m:g}"] = int(hit[0]) if hit.size else None
        level["first_step_ratio_le"] = first
        runmin = np.minimum.accumulate(gf)
        wins = []
        for lo, hi in WINDOWS:
            if lo >= len(gf)-1:
                break
            sel = slice(lo, min(hi, len(gf)-1)+1)
            ks = k[sel]
            s = slope(ks, np.log10(gf[sel]))
            sm = slope(ks, np.log10(runmin[sel]))
            ts = tau[lo:min(hi, len(tau))] if len(tau) else np.array([])
            cs = cand[lo:min(hi, len(cand))] if len(cand) else np.array([])
            wins.append(dict(window=[lo, int(ks[-1])], slope_log10=s, slope_log10_runmin=sm,
                             steps_per_decade=(-1/sm) if sm and sm < 0 else None,
                             GF_ratio_start=float(ratio[lo]), GF_ratio_end=float(ratio[ks[-1]]),
                             F_B_end=float(fb[ks[-1]]), tau_median=float(np.median(ts)) if ts.size else None,
                             tau_min=float(ts.min()) if ts.size else None,
                             candidates_mean=float(cs.mean()) if cs.size else None))
        level["windows"] = wins
        # extrapolation of the last window's running-minimum slope to T_B
        last = [w for w in wins if w["slope_log10_runmin"] is not None]
        if last and rnd["status"] != "converged_gradient":
            sm = last[-1]["slope_log10_runmin"]
            level["extrapolated_more_steps_to_T_B"] = (float(np.log10(runmin[-1]/T)/(-sm)) if sm < 0 else None)
        out["levels"].append(level)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    docs = [analyse(r) for r in a.runs]
    Path(a.out).write_text(json.dumps(docs, indent=1)+"\n")
    for d in docs:
        for lv in d["levels"]:
            print(Path(d["run"]).name, f"L{lv['level']}", lv["status"], lv["accepted"], f"ratio_end {lv['ratio_end']:.4g}",
                  "first<=", lv["first_step_ratio_le"], f"{lv['seconds_per_step']:.3f}s/step",
                  "extrap", lv.get("extrapolated_more_steps_to_T_B"))
            for w in lv["windows"]:
                print("   ", w["window"], "spd", None if w["steps_per_decade"] is None else round(w["steps_per_decade"], 1),
                      "ratio", f"{w['GF_ratio_start']:.3g}->{w['GF_ratio_end']:.3g}", "tau_med", w["tau_median"],
                      "cand", w["candidates_mean"])


if __name__ == "__main__":
    main()
