"""SEGP: the numbers of runs/20261105-segprobe/result.md that are arithmetic on the results' JSON files; nothing is read but those files (and no truth, image or posterior). Printed; the run's output is committed as report_numbers.out.

  P1 recheck    p1/p1.json against p1_recheck_from_commit/p1.json: every row of every subject equal
  P1 model      the mix - v2 gain through the slope .66-.70 of the plan (docs/bayesreg/notes/追088_12小时规划_20261105.md)
  P4 groups     0440's gap to the mean of the other 19, label by label, summed over anatomical groups, as contributions to the gap of the seg35 macro
  P4 seg4       the classes of seg4
  P4 leverage   the share of the gap that the block errors and the gaps can carry, net of what the same arithmetic gives on the other 19
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/rmcfz/seq_vp_tag_results/segprobe_v1")
TARGET = "OASIS_OAS1_0440_MR1"
GROUPS = {"cerebral cortex": (2, 21), "cerebral WM": (1, 20), "cerebellar cortex": (6, 25), "cerebellar WM": (5, 24), "brain stem": (13,), "thalamus": (7, 26), "caudate": (8, 27), "putamen": (9, 28), "pallidum": (10, 29), "hippocampus": (14, 30),
          "amygdala": (15, 31), "accumbens": (16, 32), "ventral DC": (17, 33), "lateral ventricles": (3, 22), "inf-lat ventricles": (4, 23), "3rd, 4th ventricles": (11, 12), "vessels": (18, 34), "choroid plexus": (19, 35)}


def main():
    first, again = (json.loads((ROOT/d/"p1.json").read_text()) for d in ("p1", "p1_recheck_from_commit"))
    print("P1 recheck: rows equal:", first["rows"] == again["rows"], "; means equal:", first["means"] == again["means"], "; run from commit:", again["admission"]["code"].get("commit"), "dirty:", again["admission"]["code"].get("dirty"))
    gain = first["paired"]["mix_minus_v2"]["seg35"]["mean"]
    print(f"P1 model: mix - v2 seg35 {gain:+.5f}; through the slope .66-.70: {gain*.66:+.5f} to {gain*.70:+.5f} on the registration seg35 (a model)")
    rows = first["rows"]
    subjects = first["subjects"]
    others = [s for s in subjects if s != TARGET]
    for arm in ("v1", "v2", "mix"):
        dice = np.array([rows[s][arm]["per_label"] for s in subjects], dtype=np.float64)
        t = subjects.index(TARGET)
        gap = np.nanmean(dice[[i for i in range(len(subjects)) if i != t]], axis=0)-dice[t]
        print(f"P4 groups, {arm}: macro gap {gap.sum()/35:.4f}; the mean of the macro of the other 19 {np.mean([rows[s][arm]['seg35'] for s in others]):.4f}, 0440 {rows[TARGET][arm]['seg35']:.4f}")
        if arm == "v1":
            order = sorted(GROUPS, key=lambda name: -sum(gap[l-1] for l in GROUPS[name]))
            for name in order:
                part = sum(gap[l-1] for l in GROUPS[name])
                print(f"  {name:22s} gap per label {part/len(GROUPS[name]):.4f}, contribution to the macro gap {part/35:.4f} ({100*part/gap.sum():.1f}%)")
    seg4 = np.array([rows[s]["v1"]["seg4_per_label"] for s in subjects], dtype=np.float64)
    t = subjects.index(TARGET)
    print("P4 seg4 classes (1 cortex, 2 subcortical, 3 WM, 4 CSF), v1: 0440", np.round(seg4[t], 4).tolist(), "the other 19, mean", np.round(np.delete(seg4, t, axis=0).mean(axis=0), 4).tolist(), "min", np.round(np.delete(seg4, t, axis=0).min(axis=0), 4).tolist())
    leverage = json.loads((ROOT/"p4_followup/leverage.json").read_text())
    r, o = leverage["rows"][TARGET], leverage["others_mean"]
    for metric in ("seg35", "seg4"):
        gap_size = o["base"][metric]-r["base"][metric]
        for key in ("thick", "gaps"):
            own, theirs = r[key][metric]-r["base"][metric], o[key][metric]-o["base"][metric]
            print(f"P4 leverage {metric}: {key}: 0440 {r['base'][metric]:.4f} -> {r[key][metric]:.4f} ({own:+.4f}); the other 19 on average {theirs:+.4f}; the gap {gap_size:.4f}; net share of the gap {100*(own-theirs)/gap_size:.1f}%")
    print(f"P4 leverage voxels: thick {r['n_thick']}, gaps {r['n_gaps']} (cerebellar {r['gaps_cerebellar']}), wrong {r['n_wrong']}; the other 19: gaps {min(leverage['rows'][s]['n_gaps'] for s in others)} to {max(leverage['rows'][s]['n_gaps'] for s in others)}, cerebellar {min(leverage['rows'][s]['gaps_cerebellar'] for s in others)} to {max(leverage['rows'][s]['gaps_cerebellar'] for s in others)}, thick up to {max(leverage['rows'][s]['n_thick'] for s in others)}")
    p3 = json.loads((ROOT/"p3/p3.json").read_text())
    flows = [p["error_flow"] for p in p3["pairs"]]
    print("P3 flow: warped repeats the fixed label at the fixed errors", np.round([f["warped_repeats_fixed_label_at_fixed_errors"] for f in flows], 3).tolist(), "; warped right there", np.round([f["warped_right_at_fixed_errors"] for f in flows], 3).tolist(), "; corrected", [f["corrected"] for f in flows], "; broken", [f["broken"] for f in flows], "; wrong", [f["wrong_fixed"] for f in flows])


if __name__ == "__main__":
    main()
