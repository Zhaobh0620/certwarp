"""MS-D4, package B, amendment 3, item 2: a few counts from minj_by_position.csv (written by posthoc_minj.py); reads nothing else.

    python -I -B posthoc_minj_counts.py      # writes posthoc_minj_counts.out next to this file
"""
import csv
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "minj_by_position.csv")
out = ["input minj_by_position.csv sha256 " + hashlib.sha256(open(src, "rb").read()).hexdigest()]
rows = list(csv.DictReader(open(src)))
for key, n in (("oasis", 50), ("lpba", 17)):
    x = [d for d in rows if d["set"] == key]
    assert len(x) == n
    s = np.array([float(d["sampled_min_J"]) for d in x])
    lb = np.array([float(d["certificate_lower_bound"]) for d in x])
    gs = np.array([float(d["sampled_min_J_gauss27"]) for d in x])
    ce = np.array([float(d["min_J_cell_centre_recorded"]) for d in x])
    assert np.all(lb <= s) and np.all(s <= gs) and np.all(s <= ce) and np.all(lb > 0)
    rank = lambda v: np.argsort(np.argsort(v))
    out.append(f"{key}: n {n}; pairs with the sampled minimum < 1e-5 / 1e-4 / 1e-3 / 1e-2: " + " / ".join(str(int((s < t).sum())) for t in (1e-5, 1e-4, 1e-3, 1e-2))
               + f"; geometric mean of the sampled minimum {np.exp(np.log(s).mean()):.3g}, of the certificate lower bound {np.exp(np.log(lb).mean()):.3g}")
    out.append(f"{key}: pairs with the Gauss27-only minimum < .01 / .02 / .03 / .04 / .05: " + " / ".join(str(int((gs < t).sum())) for t in (.01, .02, .03, .04, .05))
               + "; pairs with the recorded cell-centre minimum < .04 / .05: " + " / ".join(str(int((ce < t).sum())) for t in (.04, .05)))
    out.append(f"{key}: Spearman rank correlation, certificate lower bound against sampled minimum: {np.corrcoef(rank(lb), rank(s))[0, 1]:+.2f} (no test)")
    out.append(f"{key}: cells per pair with a sampled value < .01, as a fraction of all cells: median {np.median([int(d['cells_sampled_min_lt_0.01']) for d in x]) / (6772287 if key == 'oasis' else 6998400):.2e}")
out.append("script sha256 " + hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
with open(os.path.join(HERE, "posthoc_minj_counts.out"), "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
