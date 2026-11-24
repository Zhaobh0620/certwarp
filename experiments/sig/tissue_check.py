"""SIG side check: O0 vs O1 (union of the two sides at level start, theta .05, off-support dropped) for the
channel_source = "tissue" channels at u = 0 on L0 (numpy only; at u = 0 the moving spline equals its node values, so
rho = s_M - s_F at the nodes).  Node weights pi_a = q_a m_a/omega0 from the L0 trapezoid weights (mask nodes are
interior here: q_a = omega_l), omega0 = 1 mm^3 (cache native_omega).  Writes <out>/tissue_check.json.

    python -B experiments/sig/tissue_check.py [--out /home/rmcfz/seq_vp_tag_results/sig_v1]
"""
import argparse
import json
from pathlib import Path

import numpy as np

CVG = Path("/home/rmcfz/seq_vp_tag_results/cvg_v1")
PAIRS = ("0438-0439", "0012-0013")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/rmcfz/seq_vp_tag_results/sig_v1")
    args = parser.parse_args(argv)
    target = Path(args.out)/"tissue_check.json"
    if target.exists():
        raise FileExistsError(target)
    doc = {}
    for name in PAIRS:
        cache = np.load(CVG/"cache"/f"{name}.npz")
        meta = json.loads(str(cache["meta_json"]))
        mask = cache["mask_L0"].astype(bool)
        inner = np.zeros_like(mask)
        inner[1:-1, 1:-1, 1:-1] = True
        if np.any(mask & ~inner):
            raise RuntimeError("mask touches the boundary")
        omega_l = float(meta["n_q"]["0"]["n_q"])/float(mask.sum())       # q_a/omega0 on interior mask nodes
        fixed = [cache["fixed_L0"][mask]]+[np.sqrt(p[mask]) for p in cache["tissue_fixed_L0"][1:]]
        moving = [cache["moving_L0"][mask]]+[np.sqrt(p[mask]) for p in cache["tissue_moving_L0"][1:]]
        n_q = omega_l*mask.sum()
        rows = []
        s_i = None
        for c, (f, m) in enumerate(zip(fixed, moving)):
            rho2 = (m-f)**2
            s0 = omega_l*rho2.sum()/(n_q+2)
            if c == 0:
                s_i = s0
                rows.append(dict(channel=c, sigma2_O0=s0))
                continue
            on = (f > np.sqrt(.05)) | (m > np.sqrt(.05))
            s1 = omega_l*rho2[on].sum()/(omega_l*on.sum()+2)
            rows.append(dict(channel=c, sigma2_O0=s0, sigma2_O1=s1, support_fraction=float(on.mean()),
                             ratio_O0=s_i/s0, ratio_O1=s_i/s1))
        doc[name] = dict(class_names=["intensity"]+meta["tissue_fixed"]["record"]["accepted"]["class_names"]
                         if "record" in meta["tissue_fixed"] else None, n_q=n_q, rows=rows)
    target.write_text(json.dumps(doc, indent=1)+"\n", encoding="utf-8")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
