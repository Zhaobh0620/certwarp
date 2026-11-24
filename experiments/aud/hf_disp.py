"""AUD addition (the lead's request of 2026-11-02 ~5:40 pm CST; runs/20261102-fb-audit/plan.md, revision 1): the
high-frequency part of the existing final displacement fields, u_hf = u - G_sigma * u with sigma = 2 voxels, component by
component, and its RMS inside the fixed image's cortex, white matter and brain.  An evaluation-side number: the truth labels
of the FIXED image are read through bayesreg.eval.truth's guard (development pairs only); no model, no solver, no GPU.

    python -B experiments/aud/hf_disp.py --pair 0438-0439 --out /home/rmcfz/seq_vp_tag_results/fb_audit_v1

Writes hf/<short>/<field>.json, never overwriting (an existing file is skipped).  One field at a time.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_audit  # noqa: E402  (inserts the code root as well)

SIGMA = 2.
FIELDS = ("M2V1", "A0", "A1", "A4p", "P3", "P4")
REGIONS = dict(cortex_seg35=("fixed_seg35", (2, 21)), wm_seg35=("fixed_seg35", (1, 20)), seg4_1=("fixed_seg4", (1,)),
               seg4_3=("fixed_seg4", (3,)))


def hf_stats(u, regions, sigma=SIGMA):
    """RMS of u, of G_sigma * u and of u - G_sigma * u over each boolean region (node arrays; u (nz, ny, nx, 3), mm); the
    Gaussian is scipy's (truncate 4), extended by 0 (u vanishes on the faces and is taken as 0 outside)."""
    low = np.stack([gaussian_filter(u[..., c], sigma=sigma, mode="constant", cval=0., truncate=4.) for c in range(3)], axis=-1)
    high = u-low
    out = {}
    for name, mask in regions.items():
        out[name] = dict(n=int(mask.sum()), rms_total=float(np.sqrt(np.mean(np.sum(u[mask]**2, axis=-1)))),
                         rms_low=float(np.sqrt(np.mean(np.sum(low[mask]**2, axis=-1)))),
                         rms_high=float(np.sqrt(np.mean(np.sum(high[mask]**2, axis=-1)))))
    return out


def main(argv=None):
    from bayesreg.eval.truth import DEFAULT_ALLOWLIST, REPO, admit_pair, load_sealed_lists, load_truth, scheme_for

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", required=True, choices=sorted(fb_audit.PAIRS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fields", default=",".join(FIELDS))
    ns = parser.parse_args(argv)
    out = Path(ns.out)/"hf"/ns.pair
    out.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in ns.fields.split(",") if f and not (out/f"{f}.json").exists()]
    if not wanted:
        return 0
    sealed = load_sealed_lists(REPO)
    pair_dir, grid, _, guard = admit_pair(fb_audit.DATA_ROOT, "stage_L2R_val_v1", fb_audit.PAIRS[ns.pair], DEFAULT_ALLOWLIST, sealed)
    scheme = scheme_for(guard, pair_dir)
    truth, _ = load_truth(pair_dir, scheme, grid, guard, repo=REPO)
    regions ={name: np.isin(truth[key], ids) for name, (key, ids) in REGIONS.items()}
    regions["brain_seg4"] = np.asarray(truth["brain"], dtype=bool)
    del truth
    log = fb_audit.logging.getLogger("aud.hf")
    fb_audit.logging.basicConfig(level=fb_audit.logging.INFO)
    for field in wanted:
        fb_audit.wait_for_memory(fb_audit.FIELD_MEM_GB, log)
        tick = time.perf_counter()
        u, source = fb_audit.load_field(field, ns.pair, grid)
        document = dict(field=field, short=ns.pair, sigma_voxels=SIGMA, spacing=list(grid.spacing), field_record=source,
                        regions=hf_stats(u, regions), label_source="truth labels of the fixed image (bayesreg.eval.truth)",
                        code=fb_audit.code_record(), seconds=time.perf_counter()-tick)
        fb_audit.write_new(out/f"{field}.json", document)
        log.info("%s %s: cortex rms_high %.4f mm, %.1f s", ns.pair, field, document["regions"]["cortex_seg35"]["rms_high"],
                 document["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
