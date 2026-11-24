"""V1 (a): "the certified array is the exported field", read from the LPBA40 second-test-set records (17 pairs).

Reads JSON records only (allowed: LPBA40 results are final; no image, label or array file is opened):
  <R>/eval/run_1/pairs/*.json                      the evaluation records (bayesreg.eval.lpba_eval, code f4615021):
       arms.method.decision_certificate              holdout.decision_certificate (bayesreg/eval/holdout.py:2501-2520)
       arms.method.certificate                       BL1 (tests (i)+(iii) only) on u = h d of the evaluated disp.npz
       arms.method.field.disp.meta                   the export convention written beside d
  <R>/work/method/<pair>/review/review.json        the step-5 review certificate (a third process)
  <R>/work/method/<pair>/output/cert.json          the decision certificate (the repair process)
with <R> = /home/rmcfz/seq_vp_tag_results/lpba40_test_v1.  No OASIS hold-out file is read.
Usage: python -B v1_identity_lpba.py
"""
import glob
import json
import os
import sys

R = "/home/rmcfz/seq_vp_tag_results/lpba40_test_v1"
CONFIG_C = dict(config_class="PrescreenCertificateConfig", delta_hex="0x0.0p+0", max_depth=48, max_boxes=255,
                kernel="numba-parallel-hoisted-edge-hull-v1", fallback_kernel="numba-bernstein-fallback-v1",
                prescreen_kernel="numba-float-bernstein-apriori-v1")


def exact_one(values):
    return all(isinstance(v, float) and v.hex() == (1.).hex() for v in values)


def main():
    files = sorted(glob.glob(os.path.join(R, "eval/run_1/pairs/*.json")))
    print(f"# {len(files)} evaluation records under {R}/eval/run_1/pairs")
    cols = ("pair", "line_status", "h==1.0 (grid, final, disp meta)", "decision cert passed", "unc/inf",
            "stored hash==sha256(u)", "cert.json hash==u", "decision cert hash==u", "status==cert.json", "u==h*d bitwise",
            "disp meta: u_state_hash==u, exact_inverse, layout", "BL1 hash==u & passed", "review hash==u & passed",
            "config C fields", "cells.matches_map", "fast/prescreen/fallback (decision)", "min lower bound")
    print("\t".join(cols))
    totals = dict(rows=0, all_true=0)
    for path in files:
        r = json.load(open(path))
        m = r["arms"]["method"]
        dc = m["decision_certificate"]
        c, fin, bl1 = dc["certificate"], dc["final"], m["certificate"]
        meta = m["field"]["disp"]["meta"]
        pair = r["pair"]
        rv = json.load(open(os.path.join(R, "work/method", pair, "review/review.json")))
        oc = json.load(open(os.path.join(R, "work/method", pair, "output/cert.json")))
        u = fin["u_sha256"]
        checks = [
            exact_one(r["grid"]["spacing_xyz"]) and exact_one(fin["grid"]["spacing_xyz"]) and exact_one(meta["grid"]["spacing_xyz"])
            and r["grid"]["shape_zyx"] == fin["grid"]["shape_zyx"] == meta["grid"]["shape_zyx"]
            and fin["grid"]["grid_id"] == c["grid_id"] == bl1["grid_id"] == meta["grid"]["grid_id"],
            c["passed"] is True and c["status"] == "certified_feasible" and c["min_jd_lower_bound"] > 0,
            c["n_uncertain"] == 0 and c["n_infeasible"] == 0,
            fin["state_hash_matches"] is True and fin["stored_state_hash"] == u,
            dc["state_hash_matches_cert_json"] is True and dc["cert_json"]["state_hash"] == u == oc["state_hash"],
            dc["certificate_state_hash_matches_u"] is True and c["state_hash"] == u,
            dc["status_matches_cert_json"] is True and oc["status"] == c["status"] and oc["passed"] is True,
            dc["u_equals_h_d_bitwise"] is True,
            meta["u_state_hash"] == u and meta["exact_inverse"] is True and meta["layout"] == "(3, nz, ny, nx)"
            and meta["dtype"] == "float64" and m["field"]["disp"]["stored_dtype"] == "float64",
            bl1["state_hash"] == u and bl1["passed"] is True and bl1["certified_field"].startswith("u = h d"),
            rv["state_hash"] == u and rv["review"]["state_hash"] == u and rv["review"]["passed"] is True and rv["consistent"] is True,
            all(c["config"].get(k) == v for k, v in CONFIG_C.items()),
            c["cells"]["matches_map"] is True,
        ]
        totals["rows"] += 1
        totals["all_true"] += all(checks)
        print("\t".join([pair.replace("LPBA40_", ""), m["terminal_status"]] + [str(x) for x in checks]
                        + [f"{c['n_fast_cells']}/{c['n_prescreen_cells']}/{c['n_fallback_cells']}", f"{c['min_jd_lower_bound']:.3e}"]))
    print(f"# rows {totals['rows']}, rows with every check true {totals['all_true']}")
    print("# export convention (disp.npz meta, one record):", meta["convention"])
    print("# processes that certified the same bytes (one record):", oc.get("process"), "|", rv["review"].get("process"),
          "| the evaluation (decision_certificate, BL1)")
    return 0 if totals["all_true"] == totals["rows"] == 17 else 1


if __name__ == "__main__":
    sys.exit(main())
