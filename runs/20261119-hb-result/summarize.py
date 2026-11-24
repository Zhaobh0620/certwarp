#!/usr/bin/env python3
"""B 线留出集 holdout_B_v1 第 12 步结果文档的聚合表（只读；结果文档起草员 2026-11-19）。

第二版（修改轮，2026-11-19；第一次运行 3:05:57 AM CST 的输出照 git 历史 4cfe8326 与 per_position/ 保留）：
加读 HI 记录（只取逐人秒数的聚合与设备）与标签名表；加采样域与六面位移、脑内 log J 分位、分结构 Dice、
PQ 的描述量（只取位置无关的聚合）；逐对文件写到新目录 per_position_run2/，不碰第一次的 per_position/。

只读这些（不读别的任何文件，不解码 npz，不读影像、标签、位移场、真值）：
  1. /home/rmcfz/seq_vp_tag_results/holdout_B_v1/eval/decision.json           （第 12 步判定，sha256 钉在下面）
  2. /home/rmcfz/seq_vp_tag_results/holdout_B_v1/eval/v1/holdout_eval.json    （第 11 步评测总记录，sha256 钉在下面）
  3. /home/rmcfz/seq_vp_tag_results/holdout_B_v1/eval/v1/pairs/*.json          （第 11 步逐对记录；只 glob 这一层，不递归）
  4. /home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens_holdout_B_v1/hi_record.json（第 2 步 HI 记录，sha256 钉在下面；
     只取 rows.*.seconds_total / seconds_forward / seconds_write、environment.gpu 与 backend）
  5. configs/bayesreg/structure_groups_v1.json 的 label_table.names（标签名；sha256 钉在下面）

写出：
  - 本目录：arms_absolute.csv、f1_f2_main_family.csv、f3_described.csv、f4_pq.csv、ablation.csv、per_label_dice.csv、
    pq_described.csv、summarize.out（只有聚合值，不含任何受试者号、对名、对码）；
  - /home/rmcfz/seq_vp_tag_results/holdout_run_v1/step_12/per_position_run2/（不进 git）：逐对数值，只按位置序号 0–49
    （位置 p = holdout_B_pairs.tsv 去掉表头后的第 p+1 行，即 decision.json 的 positions）。

守卫：所有要写的文本先在内存里核一遍——不得出现 OAS 号、4 位-4 位的对码、任何一对的对名或受试者串、
以及这些串里 4 位以上的数字段（前后不是十六进制字符或小数点时）；核不过就整个退出、什么都不写。
逐对目录已存在就拒（不覆盖）。

用法（单线程，nice）：
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
      /home/rmcfz/miniconda3/envs/sci310/bin/python -I -B runs/20261119-hb-result/summarize.py
"""
import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import sys

import numpy as np

H = "/home/rmcfz/seq_vp_tag_results/holdout_B_v1"
DECISION = H + "/eval/decision.json"
EVAL = H + "/eval/v1/holdout_eval.json"
PAIRS_DIR = H + "/eval/v1/pairs"
HI_RECORD = "/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens_holdout_B_v1/hi_record.json"
LABELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "configs", "bayesreg", "structure_groups_v1.json")
EXPECTED_SHA = {
    DECISION: "a0dbdcf9fc03fac7b985bf0a77cd93ab34fe9fa0845039ad0a3bdda445c26eef",
    EVAL: "89a7f1dc15bce6f688f753f11c946edcb1a4ddc7a9986ab290f226d5e3cea0cf",
    HI_RECORD: "778cdba813003396a7a561caff0c7050f40744fe1b3f251f27d5ff6ede5b1bb6",
    LABELS: "20edfa374e2a9c8ad897a7ad658fc379e4123f83e6c987513b5bf745fe572c19",
}
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PER_POS_DIR = "/home/rmcfz/seq_vp_tag_results/holdout_run_v1/step_12/per_position_run2"
MAX_BYTES = 10 * 1024 * 1024

METRIC_KEYS = ("seg35", "seg4", "dice_official", "hd95", "sdlogj_official", "sdlogj_brain", "pct_J_lt_half_brain",
               "frac_abs_logj_gt_log2_brain", "logj_q01_brain", "logj_q99_brain", "min_J_centre")
COUNT_KEYS = ("cell_centre_J_nonpositive", "central_det_nonpositive", "central_det_nonpositive_brain")
F4_BASELINES = ("h", "H_F", "H2", "V_F", "V2", "G_plus", "X_lex", "B_d")
PQ_SCORES = ("q",) + F4_BASELINES + ("G_minus",)
PQ_MASKS = ("omega_e", "band", "whole")

# ---- guard ----------------------------------------------------------------------------------------------------------
GENERIC = [re.compile(r"OAS[0-9]"), re.compile(r"(?<![0-9A-Za-z])[0-9]{4}-[0-9]{4}(?![0-9A-Za-z])"),
           re.compile(r"[0-9]{4}_[0-9]{4}")]


class Guard:
    def __init__(self):
        self.strings = set()
        self.digit_runs = set()

    def add(self, s):
        if not isinstance(s, str) or not s:
            return
        self.strings.add(s)
        for run in re.findall(r"[0-9]{4,}", s):
            self.digit_runs.add(run)

    def problems(self, text):
        bad = 0
        for r in GENERIC:
            if r.search(text):
                bad += 1
        for s in self.strings:
            if s in text:
                bad += 1
        for run in self.digit_runs:
            if re.search(r"(?<![0-9A-Fa-f.])" + run + r"(?![0-9A-Fa-f])", text):
                bad += 1
        return bad


def die(msg, code=3):
    sys.stderr.write("summarize.py: " + msg + "\n")
    sys.exit(code)


# ---- reading --------------------------------------------------------------------------------------------------------
def read_json(path, expected=None):
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        die(f"refusing a file over 10 MB ({size} B)")
    with open(path, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        die(f"sha256 mismatch on a pinned input (got {digest[:8]}, want {expected[:8]})")
    return json.loads(raw.decode("utf-8")), digest, size


def num(v):
    if v is None:
        return math.nan
    if isinstance(v, bool):
        return float(v)
    return float(v)


def fmt(v, nd=4):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def fmt_sig(v, nd=3):
    if v is None:
        return "None"
    if isinstance(v, float) and not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}g}"


def dsc30(rows):
    """scripts/evaluate_l2r_style.aggregate 的 dsc30_ranking（bayesreg.eval.metrics.aggregate:134-139 的复述）：
    各对 (initial, final) 逐标签条目按位置序拼接，去掉含 NaN 的条目，按 initial 稳定排序，取最低 round(.3 n)（至少 1）条的 final 均值。"""
    entries = []
    for r in rows:
        for r0, r1 in zip(r["initial_dice_labels"], r["dice_labels"]):
            a, b = num(r0), num(r1)
            if not (math.isnan(a) or math.isnan(b)):
                entries.append((a, b))
    entries.sort(key=lambda e: e[0])
    k = max(1, int(round(.3 * len(entries))))
    return float(np.mean([e[1] for e in entries[:k]])), len(entries), k


def main():
    out = []      # summarize.out lines

    def p(line=""):
        out.append(line)

    decision, d_sha, d_size = read_json(DECISION, EXPECTED_SHA[DECISION])
    ev, e_sha, e_size = read_json(EVAL, EXPECTED_SHA[EVAL])

    guard = Guard()
    positions = decision["positions"]
    pos_of = {}
    for i, row in enumerate(positions):
        if row["position"] != i or row["index"] != i:
            die("decision positions are not 0..n-1 with index == position")
        guard.add(row["pair"])
        pos_of[row["pair"]] = i
    n_pos = len(positions)

    arm_table = ev["arm_table"]["arms"]
    arms = [a["name"] for a in arm_table]
    family = {a["name"]: a.get("family") for a in arm_table}
    main_family = list(ev["arm_table"]["main_family"])
    method = [a for a in arms if family[a] == "method"]
    if len(method) != 1:
        die("arm table: not exactly one method arm")
    method = method[0]
    for s in ev.get("evaluated", []):
        guard.add(s)
    hi, hi_sha, hi_size = read_json(HI_RECORD, EXPECTED_SHA[HI_RECORD])
    for s in hi.get("subjects", []):
        guard.add(s)
    hi_rows = hi["rows"]
    for k, r in hi_rows.items():
        guard.add(k)
        guard.add(r.get("subject"))
    hi_sec = {f: np.array([float(r[f]) for r in hi_rows.values()]) for f in ("seconds_total", "seconds_forward", "seconds_write")}
    hi_gpu = (hi.get("environment") or {}).get("gpu")
    hi_device = (hi.get("environment") or {}).get("device")
    hi_backend = hi.get("backend")
    labels_doc, l_sha, _ = read_json(LABELS, EXPECTED_SHA[LABELS])
    label_names = labels_doc["label_table"]["names"]

    files = sorted(glob.glob(os.path.join(PAIRS_DIR, "*.json")))
    pair_files = [f for f in files if os.path.dirname(f) == PAIRS_DIR]
    records = [None] * n_pos
    file_sha = [None] * n_pos
    for f in pair_files:
        rec, sha, size = read_json(f)
        guard.add(rec.get("pair"))
        guard.add(rec.get("fixed"))
        guard.add(rec.get("moving"))
        guard.add(os.path.basename(f))
        guard.add(os.path.splitext(os.path.basename(f))[0])
        pos = pos_of.get(rec.get("pair"))
        if pos is None:
            die("a pair record is not in the decision's positions")
        if records[pos] is not None:
            die("two pair records at one position")
        if rec.get("index") != pos:
            die("pair record index differs from its decision position")
        records[pos] = rec
        file_sha[pos] = sha
    if any(r is None for r in records):
        die("some positions have no pair record")

    # ---- per arm, per position ----------------------------------------------------------------------------------------
    checks = dict(complete=0, arms16=0, all_output=0, error_keys=0, seg4_equals_G=0, pq_complete=0, pq_ev1=0, pq_ev2=0,
                  ablation_complete=0, in_ablation=0)
    val = {a: {k: np.full(n_pos, np.nan) for k in METRIC_KEYS} for a in arms + ["identity"]}
    cnt = {a: {k: np.zeros(n_pos, dtype=np.int64) for k in COUNT_KEYS} for a in arms + ["identity"]}
    cert = {a: dict(passed=np.zeros(n_pos, bool), status={}, n_infeasible=np.zeros(n_pos, np.int64),
                    n_uncertain=np.zeros(n_pos, np.int64), lb=np.full(n_pos, np.nan), inf_brain=np.zeros(n_pos, np.int64),
                    inf_bg=np.zeros(n_pos, np.int64), unc_brain=np.zeros(n_pos, np.int64), unc_bg=np.zeros(n_pos, np.int64),
                    domain_ok=np.zeros(n_pos, bool), configuration={}, mode={}, cells_match=np.zeros(n_pos, bool),
                    faces=np.full(n_pos, np.nan), domain_count=np.zeros(n_pos, np.int64)) for a in arms}
    n_nodes = np.zeros(n_pos, np.int64)
    lab = {sc: {a: {} for a in ["identity"] + arms} for sc in ("seg35", "seg4")}
    pq_s = {mk: {sc: {f: np.full(n_pos, np.nan) for f in ("ause", "auroc", "largest_tie_share")} for sc in PQ_SCORES} for mk in PQ_MASKS}
    pq_q4 = {f: np.full(n_pos, np.nan) for f in ("ause", "auroc")}
    pq_err = {mk: np.full(n_pos, np.nan) for mk in PQ_MASKS}
    pq_err4 = np.full(n_pos, np.nan)
    pq_ece = {b: np.full(n_pos, np.nan) for b in ("width", "decile")}
    pq_pool = {}
    dec_cert = dict(passed=np.zeros(n_pos, bool), n_infeasible=np.zeros(n_pos, np.int64), n_uncertain=np.zeros(n_pos, np.int64),
                    lb=np.full(n_pos, np.nan), mode={}, config_class={})
    terminal = {a: {} for a in arms}
    official_rows = {a: [None] * n_pos for a in arms + ["identity"]}
    export_flag = {a: 0 for a in arms}
    for i, rec in enumerate(records):
        checks["complete"] += bool(rec.get("complete"))
        checks["arms16"] += set(rec["arms"]) == set(arms)
        checks["all_output"] += all(rec["arms"][a].get("outcome") == "output" for a in arms)
        checks["error_keys"] += ("error" in rec) or any("error" in rec["arms"][a] for a in arms)
        g = rec.get("seg4_equals_G") or {}
        checks["seg4_equals_G"] += bool(g.get("fixed", {}).get("equal")) and bool(g.get("moving", {}).get("equal"))
        pq = rec.get("pq") or {}
        checks["pq_complete"] += bool(pq.get("complete"))
        checks["pq_ev1"] += bool((pq.get("ev1") or {}).get("passed"))
        checks["pq_ev2"] += bool((pq.get("ev2") or {}).get("passed"))
        ab = rec.get("ablation") or {}
        checks["ablation_complete"] += bool(ab.get("complete"))
        checks["in_ablation"] += bool(ab.get("in_ablation"))
        n_nodes[i] = int(np.prod(rec["grid"]["shape_zyx"]))
        acc = pq.get("acceptance") or {}
        for mk in PQ_MASKS:
            pq_err[mk][i] = acc["n_errors"][mk] / acc["masks"][mk]
            for sc in PQ_SCORES:
                x = acc["scores"][mk][sc]
                for f in ("ause", "auroc", "largest_tie_share"):
                    pq_s[mk][sc][f][i] = num(x.get(f))
        q4 = acc["seg4_scale"]["q4"]
        pq_q4["ause"][i], pq_q4["auroc"][i] = num(q4.get("ause")), num(q4.get("auroc"))
        pq_err4[i] = acc["n_errors_seg4"] / acc["masks"]["omega_e"]
        for b in ("width", "decile"):
            rel = acc["reliability"][b]
            pq_ece[b][i] = num(rel.get("ece"))
            pool = pq_pool.setdefault(b, dict(count=np.zeros(len(rel["count"])), sum_q=np.zeros(len(rel["count"])),
                                              errors=np.zeros(len(rel["count"]))))
            for f in ("count", "sum_q", "errors"):
                pool[f] += np.array([num(v) for v in rel[f]])
        for a in arms + ["identity"]:
            entry = rec["identity"] if a == "identity" else rec["arms"][a]
            m = entry["metrics"]
            for k in METRIC_KEYS:
                val[a][k][i] = num(m.get(k))
            for k in COUNT_KEYS:
                cnt[a][k][i] = int(m.get(k) or 0)
            official_rows[a][i] = entry["official"]
            for sc in ("seg35", "seg4"):
                for lbl, v in (entry["per_label"].get(sc + "_per_label") or {}).items():
                    lab[sc][a].setdefault(lbl, []).append(num(v))
            if a == "identity":
                continue
            ts = entry.get("terminal_status")
            terminal[a][ts] = terminal[a].get(ts, 0) + 1
            if entry.get("export_suspicious"):
                export_flag[a] += 1
            c = entry["certificate"]
            cr = cert[a]
            cr["passed"][i] = bool(c.get("passed"))
            cr["status"][c.get("status")] = cr["status"].get(c.get("status"), 0) + 1
            cr["n_infeasible"][i] = int(c.get("n_infeasible") or 0)
            cr["n_uncertain"][i] = int(c.get("n_uncertain") or 0)
            cr["lb"][i] = num(c.get("min_jd_lower_bound"))
            cr["configuration"][c.get("configuration")] = cr["configuration"].get(c.get("configuration"), 0) + 1
            cr["mode"][c.get("mode")] = cr["mode"].get(c.get("mode"), 0) + 1
            cr["domain_ok"][i] = bool((c.get("domain") or {}).get("ok"))
            cells = c.get("cells") or {}
            regions = cells.get("regions") or {}
            cr["inf_brain"][i] = int((regions.get("brain") or {}).get("infeasible") or 0)
            cr["inf_bg"][i] = int((regions.get("background") or {}).get("infeasible") or 0)
            cr["unc_brain"][i] = int((regions.get("brain") or {}).get("uncertain") or 0)
            cr["unc_bg"][i] = int((regions.get("background") or {}).get("uncertain") or 0)
            cr["cells_match"][i] = bool(cells.get("matches_map"))
            cr["faces"][i] = num(c.get("faces_max_abs_u_mm"))
            cr["domain_count"][i] = int(((c.get("domain") or {}).get("count")) or 0)
            if a == method:
                dc = (entry.get("decision_certificate") or {}).get("certificate") or {}
                dec_cert["passed"][i] = bool(dc.get("passed"))
                dec_cert["n_infeasible"][i] = int(dc.get("n_infeasible") or 0)
                dec_cert["n_uncertain"][i] = int(dc.get("n_uncertain") or 0)
                dec_cert["lb"][i] = num(dc.get("min_jd_lower_bound"))
                mo = dc.get("mode")
                dec_cert["mode"][mo] = dec_cert["mode"].get(mo, 0) + 1
                cc = (dc.get("config") or {}).get("config_class")
                dec_cert["config_class"][cc] = dec_cert["config_class"].get(cc, 0) + 1

    # ---- absolute table -----------------------------------------------------------------------------------------------
    dec_dsc30 = decision["F3"]["dsc30"]
    abs_rows = []
    for a in ["identity"] + arms:
        r = dict(arm=a, family=("reference" if a == "identity" else family[a]), n=n_pos)
        for k in METRIC_KEYS:
            v = val[a][k]
            r[k + "_mean"] = float(np.nanmean(v))
            r[k + "_nonfinite"] = int(np.sum(~np.isfinite(v)))
        gap = np.abs(val[a]["seg35"] - val[a]["dice_official"])
        r["pairs_seg35_ne_dice_official"] = int(np.sum(gap > 1e-12))
        r["max_abs_seg35_minus_dice_official"] = float(np.nanmax(gap))
        r["seg35_sd"] = float(np.nanstd(val[a]["seg35"]))
        r["seg4_sd"] = float(np.nanstd(val[a]["seg4"]))
        r["min_J_centre_min"] = float(np.nanmin(val[a]["min_J_centre"]))
        rows_a = official_rows[a]
        dsc, n_entries, k30 = dsc30(rows_a)
        r["dsc30"] = dsc
        r["dsc30_entries"] = n_entries
        r["dsc30_k"] = k30
        offd = np.array([num(x["dice"]) for x in rows_a])
        r["official_dice_mean"] = float(np.mean(offd))
        r["official_dice_std_ddof0"] = float(np.std(offd))
        r["official_dice30_quantile"] = float(np.quantile(offd, .3))
        hd = np.array([num(x["hd95"]) for x in rows_a])
        r["official_hd95_30_quantile"] = float(np.quantile(hd, .3))
        sd = np.array([num(x["sdlogj"]) for x in rows_a])
        r["official_sdlogj_70_quantile"] = float(np.quantile(sd, .7))
        if a != "identity":
            ref = next(iter(dec_dsc30.values()))["method"] if a == method else dec_dsc30[a]["arm"]
            r["dsc30_decision"] = float(ref)
            r["dsc30_abs_diff_vs_decision"] = abs(float(ref) - dsc)
        for k in COUNT_KEYS:
            r[k + "_sum"] = int(cnt[a][k].sum())
            r[k + "_pairs_gt0"] = int((cnt[a][k] > 0).sum())
        r["pct_frac_abs_logj_gt_log2_brain_mean"] = 100. * r["frac_abs_logj_gt_log2_brain_mean"]
        if a != "identity":
            cr = cert[a]
            dj = decision["certificates"][a]["jacobian_only_bl1"]
            r["bl1_passed"] = int(cr["passed"].sum())
            r["bl1_passed_decision"] = int(dj["passed"])
            r["bl1_status"] = ";".join(f"{k}:{v}" for k, v in sorted(cr["status"].items(), key=lambda kv: str(kv[0])))
            r["bl1_n_infeasible_sum"] = int(cr["n_infeasible"].sum())
            r["bl1_n_infeasible_decision"] = int(dj["n_infeasible"])
            r["bl1_n_uncertain_sum"] = int(cr["n_uncertain"].sum())
            r["bl1_n_uncertain_decision"] = int(dj["n_uncertain"])
            r["bl1_pairs_with_infeasible"] = int((cr["n_infeasible"] > 0).sum())
            r["bl1_infeasible_brain_sum"] = int(cr["inf_brain"].sum())
            r["bl1_infeasible_background_sum"] = int(cr["inf_bg"].sum())
            r["bl1_uncertain_cells_sum"] = int(cr["unc_brain"].sum() + cr["unc_bg"].sum())
            r["bl1_min_jd_lower_bound"] = float(np.nanmin(cr["lb"]))
            r["bl1_min_jd_lower_bound_decision"] = float(dj["min_jd_lower_bound"])
            r["bl1_cells_matches_map"] = int(cr["cells_match"].sum())
            r["bl1_domain_ok"] = int(cr["domain_ok"].sum())
            r["faces_max_abs_u_mm_min"] = float(np.nanmin(cr["faces"]))
            r["faces_max_abs_u_mm_median"] = float(np.nanmedian(cr["faces"]))
            r["faces_max_abs_u_mm_max"] = float(np.nanmax(cr["faces"]))
            r["domain_invalid_nodes_min"] = int(cr["domain_count"].min())
            r["domain_invalid_nodes_max"] = int(cr["domain_count"].max())
            frac = cr["domain_count"] / n_nodes
            r["domain_invalid_fraction_min"] = float(frac.min())
            r["domain_invalid_fraction_max"] = float(frac.max())
            r["pairs_with_invalid_domain_nodes"] = int((cr["domain_count"] > 0).sum())
            r["bl1_configuration"] = ";".join(f"{k}:{v}" for k, v in cr["configuration"].items())
            r["bl1_mode"] = ";".join(f"{k}:{v}" for k, v in cr["mode"].items())
            r["terminal_status"] = ";".join(f"{k}:{v}" for k, v in terminal[a].items())
            r["export_suspicious"] = export_flag[a]
            if a == method:
                dd = decision["certificates"][method]["decision_configuration"]
                r["decision_cert_passed"] = int(dec_cert["passed"].sum())
                r["decision_cert_passed_decision"] = int(dd["passed"])
                r["decision_cert_n_infeasible_sum"] = int(dec_cert["n_infeasible"].sum())
                r["decision_cert_n_uncertain_sum"] = int(dec_cert["n_uncertain"].sum())
                r["decision_cert_min_lb"] = float(np.nanmin(dec_cert["lb"]))
                r["decision_cert_mode"] = ";".join(f"{k}:{v}" for k, v in dec_cert["mode"].items())
                r["decision_cert_config_class"] = ";".join(f"{k}:{v}" for k, v in dec_cert["config_class"].items())
        abs_rows.append(r)

    # ---- method - arm, recomputed from the pair records (cross-check) -------------------------------------------------
    diff_rows = []
    for a in arms:
        if a == method:
            continue
        r = dict(arm=a, family=family[a])
        for k in ("seg35", "seg4"):
            dlt = val[method][k] - val[a][k]
            r[k + "_mean_diff"] = float(np.mean(dlt))
            r[k + "_median_diff"] = float(np.median(dlt))
            r[k + "_wins"] = int((dlt > 0).sum())
            r[k + "_ties"] = int((dlt == 0).sum())
        diff_rows.append(r)

    # ---- F1 / F2 ------------------------------------------------------------------------------------------------------
    fam_rows = []
    for fam_key, endpoint in (("F1", "seg35"), ("F2", "seg4")):
        F = decision[fam_key]
        for a in F["order"]:
            x = F["arms"][a]
            g = x["sdlogj_guard"]
            holm = x["holm"]
            s2 = (decision.get("sensitivity") or {}).get("S2", {}).get(a) if fam_key == "F1" else None
            fam_rows.append(dict(
                family_test=fam_key, endpoint=endpoint, arm=a, n=x["n"], mean_diff=x["mean"], median_diff=x["median"],
                ci95_lo=x["interval"][0], ci95_hi=x["interval"][1], wins=x["wins"], wilcoxon_p=x["wilcoxon_p"],
                p_sup=x["p"]["sup"], p_eq=x["p"]["eq"], p_inf=x["p"]["inf"], draws=x["p"]["draws"], void_draws=x["p"]["void"],
                holm_sup_step=holm["sup"]["step"], holm_sup_adjusted=holm["sup"]["adjusted"], holm_sup_threshold=holm["sup"]["threshold"],
                holm_sup_rejected=holm["sup"]["rejected"], holm_eq_rejected=holm["eq"]["rejected"], holm_inf_rejected=holm["inf"]["rejected"],
                rejected_classes=";".join(x["rejected_classes"]), verdict=x["verdict"], not_judged=x["not_judged"],
                n_suspicious=x["n_suspicious"], imputed_method=x["imputed_method"], imputed_arm=x["imputed_arm"],
                s1_n=(x.get("s1") or {}).get("n"), s1_lo=((x.get("s1") or {}).get("interval") or [None, None])[0],
                s1_hi=((x.get("s1") or {}).get("interval") or [None, None])[1],
                guard_bound=g["bound"], guard_n=g["n"], guard_mean=g["mean"], guard_lo=g["interval"][0], guard_hi=g["interval"][1],
                guard_upper=g["upper"], guard_holds=g["holds"],
                s2_n=None if s2 is None else s2["n"], s2_lo=None if s2 is None else s2["interval"][0],
                s2_hi=None if s2 is None else s2["interval"][1]))

    # ---- F3 -------------------------------------------------------------------------------------------------------------
    f3_rows = []
    F3 = decision["F3"]
    for ep, per_arm in F3["endpoints"].items():
        for a, x in per_arm.items():
            f3_rows.append(dict(block="endpoints", endpoint=ep, arm=a, family=x["family"], mean_diff=x["mean"],
                                ci95_lo=x["interval"][0], ci95_hi=x["interval"][1], written=x["written"],
                                n_nonfinite=x["n_nonfinite"], imputed_arm=x["imputed_arm"], imputed_method=x["imputed_method"]))
    for ep, per_arm in F3["described_arms"].items():
        for a, x in per_arm.items():
            f3_rows.append(dict(block="described_arms", endpoint=ep, arm=a, family=x["family"], mean_diff=x["mean"],
                                ci95_lo=x["interval"][0], ci95_hi=x["interval"][1], written=x["written"],
                                n_nonfinite=x["n_nonfinite"], imputed_arm=x["imputed_arm"], imputed_method=x["imputed_method"]))
    for a, x in F3["dsc30"].items():
        f3_rows.append(dict(block="dsc30", endpoint="dsc30", arm=a, family=x["family"], mean_diff=x["method"] - x["arm"],
                            ci95_lo=x["interval"][0], ci95_hi=x["interval"][1], written=x["written"],
                            n_nonfinite=None, imputed_arm=None, imputed_method=None))
    sd010 = F3["sdlogj_official_mean_le_010"]

    # ---- F4 -------------------------------------------------------------------------------------------------------------
    F4 = decision["F4"]
    f4_rows = []
    for b in F4_BASELINES:
        x = F4["baselines"][b]
        per = np.array([num(e["delta"]) for e in x["per_pair"]])
        hx = F4["hx_certified_subset"][b]
        f4_rows.append(dict(baseline=b, mean_delta=x["mean_delta"], LB=x["LB"], R=x["R"], ci95_lo=x["interval"][0],
                            ci95_hi=x["interval"][1], LB_gap=x["LB_gap"], R_gap=x["R_gap"], valid_draws=x["valid_draws"],
                            void_draws=x["void_draws"], passed=x["passed"], pairs_delta_gt0=int((per > 0).sum()),
                            n_pairs=len(per), hx_subset_n=hx["n"], hx_subset_LB=hx["LB"], hx_subset_R=hx["R"]))

    # ---- ablation -----------------------------------------------------------------------------------------------------
    A = decision["ablation"]
    ab_rows = []
    for tname, t in A["tables"].items():
        for ep, x in t["endpoints"].items():
            diffs = np.array([num(e["difference"]) for e in x["per_pair"]]) if x["per_pair"] else np.array([])
            ab_rows.append(dict(table=tname, line_statuses=";".join(t["line_statuses"]), n_pairs=t["n_pairs"], endpoint=ep,
                                n=x["n"], mean_before=x["mean_before"], mean_after=x["mean_after"], mean_diff_after_minus_before=x["mean"],
                                median_diff=x["median"], ci95_lo=None if x["interval"] is None else x["interval"][0],
                                ci95_hi=None if x["interval"] is None else x["interval"][1], void_draws=x["void_draws"],
                                pairs_after_gt_before=int((diffs > 0).sum()), pairs_after_lt_before=int((diffs < 0).sum())))

    # ---- per-label Dice (position-free aggregates) ---------------------------------------------------------------------
    label_rows = []
    for sc in ("seg35", "seg4"):
        ids = sorted({int(k) for a in lab[sc] for k in lab[sc][a]})
        for lid in ids:
            for a in ["identity"] + arms:
                v = np.array(lab[sc][a].get(str(lid), []), dtype=float)
                label_rows.append(dict(scale=sc, label=lid, name=(label_names.get(str(lid), "") if sc == "seg35" else ""), arm=a,
                                       mean=float(np.mean(v)) if v.size else None, n_pairs_present=int(v.size),
                                       n_pairs_zero=int((v == 0).sum()), min=float(v.min()) if v.size else None))

    # ---- PQ described quantities (C22.H7-5; position-free aggregates) ------------------------------------------------
    pq_rows = []
    for mk in PQ_MASKS:
        for sc in PQ_SCORES:
            x = pq_s[mk][sc]
            pq_rows.append(dict(block="score", mask=mk, score=sc, ause_mean=float(np.mean(x["ause"])), ause_median=float(np.median(x["ause"])),
                                auroc_mean=float(np.mean(x["auroc"])), auroc_median=float(np.median(x["auroc"])),
                                auroc_min=float(np.min(x["auroc"])), auroc_max=float(np.max(x["auroc"])),
                                largest_tie_share_max=float(np.max(x["largest_tie_share"])),
                                pairs_q_better=(None if sc == "q" else int((x["ause"] > pq_s[mk]["q"]["ause"]).sum()))))
    pq_rows.append(dict(block="score", mask="omega_e_seg4", score="q4", ause_mean=float(np.mean(pq_q4["ause"])),
                        ause_median=float(np.median(pq_q4["ause"])), auroc_mean=float(np.mean(pq_q4["auroc"])),
                        auroc_median=float(np.median(pq_q4["auroc"])), auroc_min=float(np.min(pq_q4["auroc"])),
                        auroc_max=float(np.max(pq_q4["auroc"]))))
    for mk in PQ_MASKS:
        pq_rows.append(dict(block="error_rate", mask=mk, score="e", ause_mean=None, rate_median=float(np.median(pq_err[mk])),
                            rate_min=float(np.min(pq_err[mk])), rate_max=float(np.max(pq_err[mk]))))
    pq_rows.append(dict(block="error_rate", mask="omega_e_seg4", score="e4", rate_median=float(np.median(pq_err4)),
                        rate_min=float(np.min(pq_err4)), rate_max=float(np.max(pq_err4))))
    for b in ("width", "decile"):
        pq_rows.append(dict(block="ece", mask="omega_e", score=b, ece_median=float(np.median(pq_ece[b])),
                            ece_min=float(np.min(pq_ece[b])), ece_max=float(np.max(pq_ece[b]))))
        pool = pq_pool[b]
        for j in range(len(pool["count"])):
            c = pool["count"][j]
            pq_rows.append(dict(block="reliability_pooled", mask="omega_e", score=b, bin=j, count=int(c),
                                mean_q=(float(pool["sum_q"][j] / c) if c else None), error_rate=(float(pool["errors"][j] / c) if c else None)))
    for bname in F4_BASELINES:
        au = pq_s["omega_e"][bname]["ause"]
        dl = au - pq_s["omega_e"]["q"]["ause"]
        dec = np.array([num(e["delta"]) for e in F4["baselines"][bname]["per_pair"]])
        rk = dl / au
        pq_rows.append(dict(block="per_pair_R", mask="omega_e", score=bname, R_pair_median=float(np.median(rk)), R_pair_min=float(np.min(rk)),
                            R_pair_max=float(np.max(rk)), pairs_R_ge_05=int((rk >= .05).sum()),
                            delta_matches_decision=bool(np.allclose(np.sort(dl), np.sort(dec), rtol=0, atol=1e-12))))

    # ---- per position (not in git) ------------------------------------------------------------------------------------
    pp_metrics = io.StringIO()
    w = csv.writer(pp_metrics, lineterminator="\n")
    w.writerow(["position", "pairs_table_row", "arm"] + list(METRIC_KEYS) + list(COUNT_KEYS)
               + ["bl1_passed", "bl1_n_infeasible", "bl1_n_uncertain", "bl1_min_jd_lower_bound", "bl1_infeasible_brain",
                  "bl1_infeasible_background", "bl1_domain_ok", "decision_cert_passed", "faces_max_abs_u_mm",
                  "domain_invalid_nodes", "n_nodes"])
    for i in range(n_pos):
        for a in ["identity"] + arms:
            row = [i, i + 1, a] + [repr(float(val[a][k][i])) for k in METRIC_KEYS] + [int(cnt[a][k][i]) for k in COUNT_KEYS]
            if a == "identity":
                row += ["", "", "", "", "", "", "", "", "", "", int(n_nodes[i])]
            else:
                cr = cert[a]
                row += [int(cr["passed"][i]), int(cr["n_infeasible"][i]), int(cr["n_uncertain"][i]), repr(float(cr["lb"][i])),
                        int(cr["inf_brain"][i]), int(cr["inf_bg"][i]), int(cr["domain_ok"][i]),
                        int(dec_cert["passed"][i]) if a == method else "", repr(float(cr["faces"][i])), int(cr["domain_count"][i]),
                        int(n_nodes[i])]
            w.writerow(row)
    pp_f4 = io.StringIO()
    w = csv.writer(pp_f4, lineterminator="\n")
    w.writerow(["position", "pairs_table_row", "baseline", "delta_AUSE_k_minus_AUSE_q"])
    for b in F4_BASELINES:
        for e in F4["baselines"][b]["per_pair"]:
            w.writerow([e["position"], e["position"] + 1, b, repr(float(e["delta"]))])
    pp_ab = io.StringIO()
    w = csv.writer(pp_ab, lineterminator="\n")
    w.writerow(["position", "pairs_table_row", "table", "endpoint", "before", "after", "after_minus_before"])
    for tname, t in A["tables"].items():
        order = [pos_of[s] for s in t["pairs"]]
        for ep, x in t["endpoints"].items():
            if len(x["per_pair"]) != len(order):
                die("ablation per_pair length differs from its pair list")
            for pos, e in zip(order, x["per_pair"]):
                w.writerow([pos, pos + 1, tname, ep, repr(float(e["before"])), repr(float(e["after"])), repr(float(e["difference"]))])
    pp_pq = io.StringIO()
    w = csv.writer(pp_pq, lineterminator="\n")
    w.writerow(["position", "pairs_table_row", "mask", "score", "ause", "auroc", "largest_tie_share"])
    for i in range(n_pos):
        for mk in PQ_MASKS:
            for sc in PQ_SCORES:
                x = pq_s[mk][sc]
                w.writerow([i, i + 1, mk, sc, repr(float(x["ause"][i])), repr(float(x["auroc"][i])), repr(float(x["largest_tie_share"][i]))])
        w.writerow([i, i + 1, "omega_e_seg4", "q4", repr(float(pq_q4["ause"][i])), repr(float(pq_q4["auroc"][i])), ""])
    pp_texts = {"metrics_by_position.csv": pp_metrics.getvalue(), "f4_delta_by_position.csv": pp_f4.getvalue(),
                "ablation_by_position.csv": pp_ab.getvalue(), "pq_by_position.csv": pp_pq.getvalue()}

    # ---- CSV texts ----------------------------------------------------------------------------------------------------
    def to_csv(rows):
        buf = io.StringIO()
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
        return buf.getvalue()

    csv_texts = {
        "arms_absolute.csv": to_csv(abs_rows),
        "f1_f2_main_family.csv": to_csv(fam_rows),
        "f3_described.csv": to_csv(f3_rows + [dict(block="method_minus_arm_from_pairs", **r) for r in diff_rows]),
        "f4_pq.csv": to_csv(f4_rows),
        "ablation.csv": to_csv(ab_rows),
        "per_label_dice.csv": to_csv(label_rows),
        "pq_described.csv": to_csv(pq_rows),
    }

    # ---- summarize.out --------------------------------------------------------------------------------------------------
    p("# summarize.py 读数（B 线留出集 holdout_B_v1，第 12 步；留出集单次，每对每臂一次）")
    p(f"decision.json sha256 {d_sha} ({d_size} B)")
    p(f"holdout_eval.json sha256 {e_sha} ({e_size} B)")
    h = hashlib.sha256("".join(f"{i} {s}\n" for i, s in enumerate(file_sha)).encode()).hexdigest()
    p(f"pairs/*.json: {len(pair_files)} files (glob one level), each <= 10 MB; sha256 of the position-ordered list "
      f"'<position> <sha256>\\n' = {h}")
    p(f"decision: reading {decision['reading']}, n_used {decision['n_used']}, missing_pairs {len(decision['missing_pairs'])}, "
      f"seed {decision['seed']}, B {decision['B']}, sdlogj_guard {decision['sdlogj_guard']}, seg4_rule {decision['seg4_rule']}, "
      f"missing_rule {decision['missing_rule']}, merge_conflicts {len(decision['merge_conflicts'])}, idx_sha256 {decision['idx_sha256'][:16]}")
    ro = decision["rule_one"]
    p(f"decision rule_one: passed {ro['passed']}, head {ro['head'][:8]}, pinned {ro['pinned'][:8]}, pinned_is_ancestor {ro['pinned_is_ancestor']}, "
      f"tracked_clean {ro['tracked_clean']}, problems {len(ro['problems'])}")
    p(f"decision prereg sha256 {decision['prereg']['sha256'][:16]}, arm_table sha256 {decision['arm_table']['sha256'][:16]}")
    o = ev["opened_files"]
    p(f"holdout_eval: evaluated {len(ev['evaluated'])}, failed {len(ev['failed'])}, stopped {ev['stopped']}, not_reached {len(ev['not_reached'])}, "
      f"opened events {len(o['events'])}, refused {len(o['refused'])}, sealed_list_reads {o['sealed_list_reads']}, watched_dirs {len(o['watched_dirs'])}, "
      f"ended_utc {ev['ended_utc']}")
    p(f"pair records (of {n_pos}): " + ", ".join(f"{k} {v}" for k, v in checks.items()))
    p(f"method_line_status {decision['method_line_status']}; no_output: " +
      ", ".join(f"{a} {decision['no_output'][a]['n']}" for a in arms))
    p()
    p("## 绝对值（50 对的均值；Dice 越高越好；HD95 体素 = mm；SDlogJ、J 的量照 bayesreg/eval/metrics.py 的定义）")
    p("arm | family | seg35 | seg4 | 官方Dice | DSC30 | HD95 | 官方SDlogJ | 脑内SDlogJ | 脑内J<.5% | 脑内|logJ|>log2 % | 单元中心J<=0 合计(对数) | 官方det<=0 合计(对数) | BL1证书通过 | 判定配置证书")
    for r in abs_rows:
        a = r["arm"]
        bl1 = "" if a == "identity" else f"{r['bl1_passed']}/{n_pos}"
        dc = f"{r['decision_cert_passed']}/{n_pos}" if a == method else ""
        p(f"{a} | {r['family']} | {fmt(r['seg35_mean'])} | {fmt(r['seg4_mean'])} | {fmt(r['dice_official_mean'])} | {fmt(r['dsc30'])} | "
          f"{fmt(r['hd95_mean'], 3)} | {fmt(r['sdlogj_official_mean'])} | {fmt(r['sdlogj_brain_mean'])} | {fmt(r['pct_J_lt_half_brain_mean'], 2)} | "
          f"{fmt(r['pct_frac_abs_logj_gt_log2_brain_mean'], 2)} | {r['cell_centre_J_nonpositive_sum']} ({r['cell_centre_J_nonpositive_pairs_gt0']}) | "
          f"{r['central_det_nonpositive_sum']} ({r['central_det_nonpositive_pairs_gt0']}) | {bl1} | {dc}")
    p()
    p("## 证书细目（BL1 jacobian_only，全单元，CPU FP64 外向舍入；对每臂的 u = h d 的 Q1 插值）")
    p("arm | 通过(逐对记录/decision) | status 计数 | n_infeasible 单元合计(逐对/decision) | 有 infeasible 的对数 | 其中脑内/背景 | n_uncertain 合计(逐对/decision) | 最小下界 | domain ok 对数 | cells 与 map 一致")
    for r in abs_rows:
        if r["arm"] == "identity":
            continue
        p(f"{r['arm']} | {r['bl1_passed']}/{r['bl1_passed_decision']} | {r['bl1_status']} | {r['bl1_n_infeasible_sum']}/{r['bl1_n_infeasible_decision']} | "
          f"{r['bl1_pairs_with_infeasible']} | {r['bl1_infeasible_brain_sum']}/{r['bl1_infeasible_background_sum']} | "
          f"{r['bl1_n_uncertain_sum']}/{r['bl1_n_uncertain_decision']} | {fmt_sig(r['bl1_min_jd_lower_bound'], 4)} | {r['bl1_domain_ok']} | {r['bl1_cells_matches_map']}")
    mr = [r for r in abs_rows if r["arm"] == method][0]
    p(f"method 判定配置证书（D112 (a)）：passed {mr['decision_cert_passed']}/{mr['decision_cert_passed_decision']}，n_infeasible {mr['decision_cert_n_infeasible_sum']}，"
      f"n_uncertain {mr['decision_cert_n_uncertain_sum']}，最小下界 {fmt_sig(mr['decision_cert_min_lb'], 4)}，mode {mr['decision_cert_mode']}，config {mr['decision_cert_config_class']}")
    p("seg35 与官方 Dice 不同的对数（|差| > 1e-12）与最大 |差|: " + "; ".join(
        f"{r['arm']} {r['pairs_seg35_ne_dice_official']} {fmt(r['max_abs_seg35_minus_dice_official'])}" for r in abs_rows))
    p("terminal_status: " + "; ".join(f"{r['arm']} {r['terminal_status']}" for r in abs_rows if r["arm"] != "identity"))
    p("export_suspicious 非空的对数: " + ", ".join(f"{r['arm']} {r['export_suspicious']}" for r in abs_rows if r["arm"] != "identity"))
    p()
    p("## L2R 口径的其余聚合（official 行；dice std ddof 0；.3 / .7 分位）")
    p("arm | Dice 均值 | Dice std | Dice30 分位 | HD95 .3 分位 | SDlogJ .7 分位 | DSC30（本脚本 / decision，差） | 最小单元中心 J")
    for r in abs_rows:
        dd = "" if r["arm"] == "identity" else f" / {fmt(r['dsc30_decision'])}, {fmt_sig(r['dsc30_abs_diff_vs_decision'], 2)}"
        p(f"{r['arm']} | {fmt(r['official_dice_mean'])} | {fmt(r['official_dice_std_ddof0'])} | {fmt(r['official_dice30_quantile'])} | "
          f"{fmt(r['official_hd95_30_quantile'], 3)} | {fmt(r['official_sdlogj_70_quantile'])} | {fmt(r['dsc30'])}{dd} | {fmt_sig(r['min_J_centre_min'], 4)}")
    p()
    p("## F1（seg35）与 F2（seg4）：方法 − 臂（decision.json）")
    p("族 | 臂 | 均值差 | 中位差 | 95% 区间 | 胜对 | p_sup | Holm sup（步，调整后，阈） | 判定 | SDlogJ 护栏（均值，区间，上端，成立）")
    for r in fam_rows:
        p(f"{r['family_test']} | {r['arm']} | {fmt(r['mean_diff'])} | {fmt(r['median_diff'])} | [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | "
          f"{r['wins']}/{r['n']} | {fmt_sig(r['p_sup'], 4)} | ({r['holm_sup_step']}, {fmt_sig(r['holm_sup_adjusted'], 4)}, {r['holm_sup_threshold']}) rejected {r['holm_sup_rejected']} | "
          f"{r['verdict']} | {fmt(r['guard_mean'])}, [{fmt(r['guard_lo'])}, {fmt(r['guard_hi'])}], {fmt(r['guard_upper'])}, {r['guard_holds']}")
    p("S2（hx_certified 子集，F1 seg35 区间）：" + "; ".join(f"{r['arm']} n {r['s2_n']} [{fmt(r['s2_lo'])}, {fmt(r['s2_hi'])}]"
                                                    for r in fam_rows if r["family_test"] == "F1"))
    p("从逐对记录复算的方法 − 臂（核对用）：")
    for r in diff_rows:
        p(f"  {r['arm']} ({r['family']}): seg35 均值差 {fmt(r['seg35_mean_diff'])} 中位 {fmt(r['seg35_median_diff'])} 胜 {r['seg35_wins']} 平 {r['seg35_ties']}; "
          f"seg4 均值差 {fmt(r['seg4_mean_diff'])} 中位 {fmt(r['seg4_median_diff'])} 胜 {r['seg4_wins']} 平 {r['seg4_ties']}")
    p()
    p("## F3（只描述，不判；未校正 95% 百分位区间；written 列给差的方向）")
    for r in f3_rows:
        p(f"{r['block']} | {r['endpoint']} | {r['arm']} ({r['family']}) | {fmt(r['mean_diff'], 5)} [{fmt(r['ci95_lo'], 5)}, {fmt(r['ci95_hi'], 5)}] | {r['written']}")
    p("官方 SDlogJ 均值 <= .10（itt）: " + ", ".join(f"{a} {v}" for a, v in sd010["itt"].items()))
    p("官方 SDlogJ 均值 <= .10（outputs_only）: " + ", ".join(f"{a} {v}" for a, v in sd010["outputs_only"].items()))
    p()
    p("## F4（PQ 的确认性检验；Ω_e 上 Δ_k = AUSE_k − AUSE_q）")
    p(f"reading {F4['reading']}; n_prime {F4['n_prime']}; n_P {F4['n_P']}; excluded {len(F4['excluded'])}; failed {len(F4['failed'])}; "
      f"eval_side " + ", ".join(f"{k} {len(v)}" for k, v in F4["eval_side"].items()) + f"; idx_sha256 {F4['idx_sha256'][:16]}")
    p("基线 | Δ 均值 | LB（2.5 百分位） | R | 区间 | Δ>0 的对 | 过 | hx_certified 子集 n / LB / R")
    for r in f4_rows:
        p(f"{r['baseline']} | {fmt(r['mean_delta'], 5)} | {fmt(r['LB'], 5)} | {fmt(r['R'], 4)} | [{fmt(r['ci95_lo'], 5)}, {fmt(r['ci95_hi'], 5)}] | "
          f"{r['pairs_delta_gt0']}/{r['n_pairs']} | {r['passed']} | {r['hx_subset_n']} / {fmt(r['hx_subset_LB'], 5)} / {fmt(r['hx_subset_R'], 4)}")
    p()
    p("## 修复消融（描述，不是臂，不进任何族；after − before，after = 方法臂的输出，before = 修复前位移）")
    for r in ab_rows:
        if r["n"]:
            p(f"{r['table']} ({r['line_statuses']}, n_pairs {r['n_pairs']}) | {r['endpoint']} | before {fmt(r['mean_before'], 5)} after {fmt(r['mean_after'], 5)} | "
              f"差 {fmt(r['mean_diff_after_minus_before'], 5)} 中位 {fmt(r['median_diff'], 5)} [{fmt(r['ci95_lo'], 5)}, {fmt(r['ci95_hi'], 5)}] | "
              f"after>before {r['pairs_after_gt_before']}，after<before {r['pairs_after_lt_before']}")
        else:
            p(f"{r['table']} ({r['line_statuses']}, n_pairs {r['n_pairs']}) | {r['endpoint']} | n 0")
    p(f"ablation other_statuses {len(A['other_statuses'])}, excluded {len(A['excluded'])}")
    p()
    p("## 第二次运行加的几节")
    p(f"hi_record.json sha256 {hi_sha} ({hi_size} B); structure_groups_v1.json sha256 {l_sha}")
    p(f"HI（第 2 步）：{len(hi_rows)} 人；backend {hi_backend}；device {hi_device}；gpu {hi_gpu}；每人 seconds_total 中位 {fmt(float(np.median(hi_sec['seconds_total'])), 2)}"
      f"（{fmt(float(hi_sec['seconds_total'].min()), 2)}–{fmt(float(hi_sec['seconds_total'].max()), 2)}），合计 {fmt(float(hi_sec['seconds_total'].sum()) / 60, 1)} 分钟；"
      f"前向中位 {fmt(float(np.median(hi_sec['seconds_forward'])), 2)}（{fmt(float(hi_sec['seconds_forward'].min()), 2)}–{fmt(float(hi_sec['seconds_forward'].max()), 2)}）；"
      f"写文件中位 {fmt(float(np.median(hi_sec['seconds_write'])), 2)}（{fmt(float(hi_sec['seconds_write'].min()), 2)}–{fmt(float(hi_sec['seconds_write'].max()), 2)}）")
    p("采样域与六面位移（BL1 记录的 domain 与 faces_max_abs_u_mm；不进证书状态）：节点数 " + ", ".join(sorted({str(int(x)) for x in n_nodes})))
    p("arm | 六面最大 |u| mm（最小/中位/最大） | 域内对数 | 有越界节点的对数 | 越界节点数（最小–最大） | 占节点比例（最小–最大）")
    for r in abs_rows:
        if r["arm"] == "identity":
            continue
        p(f"{r['arm']} | {fmt_sig(r['faces_max_abs_u_mm_min'], 4)} / {fmt_sig(r['faces_max_abs_u_mm_median'], 4)} / {fmt_sig(r['faces_max_abs_u_mm_max'], 4)} | "
          f"{r['bl1_domain_ok']} | {r['pairs_with_invalid_domain_nodes']} | {r['domain_invalid_nodes_min']}–{r['domain_invalid_nodes_max']} | "
          f"{100 * r['domain_invalid_fraction_min']:.4f}%–{100 * r['domain_invalid_fraction_max']:.3f}%")
    p("脑内 log J 的 1% / 99% 分位（50 对均值）: " + "; ".join(f"{r['arm']} {fmt(r['logj_q01_brain_mean'], 3)} / {fmt(r['logj_q99_brain_mean'], 3)}" for r in abs_rows))
    p()
    p("## 分结构 Dice（50 对均值；seg35 标签名照 structure_groups_v1.json；n 是该标签被计入的对数，0 记 0 的对数另列在 per_label_dice.csv）")
    show = [method] + list(main_family)
    p("label | name | " + " | ".join(show))
    for sc in ("seg35", "seg4"):
        ids = sorted({row["label"] for row in label_rows if row["scale"] == sc})
        for lid in ids:
            cells = {row["arm"]: row for row in label_rows if row["scale"] == sc and row["label"] == lid}
            p(f"{sc}:{lid} | {cells[method]['name']} | " + " | ".join(
                f"{fmt(cells[a]['mean'])} (n {cells[a]['n_pairs_present']}, 0 {cells[a]['n_pairs_zero']})" for a in show))
    p()
    p("## PQ 的描述量（C22.H7-5；位置无关的聚合；不进判定）")
    for r in pq_rows:
        p(" | ".join(f"{k} {fmt(v, 5) if isinstance(v, float) else v}" for k, v in r.items() if v is not None))
    p()
    p("## 逐对文件（只按位置序号，不进 git）")
    pp_sha = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in pp_texts.items()}
    for name in pp_texts:
        p(f"{PER_POS_DIR}/{name} sha256 {pp_sha[name]}")
    p("## 本目录 CSV")
    for name, text in csv_texts.items():
        p(f"{name} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    summary = "\n".join(out) + "\n"

    # ---- guard and write --------------------------------------------------------------------------------------------
    all_texts = dict(csv_texts)
    all_texts["summarize.out"] = summary
    for name, text in list(all_texts.items()) + list(pp_texts.items()):
        if guard.problems(name) or guard.problems(text):
            die(f"guard: a forbidden token in the text of {name}; nothing written")
    if os.path.exists(PER_POS_DIR):
        die("per_position directory exists; refusing to overwrite; nothing written")
    os.makedirs(PER_POS_DIR)
    for name, text in pp_texts.items():
        with open(os.path.join(PER_POS_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    for name, text in all_texts.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    sys.stdout.write(f"summarize.py: wrote {len(all_texts)} files to {OUT_DIR} and {len(pp_texts)} to {PER_POS_DIR}\n")


if __name__ == "__main__":
    # 试跑：--dry-run <新的空目录>，两组输出都写到这个目录下（out/ 与 per_position/），不碰本目录与结果根。
    if len(sys.argv) == 3 and sys.argv[1] == "--dry-run":
        target = os.path.abspath(sys.argv[2])
        if not os.path.isdir(target) or os.listdir(target) or target.startswith("/home/rmcfz/seq_vp_tag_results"):
            die("--dry-run needs an existing empty directory outside the results root")
        OUT_DIR = os.path.join(target, "out")
        os.makedirs(OUT_DIR)
        PER_POS_DIR = os.path.join(target, "per_position")
    elif len(sys.argv) != 1:
        die("usage: summarize.py [--dry-run <empty dir>]", 2)
    main()
