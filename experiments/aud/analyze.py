"""AUD analysis (runs/20261102-fb-audit/plan.md): reads only the JSON of experiments/aud/fb_audit.py and the existing
evaluation records (Dice, certificate), writes summary.json and summary.md.  No model, no data/, no label array.

    python -B experiments/aud/analyze.py --root /home/rmcfz/seq_vp_tag_results/fb_audit_v1

The offline weight analysis treats the blocks of a field as fixed numbers:
    F(nu, s_tau, w_I, w_S) = w_I F_I + w_S F_S + F_diff + (nu/nu_0) F_curl + F_vol(s_tau),
with X preferred to the reference R when M = F(X) - F(R) < 0.  It compares these finitely many fields only: it says
nothing about the minimizer of F at the changed weights.  alpha stays .01 (the sweep's alpha follows tau; not here).

Revision 1 (the lead's request of 2026-11-02 ~5:40 pm CST; plan, revision 1): the profile objective
    L(u) = (N_q/2) sum_c log sigma~_c^2(u) + Pi_D + Pi_J + Pi_C,   sigma~_c^2(u) = <rho_c^2>_B/(N_q + 2)
channel by channel; L(A0) - L(M2V1), L(A1) - L(M2V1) on channel subsets (14 channels; intensity + cerebral_wm +
cerebral_cortex; intensity only) x nu in {21.855, 7, 1.37, 0} x tau in {.5, 1, inf} (tau_k, k >= 1; tau_0 = 1 fixed; tau = inf
is the limit F_vol = 0) with the flip boundaries; cortex and white matter Dice per field (existing records); the
high-frequency displacement RMS of experiments/aud/hf_disp.py when hf/ exists.  --name chooses the output name, so that the
first summary of the queue stays as it was written.

Two "L" in the output (review AUD_复核_20261102, minor 7): sections 3 and 10 of summary.md report the full C11-1 form (m_step
"L" of fb_audit.py), sections 12-14 the form above; the full form has the constant C (N_q + 2)/2 (C channels) and sum_c log
sigma~_c^2 in addition, so the differences between fields agree up to sum_c delta log sigma~_c^2.  The table headers say so
since the commit after 6f7970e: this code's summary.md differs from fb_audit_v1/summary_r1.md (reproduced byte for byte from
the snapshot 6f7970e; runs/20261102-fb-audit/result.md) in those header lines; summary.json is unchanged.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
GROUPS_JSON = Path(__file__).resolve().parents[2]/"configs"/"bayesreg"/"structure_groups_v1.json"
FIELD_ORDER = ("u0", "M2V1", "P3", "P4", "A0", "A1", "A4p")
COMPARISONS = (("A0", "M2V1"), ("A1", "M2V1"), ("A4p", "M2V1"), ("A0", "P3"), ("A1", "P3"), ("A4p", "P3"))
TERMS = ("F_intensity", "F_structure", "F_diff", "F_vol", "F_curl")
RANGE = dict(nu=(.7, 874.), s_tau=(.5, 4.), w_I=(.1, 10.), w_S=(.1, 10.), prior_scale=(.1, 10.))
NU_LIST, S_LIST = (21.85496512638118, 7., .7), (1., 2., 4.)
NU_GRID = (21.85496512638118, 7., 1.37, 0.)
TAU_GRID = ((.5, 1.), (1., 2.), (float("inf"), None))           # (tau_k, s_tau of the F_vol table; None: the limit F_vol = 0)
SUBSETS = (("14 通道", None), ("强度 + 白质 + 皮层", ("intensity", "cerebral_wm", "cerebral_cortex")), ("只有强度", ("intensity",)))
PROFILE_COMPARISONS = (("A0", "M2V1"), ("A1", "M2V1"), ("A4p", "M2V1"), ("A0", "P3"), ("A1", "P3"))


# ---- pure pieces (tested on made-up numbers) ------------------------------------------------------------------------

def linear_crossing(rest, coef, base=1.):
    """M(p) = rest + coef p/base on p > 0.  X is preferred where M < 0.  Returns dict(star, side, always): star the crossing
    (None without a positive one), side "below" (X preferred for p < star) or "above", always True / False when M has one
    sign for all p > 0."""
    if coef == 0.:
        return dict(star=None, side=None, always=bool(rest < 0))
    star = -rest*base/coef
    if star > 0.:
        return dict(star=float(star), side="below" if coef > 0. else "above", always=None)
    return dict(star=None, side=None, always=bool(coef < 0.))


def grid_crossings(grid, values):
    """Sign changes of `values` on the increasing `grid`, log-linear in the grid value; dict(stars, preferred_at_ends)."""
    grid, values = np.asarray(grid, dtype=float), np.asarray(values, dtype=float)
    stars = []
    for j in range(len(grid)-1):
        a, b = values[j], values[j+1]
        if (a < 0.) != (b < 0.):
            t = a/(a-b)
            stars.append(float(np.exp(np.log(grid[j])+t*(np.log(grid[j+1])-np.log(grid[j])))))
    return dict(stars=stars, preferred_at_low_end=bool(values[0] < 0.), preferred_at_high_end=bool(values[-1] < 0.))


def in_range(name, value):
    if value is None:
        return None
    low, high = RANGE[name]
    return bool(low <= value <= high)


def judge(n_lower, n_total=5):
    """Plan: J1 F_B(A0) < F_B(M2V1) on >= 4/5 pairs; J2 the opposite on >= 4/5; J3 otherwise."""
    if n_lower >= n_total-1:
        return "J1"
    if n_total-n_lower >= n_total-1:
        return "J2"
    return "J3"


def main_term(mean_diff):
    """The largest positive mean difference among TERMS and its share of the sum of the positive ones."""
    positive = {k: v for k, v in mean_diff.items() if k in TERMS and v > 0.}
    if not positive:
        return dict(term=None, share=None)
    term = max(positive, key=positive.get)
    return dict(term=term, share=float(positive[term]/sum(positive.values())))


def flip_boundaries(x, r, nu0):
    """The flip of X over R (blocks of fb_audit's field JSON at sigma (a)); per parameter, the others at their values."""
    d = {k: x[k]-r[k] for k in TERMS}
    data = d["F_intensity"]+d["F_structure"]
    prior = d["F_diff"]+d["F_vol"]+d["F_curl"]
    grid = x["prior_pass"]["s_grid"]
    vol_x = np.add(x["prior_pass"]["F_vol_gauss_of_s"], x["prior_pass"]["F_vol_closed27_of_s"])
    vol_r = np.add(r["prior_pass"]["F_vol_gauss_of_s"], r["prior_pass"]["F_vol_closed27_of_s"])
    d_vol_s = vol_x-vol_r
    out = dict(M=data+prior, data=data, prior=prior, terms=d,
               nu=linear_crossing(data+d["F_diff"]+d["F_vol"], d["F_curl"], nu0),
               w_I=linear_crossing(d["F_structure"]+prior, d["F_intensity"]),
               w_S=linear_crossing(d["F_intensity"]+prior, d["F_structure"]),
               prior_scale=linear_crossing(data, prior),
               s_tau=grid_crossings(grid, data+d["F_diff"]+d["F_curl"]+d_vol_s),
               s_grid=list(grid), d_vol_of_s=[float(v) for v in d_vol_s])
    out["nu_of_s"] = [dict(s_tau=float(s), **linear_crossing(data+d["F_diff"]+float(v), d["F_curl"], nu0))
                      for s, v in zip(grid, d_vol_s)]
    table = []
    for nu in NU_LIST:
        for s in S_LIST:
            v = float(d_vol_s[list(grid).index(s)])
            table.append(dict(nu=nu, s_tau=s, **linear_crossing(d["F_intensity"]+d["F_diff"]+nu/nu0*d["F_curl"]+v,
                                                               d["F_structure"])))
    out["w_S_on_nu_s"] = table
    return out


def data_at_sigma(rho2, sigma2):
    """E_c = <rho_c^2>_B/(2 sigma_c^2)."""
    return [.5*a/b for a, b in zip(rho2, sigma2)]


def energy_at_sigma(field, sigma2):
    return float(sum(data_at_sigma(field["rho2_B"], sigma2)))+field["F_diff"]+field["F_vol"]+field["F_curl"]


def shared_structure_sigma(field, n_q):
    """sigma_I^2 by the field's M step, sigma_S^2 by one shared M step over all structure channels on the whole brain (a
    what-if of the audit; not the contract's support_shared, which has a support weight).

    The denominator is k (N_q + 2); the M step of the shared model at a = b = 0 (C11-8) has k N_q + 2 (review
    AUD_复核_20261102, minor 5).  The difference is about 1e-6 relative in the fixed-sigma rows; kept, so that the outputs stay
    those of summary_r1.*."""
    rho2 = field["rho2_B"]
    k = len(rho2)-1
    return [rho2[0]/(n_q+2.)]+[sum(rho2[1:])/(k*(n_q+2.))]*k


def profile_terms(field, n_q):
    """(N_q/2) log sigma~_c^2(u) per channel, sigma~_c^2 = <rho_c^2>_B/(N_q + 2)."""
    return .5*n_q*np.log(np.asarray(field["rho2_B"], dtype=np.float64)/(n_q+2.))


def vol_table(field):
    """F_vol on the s_tau grid (Gauss27 + closed27 parts)."""
    return np.add(field["prior_pass"]["F_vol_gauss_of_s"], field["prior_pass"]["F_vol_closed27_of_s"])


def profile_value(field, n_q, indices, nu_over_nu0=1., s_tau=1.):
    """L(u) on the channel subset `indices` at (nu, s_tau); s_tau None: the limit F_vol = 0.  A direct evaluation, for the
    tests of profile_grid."""
    grid = field["prior_pass"]["s_grid"]
    vol = 0. if s_tau is None else float(vol_table(field)[grid.index(s_tau)])
    return float(profile_terms(field, n_q)[list(indices)].sum())+field["F_diff"]+nu_over_nu0*field["F_curl"]+vol


def profile_grid(x, r, n_q, nu0, names):
    """L(X) - L(R) on SUBSETS x NU_GRID x TAU_GRID and the flip boundaries; X is preferred where the difference is < 0."""
    grid = x["prior_pass"]["s_grid"]
    d_terms = profile_terms(x, n_q)-profile_terms(r, n_q)
    d_diff, d_curl = x["F_diff"]-r["F_diff"], x["F_curl"]-r["F_curl"]
    d_vol_s = vol_table(x)-vol_table(r)
    d_vol = {tau: (0. if s is None else float(d_vol_s[grid.index(s)])) for tau, s in TAU_GRID}
    out = dict(d_diff=d_diff, d_curl=d_curl, d_vol={str(tau): v for tau, v in d_vol.items()}, channels=[float(v) for v in d_terms], subsets={})
    for label, chosen in SUBSETS:
        indices = list(range(len(names))) if chosen is None else [names.index(n) for n in chosen]
        D = float(d_terms[indices].sum())
        cells = [dict(nu=nu, tau=tau, M=D+d_diff+nu/nu0*d_curl+d_vol[tau]) for nu in NU_GRID for tau, _ in TAU_GRID]
        nu_of_tau = [dict(tau=tau, **linear_crossing(D+d_diff+d_vol[tau], d_curl, nu0)) for tau, _ in TAU_GRID]
        s_of_nu = [dict(nu=nu, M_at_tau_inf=D+d_diff+nu/nu0*d_curl, **grid_crossings(grid, D+d_diff+nu/nu0*d_curl+d_vol_s))
                   for nu in NU_GRID]
        data_weight = linear_crossing(d_diff+d_curl+d_vol[.5], D)
        out["subsets"][label] = dict(channels=[names[i] for i in indices], D=D, cells=cells, nu_of_tau=nu_of_tau, s_of_nu=s_of_nu,
                                     data_weight=data_weight)
    return out


def region_dice(per35, per4):
    """Cortex and white matter: seg35 {2, 21}, {1, 20} (structure_groups_v1 names, checked) and seg4 labels 1, 3."""
    return dict(cortex_seg35=group_dice(per35, (2, 21)), wm_seg35=group_dice(per35, (1, 20)), cortex_seg4=per4.get("1"), wm_seg4=per4.get("3"))


# ---- readers ----------------------------------------------------------------------------------------------------------

def load_raw(root):
    pairs, raw = {}, {}
    for directory in sorted((root/"raw").iterdir()):
        if not directory.is_dir():
            continue
        short = directory.name
        pairs[short] = json.loads((directory/"pair.json").read_text())
        raw[short] = {f: json.loads((directory/f"{f}.json").read_text()) for f in FIELD_ORDER}
    return pairs, raw


def record_path(field, short, long):
    ev = "eval"
    if field == "M2V1":
        return RESULTS/"m2v1_dev"/ev/"dev"/short/f"m2v1_dev_{short}"/"metrics.json"
    if field in ("P3", "P4"):
        low = field.lower()
        return RESULTS/"path_pilot_v1"/ev/low/short/f"pilot_{low}_{short}"/"metrics.json"
    if field in ("A0", "A4p"):
        hits = sorted((RESULTS/"m2v1_dev"/"ref_reeval"/field/short).glob("*/metrics.json"))
        if len(hits) != 1:
            raise RuntimeError(f"{field} {short}: {len(hits)} records")
        return hits[0]
    if field == "A1":
        return RESULTS/"d0_v1"/"runs"/"A1"/long/"metrics.json"
    raise ValueError(field)


def read_record(field, short, long):
    """Dice and certificate from the existing records; nothing recomputed."""
    if field == "u0":
        path = record_path("A0", short, long)
        doc = json.loads(path.read_text())
        labels = doc["per_label"]["initial_dice_official_labels"]
        m1 = json.loads((RESULTS/"gap_decomp_v1"/"m1.json").read_text())["A0"]
        row = [r for r in m1 if r["pair"] == short][0]
        return dict(seg35=doc["metrics"]["initial_dice_official"], seg4=float(np.nanmean(row["init_seg4_per"])),
                    sdlogj_official=None, sdlogj_brain=None, hd95=None, cert_status="u = 0 (J = 1; no record)",
                    cert_min=None, per35={str(i+1): v for i, v in enumerate(labels)},
                    per4={str(i+1): v for i, v in enumerate(row["init_seg4_per"])}, field_role="u = 0",
                    source=f"{path} initial_dice_official; gap_decomp_v1/m1.json init_seg4_per")
    path = record_path(field, short, long)
    doc = json.loads(path.read_text())
    m = doc["metrics"]
    per = doc.get("per_label") or m
    cert = doc["certificate"]
    return dict(seg35=m["seg35"], seg4=m["seg4"], sdlogj_official=m.get("sdlogj_official"), sdlogj_brain=m.get("sdlogj_brain"),
                hd95=m.get("hd95"), cert_status=cert.get("status"), cert_min=cert.get("min_jd_lower_bound"),
                per35=per["seg35_per_label"], per4=per["seg4_per_label"], field_role=doc.get("field_role"), source=str(path))


def groups_of(names):
    doc = json.loads(GROUPS_JSON.read_text())
    groups = doc["groupings"]["structure13_v1"]["groups"]
    if [g["name"] for g in groups] != list(names[1:]):
        raise RuntimeError("channel names differ from structure13_v1")
    return groups


def group_dice(per35, labels):
    values = [per35[str(i)] for i in labels if str(i) in per35 and per35[str(i)] is not None
              and not (isinstance(per35[str(i)], float) and np.isnan(per35[str(i)]))]
    return float(np.mean(values)) if values else None


# ---- tables ------------------------------------------------------------------------------------------------------------

def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def mil(x):
    return "—" if x is None else f"{x/1e6:.3f}"


def num(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def sci(x):
    return "—" if x is None else f"{x:.3g}"


def crossing_text(c, name=None):
    """name: a key of RANGE; the plan's reading "（超出）" for a crossing outside the range tried."""
    if c.get("star") is not None:
        text = ("< " if c["side"] == "below" else "> ")+f"{c['star']:.3g}"
        return text if name is None or in_range(name, c["star"]) else text+"（超出）"
    return "任意正值都偏爱 X" if c.get("always") else "无解"


def s_tau_text(c):
    """The crossings of grid_crossings with the side where X is preferred; "（超出）" outside the range tried."""
    if c["stars"]:
        marks = [f"{s:.3g}"+("" if in_range("s_tau", s) else "（超出）") for s in c["stars"]]
        if len(marks) == 1:
            return ("> " if c["preferred_at_high_end"] else "< ")+marks[0]
        return ", ".join(marks)
    if c["preferred_at_low_end"] and c["preferred_at_high_end"]:
        return "网格内都偏爱 X"
    return "网格内无解"


def analyze(root):
    pairs, raw = load_raw(root)
    shorts = sorted(raw)
    records = {s: {f: read_record(f, s, pairs[s]["pair"].split("/")[1]) for f in FIELD_ORDER} for s in shorts}
    names = pairs[shorts[0]]["channel_names"]
    groups = groups_of(names)
    nu0 = pairs[shorts[0]]["eta"]["nu"]
    doc = dict(schema="bayesreg.aud.summary.v1", pairs=shorts, fields=FIELD_ORDER, channel_names=names, nu0=nu0,
               note="dev set, fixed fields, single observations; finite comparison of existing fields only")

    checks = {s: {f: raw[s][f]["consistency"] for f in ("M2V1", "P3")} for s in shorts}
    doc["consistency"] = checks
    doc["tile_checks_max"] = float(max(v for s in shorts for f in FIELD_ORDER for v in raw[s][f]["checks"].values()))

    # per-field means
    per_field = {}
    for f in FIELD_ORDER:
        row = {k: mean([raw[s][f][k] for s in shorts]) for k in ("F_B", "F_data", "F_diff", "F_vol", "F_vol_gauss",
                                                                 "F_vol_closed27", "F_curl")+TERMS[:2]}
        row.update({k: mean([records[s][f][k] for s in shorts]) for k in ("seg35", "seg4", "sdlogj_official", "sdlogj_brain", "hd95")})
        row.update(curl_sq=mean([raw[s][f]["curl_sq_total"] for s in shorts]), uKu=mean([raw[s][f]["uKu"] for s in shorts]),
                   logvol_sq=mean([raw[s][f]["logvol_sq_total"] for s in shorts]),
                   logj_q01=mean([raw[s][f]["log_j"]["values"][0] for s in shorts]),
                   logj_q99=mean([raw[s][f]["log_j"]["values"][-1] for s in shorts]),
                   logj_std=mean([raw[s][f]["log_j"]["std"] for s in shorts]),
                   u_mean=mean([raw[s][f]["u_mean_norm_brain_mm"] for s in shorts]),
                   cert_min=[records[s][f]["cert_min"] for s in shorts], cert_status=[records[s][f]["cert_status"] for s in shorts],
                   field_role=[records[s][f]["field_role"] for s in shorts],
                   closed27_min_J=[raw[s][f]["closed27_min_J"] for s in shorts],
                   L=mean([raw[s][f]["m_step"]["L"] for s in shorts]))
        row["curl_brain_share"] = mean([raw[s][f]["prior_pass"]["curl_sq"]["brain"]/raw[s][f]["curl_sq_total"]
                                        if raw[s][f]["curl_sq_total"] else None for s in shorts])
        row["logvol_brain_share"] = mean([(raw[s][f]["prior_pass"]["logvol_gauss_sq"]["brain"]+raw[s][f]["prior_pass"]["logvol_closed27_sq"]["brain"])
                                          / raw[s][f]["logvol_sq_total"] if raw[s][f]["logvol_sq_total"] else None for s in shorts])
        per_field[f] = row
    doc["per_field_mean"] = per_field
    doc["per_pair"] = {s: {f: {k: raw[s][f][k] for k in ("F_B", "F_data", "F_diff", "F_vol", "F_vol_gauss", "F_vol_closed27",
                                                         "F_curl")+TERMS[:2]}
                           | dict(seg35=records[s][f]["seg35"], seg4=records[s][f]["seg4"], cert_min=records[s][f]["cert_min"],
                                  L=raw[s][f]["m_step"]["L"], record=records[s][f]["source"]) for f in FIELD_ORDER} for s in shorts}

    # comparisons and flips
    comparisons = {}
    for x, r in COMPARISONS:
        key = f"{x}-{r}"
        rows = {s: flip_boundaries(raw[s][x], raw[s][r], nu0) for s in shorts}
        mean_diff = {k: mean([rows[s]["terms"][k] for s in shorts]) for k in TERMS}
        lower = [s for s in shorts if rows[s]["M"] < 0.]
        d_l = {s: raw[s][x]["m_step"]["L"]-raw[s][r]["m_step"]["L"] for s in shorts}
        comparisons[key] = dict(x=x, r=r, per_pair=rows, mean_diff=mean_diff, mean_M=mean([rows[s]["M"] for s in shorts]),
                                n_x_lower=len(lower), lower_pairs=lower, judge=judge(len(lower), len(shorts)),
                                main_term=main_term(mean_diff), L_diff=d_l, n_x_lower_L=sum(v < 0. for v in d_l.values()),
                                d_seg35=mean([records[s][x]["seg35"]-records[s][r]["seg35"] for s in shorts]),
                                d_seg4=mean([records[s][x]["seg4"]-records[s][r]["seg4"] for s in shorts]),
                                L_parts={s: dict(
                                    intensity=raw[s][x]["m_step"]["L_log_channel"][0]-raw[s][r]["m_step"]["L_log_channel"][0],
                                    structure=sum(raw[s][x]["m_step"]["L_log_channel"][1:])-sum(raw[s][r]["m_step"]["L_log_channel"][1:]),
                                    prior=rows[s]["prior"]) for s in shorts})
    doc["comparisons"] = comparisons
    doc["decision"] = dict(rule="plan: A0 vs M2V1, sigma (a), n = 5", comparison="A0-M2V1", **{k: comparisons["A0-M2V1"][k] for k in ("judge", "n_x_lower", "main_term")})

    # sigma what-ifs: the same field pair under other (common) sigma
    whatif = {}
    for x in ("A0", "A1", "A4p"):
        rows = {}
        for s in shorts:
            nq = pairs[s]["N_q"]
            fx, fr = raw[s][x], raw[s]["M2V1"]
            options = dict(sigma_a=pairs[s]["sigma2_a"], m_step_M2V1=fr["m_step"]["sigma2"], m_step_X=fx["m_step"]["sigma2"],
                           shared_S_M2V1=shared_structure_sigma(fr, nq), shared_S_X=shared_structure_sigma(fx, nq))
            rows[s] = {name: dict(M=energy_at_sigma(fx, sig)-energy_at_sigma(fr, sig),
                                  data=float(sum(data_at_sigma(fx["rho2_B"], sig))-sum(data_at_sigma(fr["rho2_B"], sig))))
                       for name, sig in options.items()}
        whatif[f"{x}-M2V1"] = rows
    doc["sigma_what_if"] = whatif

    # channels
    channels = {}
    for x, r in (("A1", "M2V1"), ("A0", "M2V1"), ("A4p", "M2V1"), ("A1", "A0"), ("A0", "P3")):
        rows = []
        for c, name in enumerate(names):
            labels = None if c == 0 else groups[c-1]["labels"]
            dE, ratio, dD, sigma, mass = [], [], [], [], []
            for s in shorts:
                dE.append(raw[s][x]["channel"][c]-raw[s][r]["channel"][c])
                ratio.append(raw[s][x]["rho2_B"][c]/raw[s][r]["rho2_B"][c])
                sigma.append(pairs[s]["sigma2_a"][c])
                mass.append(pairs[s]["channel_mean_fixed_square"][c])
                if labels is not None:
                    a, b = group_dice(records[s][x]["per35"], labels), group_dice(records[s][r]["per35"], labels)
                    dD.append(None if a is None or b is None else a-b)
            rows.append(dict(channel=c, name=name, labels=labels, mean_dE=mean(dE), per_pair_dE=dE, n_dE_positive=sum(v > 0. for v in dE),
                             geomean_rho2_ratio=float(np.exp(np.mean(np.log(ratio)))), mean_inverse_sigma2=mean([1./v for v in sigma]),
                             mean_fixed_square=mean(mass),
                             mean_dDice=None if labels is None else mean(dD), n_dDice_positive=None if labels is None else sum(v is not None and v > 0. for v in dD)))
        struct = rows[1:]
        rho = spearmanr([t["mean_dDice"] for t in struct], [np.log(t["geomean_rho2_ratio"]) for t in struct])
        channels[f"{x}-{r}"] = dict(rows=rows, spearman_dDice_vs_logrho2=dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(struct)),
                                    contradictions=[t["name"] for t in struct if (t["mean_dDice"] or 0.) > 0. and t["mean_dE"] > 0.],
                                    seg4_per=[dict(label=str(k), x=mean([records[s][x]["per4"][str(k)] for s in shorts]),
                                                   r=mean([records[s][r]["per4"][str(k)] for s in shorts])) for k in (1, 2, 3, 4)])
    doc["channels"] = channels

    # (b) sigma~ per field
    doc["m_step_sigma2"] = {f: {name: mean([raw[s][f]["m_step"]["sigma2"][c] for s in shorts]) for c, name in enumerate(names)}
                            for f in FIELD_ORDER}

    # revision 1: the profile objective, subsets x (nu, tau), flips; cortex / white matter Dice; high-frequency displacement
    terms = {f: {s: profile_terms(raw[s][f], pairs[s]["N_q"]) for s in shorts} for f in FIELD_ORDER}
    doc["profile"] = dict(
        definition="L(u) = (N_q/2) sum_c log sigma~_c^2(u) + Pi_D + Pi_J + Pi_C, sigma~_c^2 = <rho_c^2>_B/(N_q + 2)",
        per_field={f: dict(L_mean=mean([float(terms[f][s].sum())+raw[s][f]["F_diff"]+raw[s][f]["F_vol"]+raw[s][f]["F_curl"] for s in shorts]),
                           L_per_pair={s: float(terms[f][s].sum())+raw[s][f]["F_diff"]+raw[s][f]["F_vol"]+raw[s][f]["F_curl"] for s in shorts},
                           channel_mean={name: mean([float(terms[f][s][c]) for s in shorts]) for c, name in enumerate(names)})
                   for f in FIELD_ORDER},
        comparisons={f"{x}-{r}": {s: profile_grid(raw[s][x], raw[s][r], pairs[s]["N_q"], nu0, names) for s in shorts}
                     for x, r in PROFILE_COMPARISONS})
    doc["region_dice"] = {f: {s: region_dice(records[s][f]["per35"], records[s][f]["per4"]) for s in shorts} for f in FIELD_ORDER}
    hf_root = root/"hf"
    if hf_root.is_dir():
        doc["hf"] = {s: {p.stem: json.loads(p.read_text())["regions"] for p in sorted((hf_root/s).glob("*.json"))} for s in shorts}
    return doc


def markdown_revision1(doc):
    names, shorts, fields = doc["channel_names"], doc["pairs"], doc["fields"]
    prof = doc["profile"]
    out = ["", "# 修订 1 的读数（总负责人 11-02 约 5:40 pm CST 的追加要求）", "",
           f"**{prof['definition']}**。每个场各自做一次 M 步，所以 L 可以跨场比较。能量单位 1e6。"
           "第 3、10 节的 L 是 C11-1 全式，比这里多常数 C(N_q + 2)/2（C 为通道数）与 Σ_c log σ̃_c²，场间差只差 Σ_c Δlog σ̃_c²。", "",
           "## 12. 各场的 L 与逐通道项 (N_q/2) log σ̃_c²（5 对均值）", "",
           "| 通道 | "+" | ".join(fields)+" |", "|---|"+"---|"*len(fields)]
    for name in names:
        out.append(f"| {name} | "+" | ".join(mil(prof["per_field"][f]["channel_mean"][name]) for f in fields)+" |")
    out.append("| **L** | "+" | ".join(mil(prof["per_field"][f]["L_mean"]) for f in fields)+" |")
    out += ["", "### 逐通道的差（X − 参照，5 对均值；括号里是该通道 X 更低的对数）", "",
            "| 通道 | "+" | ".join(prof["comparisons"])+" |", "|---|"+"---|"*len(prof["comparisons"])]
    for c, name in enumerate(names):
        cells = []
        for rows in prof["comparisons"].values():
            values = [rows[s]["channels"][c] for s in shorts]
            cells.append(f"{mean(values)/1e6:+.3f}（{sum(v < 0. for v in values)}/5）")
        out.append(f"| {name} | "+" | ".join(cells)+" |")
    out += ["", "## 13. 通道子集 × (ν, τ)：L(X) − L(M2V1)（5 对均值；括号里是 X 更低的对数）", ""]
    for key in ("A0-M2V1", "A1-M2V1", "A4p-M2V1", "A0-P3", "A1-P3"):
        rows = prof["comparisons"][key]
        out += [f"### {key}", "", "| 子集 | ν | τ = .5 | τ = 1 | τ = ∞ |", "|---|---|---|---|---|"]
        for label, _ in SUBSETS:
            for j, nu in enumerate(NU_GRID):
                cells = []
                for k in range(len(TAU_GRID)):
                    values = [rows[s]["subsets"][label]["cells"][j*len(TAU_GRID)+k]["M"] for s in shorts]
                    cells.append(f"{mean(values)/1e6:+.3f}（{sum(v < 0. for v in values)}/5）")
                out.append(f"| {label} | {nu:.3g} | "+" | ".join(cells)+" |")
        out.append("")
    out += ["## 14. 翻转边界（剖面目标）", "", "\"> x\" 表示大于 x 时 X 更优，\"< x\" 表示小于 x 时更优；其余参数不动。s_τ 的网格是 1/16 到 16，τ_k = .5 s_τ。", ""]
    for key in ("A0-M2V1", "A1-M2V1"):
        rows = prof["comparisons"][key]
        out += [f"### {key}", "", "| 子集 | 边界 | "+" | ".join(shorts)+" |", "|---|---|"+"---|"*len(shorts)]
        for label, _ in SUBSETS:
            for k, (tau, _s) in enumerate(TAU_GRID):
                out.append(f"| {label} | 临界 ν（τ = {tau:g}） | "+" | ".join(crossing_text(rows[s]["subsets"][label]["nu_of_tau"][k], "nu") for s in shorts)+" |")
            for j, nu in enumerate(NU_GRID):
                out.append(f"| {label} | 临界 s_τ（ν = {nu:.3g}） | "+" | ".join(s_tau_text(rows[s]["subsets"][label]["s_of_nu"][j]) for s in shorts)+" |")
            out.append(f"| {label} | 数据项整体权重（ν 21.855，τ .5） | "+" | ".join(crossing_text(rows[s]["subsets"][label]["data_weight"], "prior_scale") for s in shorts)+" |")
        out.append("")
    out += ["## 15. 皮层、白质的 Dice（读已有记录，5 对均值）", "", "| 场 | 皮层 seg35 {2, 21} | 白质 seg35 {1, 20} | 皮层 seg4 标签 1 | 白质 seg4 标签 3 |", "|---|---|---|---|---|"]
    for f in fields:
        out.append(f"| {f} | "+" | ".join(num(mean([doc["region_dice"][f][s][k] for s in shorts])) for k in ("cortex_seg35", "wm_seg35", "cortex_seg4", "wm_seg4"))+" |")
    out += ["", "seg4 标签名按 neurite-oasis 标签表，未与 seg4 标签文件逐项核对（同 runs/20261102-gap-decomp/result.md）；seg35 的名称已核对。"]
    if "hf" in doc:
        out += ["", "## 16. 高频位移（u − G_{σ=2} * u，RMS，mm，5 对均值）", "",
                "区域取固定图像的真值标签（评测一侧）。", "",
                "| 场 | 皮层 seg35 高频 | 皮层 低频 | 白质 seg35 高频 | 白质 低频 | 全脑 高频 | 全脑 低频 |", "|---|---|---|---|---|---|---|"]
        for f in ("A0", "M2V1", "A1", "A4p", "P3", "P4"):
            def get(region, key):
                return mean([doc["hf"][s][f][region][key] for s in shorts])
            out.append(f"| {f} | {num(get('cortex_seg35', 'rms_high'), 3)} | {num(get('cortex_seg35', 'rms_low'), 3)} | {num(get('wm_seg35', 'rms_high'), 3)} | "
                       f"{num(get('wm_seg35', 'rms_low'), 3)} | {num(get('brain_seg4', 'rms_high'), 3)} | {num(get('brain_seg4', 'rms_low'), 3)} |")
        out += ["", "### A0、M2V1 逐对（皮层高频）", "", "| 对 | A0 | M2V1 | A0/M2V1 |", "|---|---|---|---|"]
        for s in shorts:
            a, b = doc["hf"][s]["A0"]["cortex_seg35"]["rms_high"], doc["hf"][s]["M2V1"]["cortex_seg35"]["rms_high"]
            out.append(f"| {s} | {a:.3f} | {b:.3f} | {a/b:.2f} |")
    return "\n".join(out)+"\n"


def markdown(doc):
    per, comps, names = doc["per_field_mean"], doc["comparisons"], doc["channel_names"]
    shorts = doc["pairs"]
    out = ["# F_B 审计：汇总（experiments/aud/analyze.py 生成）", "",
           "开发集 stage_L2R_val_v1 前 5 对，固定场上的求值，每对每场单次观测。只比较这几个有限的场，不代表真正的极小元。", "",
           "能量单位 1e6。σ 取 (a)：M2V1 该对原生层的 sigma2_used，所有场相同。", "", "## 1. 各场 5 对均值", "",
           "| 场 | F_B | F_强度 | F_结构层 | F_diff | F_vol | F_curl | seg35 | seg4 | SDlogJ 官方 | 脑内 log J 标准差 | log J q01 / q99 | 脑内 \\|u\\| 均值 mm |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in doc["fields"]:
        r = per[f]
        out.append(f"| {f} | {mil(r['F_B'])} | {mil(r['F_intensity'])} | {mil(r['F_structure'])} | {mil(r['F_diff'])} | {mil(r['F_vol'])} | {mil(r['F_curl'])} | "
                   f"{num(r['seg35'])} | {num(r['seg4'])} | {num(r['sdlogj_official'])} | {num(r['logj_std'], 3)} | {num(r['logj_q01'], 2)} / {num(r['logj_q99'], 2)} | {num(r['u_mean'], 2)} |")
    out += ["", "u0 的 seg35 是记录里的 initial_dice_official，seg4 取 gap_decomp_v1/m1.json。log J 在单元中心，脑取模型的 Atropos 掩膜。", "",
            "## 2. 逐对 F_B", "", "| 对 | "+" | ".join(doc["fields"])+" |", "|---|"+"---|"*len(doc["fields"])]
    for s in shorts:
        out.append(f"| {s} | "+" | ".join(mil(doc["per_pair"][s][f]["F_B"]) for f in doc["fields"])+" |")
    out += ["", "## 3. 分项差（X − 参照，5 对均值）", "",
            "| X − 参照 | F_B 差 | F_强度 | F_结构层 | F_diff | F_vol | F_curl | X 更低的对数 | 主项（份额） | L 差，C11-1 全式（X 更低的对数） | seg35 差 | seg4 差 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in comps.items():
        d, m = c["mean_diff"], c["main_term"]
        term = "—" if m["term"] is None else f"{m['term']}（{m['share']:.2f}）"
        out.append(f"| {key} | {mil(c['mean_M'])} | {mil(d['F_intensity'])} | {mil(d['F_structure'])} | {mil(d['F_diff'])} | {mil(d['F_vol'])} | {mil(d['F_curl'])} | "
                   f"{c['n_x_lower']}/5 | {term} | {mil(mean(list(c['L_diff'].values())))}（{c['n_x_lower_L']}/5） | {c['d_seg35']:+.4f} | {c['d_seg4']:+.4f} |")
    decision = doc["decision"]
    main = decision["main_term"]
    main_text = "无" if main["term"] is None else f"{main['term']}，占正差之和的 {main['share']:.2f}"
    out += ["", f"**主判定（A0 − M2V1，σ (a)）**：{decision['judge']}，F_B(A0) 更低 {decision['n_x_lower']}/5 对；主项 {main_text}。"
            "这只说明在这些有限的场上目标函数偏爱谁，不是路径或目标函数为主因的结论。", "",
            "## 4. 翻转边界（X 优于参照所需；其余参数不动）", "",
            "| X − 参照 | 对 | F_B 差 | ν | s_τ | w_I | w_S | 先验整体倍数 |", "|---|---|---|---|---|---|---|---|"]
    for key, c in comps.items():
        for s in shorts:
            r = c["per_pair"][s]
            out.append(f"| {key} | {s} | {mil(r['M'])} | {crossing_text(r['nu'], 'nu')} | {s_tau_text(r['s_tau'])} | {crossing_text(r['w_I'], 'w_I')} | {crossing_text(r['w_S'], 'w_S')} | {crossing_text(r['prior_scale'], 'prior_scale')} |")
    out += ["", "ν 的现值 21.855，s_τ、w_I、w_S、倍数的现值都是 1。\"> x\" 表示参数大于 x 时 X 更优，\"< x\" 表示小于 x 时更优。"
            "\"（超出）\"：临界值在计划写的范围之外（ν ∈ [.7, 874]，s_τ ∈ [1/2, 4]，w_I、w_S、倍数 ∈ [.1, 10]）。", "", "## 5. ν–s_τ 平面上的边界（A0 − M2V1、A1 − M2V1）", ""]
    for key in ("A0-M2V1", "A1-M2V1"):
        c = comps[key]
        grid = c["per_pair"][shorts[0]]["s_grid"]
        out += [f"### {key}", "", "| s_τ | "+" | ".join(shorts)+" |", "|---|"+"---|"*len(shorts)]
        for j, s_value in enumerate(grid):
            out.append(f"| {s_value:.3g} | "+" | ".join(crossing_text(c["per_pair"][s]["nu_of_s"][j], "nu") for s in shorts)+" |")
        out.append("")
        out += ["在 (ν, s_τ) 上再看 w_S 的边界（w_I = 1）：", "", "| ν | s_τ | "+" | ".join(shorts)+" |", "|---|---|"+"---|"*len(shorts)]
        for j, row in enumerate(c["per_pair"][shorts[0]]["w_S_on_nu_s"]):
            out.append(f"| {row['nu']:.3g} | {row['s_tau']:g} | "+" | ".join(crossing_text(c["per_pair"][s]["w_S_on_nu_s"][j], "w_S") for s in shorts)+" |")
        out.append("")
    out += ["## 6. 换一组公共 σ 之后（F_B 差，1e6；每行用同一组 σ 算两个场）", "",
            "| 比较 | σ | "+" | ".join(shorts)+" | X 更低的对数 |", "|---|---|"+"---|"*len(shorts)+"---|"]
    labels = dict(sigma_a="(a) M2V1 种子的 M 步", m_step_M2V1="M2V1 终态的 M 步", m_step_X="X 的 M 步",
                  shared_S_M2V1="结构层共享（M2V1 终态）", shared_S_X="结构层共享（X）")
    for key, rows in doc["sigma_what_if"].items():
        for name, text in labels.items():
            values = [rows[s][name]["M"] for s in shorts]
            out.append(f"| {key} | {text} | "+" | ".join(mil(v) for v in values)+f" | {sum(v < 0. for v in values)}/5 |")
    out += ["", "\"结构层共享\"是这里的假想：全脑共用一个 σ_S²，没有合同 C04-6 的支撑权。", "", "## 7. 逐通道（σ (a)）", ""]
    for key, c in doc["channels"].items():
        rho = c["spearman_dDice_vs_logrho2"]
        out += [f"### {key}", "", "| 通道 | ⟨g²⟩ 占比 | 1/σ² 均值 | ΔE 均值（1e6） | ΔE > 0 对数 | ⟨ρ²⟩ 比（几何均值） | 该组 seg35 Dice 差 | Dice 差 > 0 对数 |",
                "|---|---|---|---|---|---|---|---|"]
        for t in c["rows"]:
            out.append(f"| {t['name']} | {t['mean_fixed_square']:.4f} | {sci(t['mean_inverse_sigma2'])} | {t['mean_dE']/1e6:+.3f} | {t['n_dE_positive']}/5 | {t['geomean_rho2_ratio']:.3f} | "
                       f"{'—' if t['mean_dDice'] is None else format(t['mean_dDice'], '+.4f')} | {'—' if t['n_dDice_positive'] is None else str(t['n_dDice_positive'])+'/5'} |")
        seg4 = "；".join(f"标签 {r['label']} {r['x']:.3f} 对 {r['r']:.3f}" for r in c["seg4_per"])
        out += ["", f"seg4 逐标签（X 对参照）：{seg4}。", f"Dice 差与 log ⟨ρ²⟩ 比的 Spearman（13 组）：{rho['rho']:.2f}（p = {rho['p']:.3g}，描述性）。",
                f"Dice 差 > 0 而 ΔE > 0 的通道：{'、'.join(c['contradictions']) or '无'}。", ""]
    out += ["## 8. (b) 每场一次 M 步的 σ̃_c²（5 对均值，只描述）", "",
            "| 通道 | "+" | ".join(doc["fields"])+" |", "|---|"+"---|"*len(doc["fields"])]
    for name in names:
        out.append(f"| {name} | "+" | ".join(sci(doc["m_step_sigma2"][f][name]) for f in doc["fields"])+" |")
    out += ["", "## 9. 证书（只读已有记录）", "", "| 场 | 状态 | 最小 J 下界（逐对） | closed27 点上最小 J（本次求值） |", "|---|---|---|---|"]
    for f in doc["fields"]:
        r = per[f]
        out.append(f"| {f} | {'、'.join(sorted(set(str(v) for v in r['cert_status'])))} | {', '.join(sci(v) for v in r['cert_min'])} | {', '.join(sci(v) for v in r['closed27_min_J'])} |")
    out += ["", "## 10. 未归一的量（5 对均值）", "", "| 场 | ‖Cu‖² | uᵀK_Qu | Σ(log J)²/τ² | F_vol Gauss27 / closed27（1e6） | ‖Cu‖² 脑内份额 | Σ(log J)²/τ² 脑内份额 | L，C11-1 全式（1e6） |",
            "|---|---|---|---|---|---|---|---|"]
    for f in doc["fields"]:
        r = per[f]
        out.append(f"| {f} | {sci(r['curl_sq'])} | {sci(r['uKu'])} | {sci(r['logvol_sq'])} | {mil(r['F_vol_gauss'])} / {mil(r['F_vol_closed27'])} | {num(r['curl_brain_share'], 3)} | {num(r['logvol_brain_share'], 3)} | {mil(r['L'])} |")
    out += ["", "## 11. 一致性核对", "", "| 对 | M2V1 相对差 | P3 相对差（数据项换算到 P3 的 σ） |", "|---|---|---|"]
    for s in shorts:
        check = doc["consistency"][s]
        out.append(f"| {s} | {check['M2V1']['rel_diff']:.2e} | {check['P3']['rel_diff']:.2e} |")
    out.append("")
    out.append(f"tile 重算的先验三项与 evaluate 的最大相对差 {doc['tile_checks_max']:.2e}。")
    return "\n".join(out)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", default="summary", help="output stem (existing files are not overwritten)")
    ns = parser.parse_args(argv)
    root = Path(ns.root)
    doc = analyze(root)
    for name, text in ((ns.name+".json", json.dumps(doc, indent=1, default=str)+"\n"),
                       (ns.name+".md", markdown(doc)+markdown_revision1(doc))):
        path = root/name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
