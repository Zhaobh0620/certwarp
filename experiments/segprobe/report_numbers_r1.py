"""SEGP, the revision after docs/bayesreg/reviews/SEGP_结果_复核_20261105.md: the numbers it adds to runs/20261105-segprobe/result.md. Arithmetic on files that exist already (the results' JSON and runs/20261103-n0/n0_main.json); no image, truth, posterior or field is opened. Printed; the output is committed as report_numbers_r1.out.
Every line is a model on development subjects (0440 was selected for its low Dice), not a corrected truth.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path("/home/rmcfz/seq_vp_tag_results/segprobe_v1")
REPO = Path(__file__).resolve().parents[2]
TARGET = "OASIS_OAS1_0440_MR1"
ACROSS = ("anatomy.native_volume_mm3", "anatomy.cortex_share", "anatomy.cortex_over_wm", "anatomy.thickness_proxy", "anatomy.ventricle_share", "anatomy.linear_scale_of_alignment", "intensity.cnr_wm_gm", "intensity.noise_wm_high_pass_sd_over_wm")


def read(row, path):
    node = row
    for key in path.split("."):
        node = node[key]
    return node


def main():
    features = json.loads((ROOT/"p4/features.json").read_text())
    follow = json.loads((ROOT/"p4_followup/followup.json").read_text())
    leverage = json.loads((ROOT/"p4_followup/leverage.json").read_text())
    image_fit = json.loads((ROOT/"p4_followup/image_fit.json").read_text())
    blocks = json.loads((ROOT/"p4_blocks/blocks.json").read_text())
    rows, subjects = features["rows"], features["subjects"]
    others = [s for s in subjects if s != TARGET]

    print("across the other 19, seg4 of v1 (the seg35 column is in p4_followup/summary.md):")
    y = np.array([rows[s]["v1"]["seg4"] for s in others])
    for path in ACROSS:
        x = np.array([read(rows[s], path) for s in others], dtype=np.float64)
        rho, p = scipy_stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        print(f"  {path}: Spearman {rho:+.3f}, p {p:.3f}, line at 0440 {slope*read(rows[TARGET], path)+intercept:.4f}; 0440 {rows[TARGET]['v1']['seg4']:.4f}, mean of the 19 {y.mean():.4f}")

    print("Atropos votes, voxel counts, 0440 against the mean of the other 19:")
    arbiter = {s: follow["rows"][s]["arbiter"] for s in subjects}

    def votes(keys_for_pred, keys_for_truth):
        own_p, own_t = sum(arbiter[TARGET][k]["for_a"] for k in keys_for_pred), sum(arbiter[TARGET][k]["for_b"] for k in keys_for_truth)
        other_p, other_t = np.mean([sum(arbiter[s][k]["for_a"] for k in keys_for_pred) for s in others]), np.mean([sum(arbiter[s][k]["for_b"] for k in keys_for_truth) for s in others])
        return own_p, own_t, other_p, other_t

    for name, keys in (("WM / cortex inside the brain", ("pred_cortex_truth_wm", "pred_wm_truth_cortex")), ("cortex / background", ("pred_cortex_truth_background", "pred_background_truth_cortex"))):
        own_p, own_t, other_p, other_t = votes(keys, keys)
        extra_p, extra_t = own_p-other_p, own_t-other_t
        share = 100*extra_p/(extra_p+extra_t) if extra_p > 0 and extra_t > 0 else float("nan")
        print(f"  {name}: with the prediction {own_p} (others' mean {other_p:.0f}, extra {extra_p:+.0f}); with the truth {own_t} (others' mean {other_t:.0f}, extra {extra_t:+.0f}); share of the extra with the prediction {share:.1f}%")
    for key in ("pred_cortex_truth_wm", "pred_wm_truth_cortex", "pred_cortex_truth_background", "pred_background_truth_cortex"):
        values = [arbiter[s][key]["for_b"] for s in others]
        print(f"  {key}: with the truth {arbiter[TARGET][key]['for_b']}, the others {min(values)} to {max(values)}")

    print("what the unlabelled voxels carry of the truth's proportions: the extra truth background -> cerebral cortex counted as cortex:")
    a = rows[TARGET]["anatomy"]
    confusion_extra = 58713-float(np.mean([rows[s]["labels_v1"]["confusions_normalized_intensity"]["truth_background_pred_cortex"]["n"] for s in others]))
    n_cortex, n_brain, n_wm = a["n_cortex"]+confusion_extra, a["n_brain"]+confusion_extra, a["n_wm"]
    contact = a["n_cortex"]/a["thickness_proxy"]
    print(f"  extra {confusion_extra:.0f}: cortex share {n_cortex/n_brain:.4f}, WM share {n_wm/n_brain:.4f}, cortex/WM {n_cortex/n_wm:.3f}, thickness proxy {n_cortex/contact:.3f}")

    print("the brain size with the predicted foreground (native volume x foreground pred / truth), smallest three:")
    sized = sorted((rows[s]["anatomy"]["native_volume_mm3"]*follow["rows"][s]["volumes"]["foreground_pred_over_truth"], s) for s in subjects)
    print("  ", [(round(v), s[11:15]) for v, s in sized[:3]])

    print("the WM surface offset, truth - prediction:")
    gap = {s: rows[s]["labels_v1"]["truth_crossing"]-rows[s]["labels_v1"]["pred_crossing"] for s in subjects}
    print(f"  0440 {gap[TARGET]:+.3f}; the other 19 {min(gap[s] for s in others):+.3f} to {max(gap[s] for s in others):+.3f}; contact voxels {contact:.0f}; contact x gap {contact*gap[TARGET]:.0f}; WM -> cortex net 45931 - 19249 = {45931-19249}")

    print("the intensity model, absolute NCC at s = 0:")
    for labels in ("truth", "pred"):
        for box in ("inferior", "middle", "superior"):
            s = image_fit["stats"][f"{labels}.{box}.ncc_at_0"]
            print(f"  {labels} {box}: {s['value']:.4f}, others {s['others_min']:.4f} to {s['others_max']:.4f}, rank {s['rank_of_20_from_low']}")

    print("the gaps on the other 19, the largest gain when counted right:")
    r = leverage["rows"]
    print(f"  seg35 {max(r[s]['gaps']['seg35']-r[s]['base']['seg35'] for s in others):+.4f}, seg4 {max(r[s]['gaps']['seg4']-r[s]['base']['seg4'] for s in others):+.4f}")
    t = r[TARGET]
    cerebral = t["n_gaps"]-t["gaps_cerebellar"]
    print(f"  0440 cerebral {cerebral}, cerebellar {t['gaps_cerebellar']}; the other 19 cerebral {min(r[s]['n_gaps']-r[s]['gaps_cerebellar'] for s in others)} to {max(r[s]['n_gaps']-r[s]['gaps_cerebellar'] for s in others)}")
    o = leverage["others_mean"]
    print(f"  the line of a segmenter that only misses the blocks (the mean of the other 19 minus what the blocks carry on 0440): {o['base']['seg35']-(t['thick']['seg35']-t['base']['seg35']):.4f} / {o['base']['seg4']-(t['thick']['seg4']-t['base']['seg4']):.4f}")

    unknown = sum(c["voxels"] for c in blocks["composition"] if c["truth"] == "Unknown")
    cerebellar = sum(c["voxels"] for c in blocks["composition"] if c["truth"] == "Unknown" and "Cerebellum-Cortex" in c["pred"])
    print(f"blocks: truth Unknown in the 12 pairs stored {unknown} of {blocks['n_thick']}: {100*unknown/blocks['n_thick']:.1f}% (a lower bound); predicted cerebellar cortex {cerebellar}: {100*cerebellar/blocks['n_thick']:.1f}%")

    n0 = json.loads((REPO/"runs/20261103-n0/n0_main.json").read_text())
    index = next(i for i, s in enumerate(n0["subjects"]) if "0440" in str(s))        # N0 keeps its own spelling of the ids
    print(f"N0 keys of macro: {sorted(n0['macro'])}; subject at that index: {n0['subjects'][index]}")
    base_key = next(k for k in n0["macro"] if k not in ("tta", "flip") and len(n0["macro"][k]) == len(n0["subjects"]))
    rest = [i for i in range(len(n0["subjects"])) if i != index]
    for name, table in (("seg35", n0["macro"]), ("seg4", n0["macro4"])):
        base, flip, tta = (np.array(table[k], dtype=np.float64) for k in (base_key, "flip", "tta"))
        print(f"N0 (g), {name} (unflipped = {base_key!r}): 0440 flip only {flip[index]-base[index]:+.4f}, averaged {tta[index]-base[index]:+.5f}; the other 19, median flip only {np.median((flip-base)[rest]):+.4f}, averaged {np.median((tta-base)[rest]):+.4f}, positive {int(((tta-base)[rest] > 0).sum())}/19")


if __name__ == "__main__":
    main()
