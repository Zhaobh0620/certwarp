"""CVG prep: the coarse-level inputs of one development pair within a 4 GB memory limit (diagnostic wrapper).

    CVG_CODE=<snapshot> python -B experiments/cvg/prep.py --pair <stage>/<pair> --out <cache.npz> [--levels 0,1]

bayesreg.cli.run_pair.prepare_channels peaks at ~7.8 GB for the structure layer (eta probe timing.json), above this
work package's 4G unit limit.  This script calls the same functions with the same arguments -- load_pair (subject
guard, label firewall: only fixed / moving), StructureLayer (registry, D24, read-back; structure_infer "never"),
structure_ptilde, load_tissue_layer, simplex_with_background, brain_mask, normalize_intensity, segmentation_qc,
tissue_channels, eta_temporary -- but side by side (structure fixed, structure moving, tissue fixed, tissue moving),
dropping each native array once its coarse levels are made, and keeps only the levels asked for.  The coarse levels
are channel_pyramid's (bayesreg.seg.channels, C03.6): every array through jd_curl's downsample_array(...,
labels=False) onto four_level_grids, every simplex renormalized over its own channels, mask_ell =
1[downsampled mask >= .5], the N_q gate (rtol .1).  The native C03 gates of channel_pyramid (simplex sum 1e-12, floor)
are checked too.  Values are therefore those of run_pair's pyramid (checked downstream against hyper.json of the
smoke run H2: G0 and the level-start sigma^2 of level 0).
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401  (sets sys.path to the snapshot)

import numpy as np  # noqa: E402

from jd_curl.three_d.nifti import downsample_array  # noqa: E402
from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.grids import four_level_grids  # noqa: E402
from bayesreg.io import load_pair  # noqa: E402
from bayesreg.seg import (brain_mask, normalize_intensity, segmentation_qc, simplex_with_background,  # noqa: E402
                          tissue_channels)
from bayesreg.seg.channels import FLOOR_TOL, SIMPLEX_SUM_TOL  # noqa: E402
from bayesreg.seg.structure_layer import DEFAULT_GROUPING, SEGMENTER_ID, StructureLayer  # noqa: E402
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH  # noqa: E402
from bayesreg.solve import eta_temporary  # noqa: E402

_RAISE = dict(over="raise", invalid="raise", divide="raise")
EPS = 1e-3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", default="0,1")
    parser.add_argument("--data-root", default=str(common.DATA_ROOT))
    parser.add_argument("--tissue-root", default=str(rp.DEFAULT_TISSUE_ROOT))
    parser.add_argument("--structure-root", default="/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists (nothing is overwritten)")
    levels = tuple(int(x) for x in args.levels.split(","))
    tick = time.perf_counter()
    timing = {}
    pair_path = rp.resolve_pair(args.data_root, args.pair)
    pair = load_pair(pair_path, allowlist=rp.DEFAULT_ALLOWLIST)
    grids = four_level_grids(pair.grid)
    native = pair.grid
    sargs = SimpleNamespace(structure_root=args.structure_root, segmenter_registry=str(rp.REPO_REGISTRY),
                            allowlist=str(rp.DEFAULT_ALLOWLIST), segmenter_models_root=str(rp.DEFAULT_SEGMENTER_OUT),
                            structure_image_root=str(rp.DEFAULT_STRUCTURE_IMAGE_ROOT),
                            structure_groups=str(DEFAULT_GROUPS_PATH), structure_grouping=DEFAULT_GROUPING,
                            segmenter_id=SEGMENTER_ID, structure_infer="never", structure_device=None,
                            structure_synthetic=False, structure_registry_staging=False)
    source = rp.structure_source(sargs, command=("python", "-B", "experiments/cvg/prep.py", "--pair", args.pair))
    layer = StructureLayer(pair, source, log=print)
    arrays, records = {}, dict(pair=args.pair, pair_path=str(pair_path), pair_provenance=pair.provenance,
                               levels=list(levels), structure_source=source.record())

    def down(array, ell):
        values, coarse, meta = downsample_array(array, native, grids[ell].shape, labels=False)
        if coarse.grid_id != grids[ell].grid_id:
            raise RuntimeError(f"level {ell}: downsample grid differs from four_level_grids")
        return values

    def simplex_levels(ptilde, name):
        deviation = float(np.max(np.abs(ptilde.sum(axis=0)-1)))           # channel_pyramid's native gates
        if deviation > SIMPLEX_SUM_TOL:
            raise RuntimeError(f"{name}: |sum - 1| = {deviation:.3e}")
        if float(ptilde.min()) < EPS/(1+ptilde.shape[0]*EPS)-FLOOR_TOL:
            raise RuntimeError(f"{name}: below the floor")
        for ell in levels:
            stacked = np.stack([down(channel, ell) for channel in ptilde])
            with np.errstate(**_RAISE):
                arrays[f"{name}_L{ell}"] = stacked/stacked.sum(axis=0)

    # structure layer, one side at a time (the largest arrays: K_S + 1 = 14 native channels per side)
    names = {}
    for side in ("fixed", "moving"):
        ptilde, accepted, record = rp.structure_ptilde(layer, side)
        names[side] = accepted["class_names"]
        records[f"structure_{side}"] = dict(record, n_channels=int(ptilde.shape[0]))
        simplex_levels(ptilde, f"structure_{side}")
        del ptilde
        timing[f"structure_{side}"] = time.perf_counter()-tick
    if names["fixed"] != names["moving"]:
        raise RuntimeError("structure layer: class_names differ")
    # tissue layer, intensity normalization, mask (fixed side), eta_tmp
    tissue_names = {}
    mask_f = None
    for side in ("fixed", "moving"):
        p, accepted, record = rp.load_tissue_layer(pair, side, args.tissue_root)
        tissue_names[side] = accepted["class_names"]
        pbar, simplex = simplex_with_background(p)
        del p
        mask, mask_prov = brain_mask(pbar)
        image, norm = normalize_intensity(pair.fixed if side == "fixed" else pair.moving, pbar, mask, accepted)
        records[f"tissue_{side}"] = dict(record=record, simplex=simplex, mask=mask_prov, normalization=norm)
        if side == "fixed":
            records["qc"] = segmentation_qc(pbar, mask)
        pt, smoothing = tissue_channels(pbar)
        del pbar
        records["smoothing"] = smoothing
        for ell in levels:
            arrays[f"{side}_L{ell}"] = down(image, ell)
        simplex_levels(pt, f"tissue_{side}")
        if side == "fixed":
            eta = eta_temporary(pt, mask, native)
            records["eta_tmp"] = eta.record()
            mask_f = mask
            n_b = int(mask.sum())
            for ell in levels:
                level_mask = down(mask.astype(np.float64), ell) >= .5
                n_q = float(np.sum(grids[ell].node_weights()*level_mask)/native.omega)
                if abs(n_q-n_b)/n_b > .1:
                    raise RuntimeError(f"level {ell}: N_q gate")
                arrays[f"mask_L{ell}"] = level_mask
                records.setdefault("n_q", {})[str(ell)] = dict(n_q=n_q, native_n_b=n_b, relative=abs(n_q-n_b)/n_b)
        del pt, image, mask
        timing[f"tissue_{side}"] = time.perf_counter()-tick
    if tissue_names["fixed"] != tissue_names["moving"]:
        raise RuntimeError("tissue layer: class_names differ")
    del mask_f
    meta = dict(records, grid_ids={str(ell): grids[ell].grid_id for ell in levels},
                shapes={str(ell): list(grids[ell].shape) for ell in levels}, native_omega=native.omega,
                eta_tmp=records["eta_tmp"], code=common.code_record(), timing=timing,
                note="CVG prep (experiments/cvg/prep.py): run_pair's prepare_channels functions side by side, coarse "
                "levels only; diagnostic input, not a run_pair output")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta_json=np.array(common.json.dumps(meta, default=common._default)), **arrays)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print(f"wrote {out}: {sorted(arrays)}; eta_tmp {records['eta_tmp']}; {time.perf_counter()-tick:.1f} s; "
          f"peak RSS {peak:.0f} MB")


if __name__ == "__main__":
    main()
