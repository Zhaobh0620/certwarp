# Where this snapshot comes from

- **Source.** The authors' private repository, branch `main`, commit `803f3a8bd14c351fddbdfe0a266c31092f78ddec` (2026-11-24).
- **History.** This repository starts a new history with a single commit. The private history is not published: it continues the history of an earlier project.
- **How it was built** (2026-11-24). Only files tracked at that commit were taken, chosen by a white list. Each of the 880 copied files was checked against its git blob id at that commit. 58 of them were then edited for this release (listed below); the other 822 are byte-identical to the private files. `README.md`, `LICENSE`, `SPLITS.md` and this file were written for the snapshot.
- **No data.** No image, label, segmentation, displacement field or model weight is here; see `README.md`.

## What was copied

| Part | Files |
|---|---|
| code: `bayesreg/`, `jd_curl/`, `experiments/`, `scripts/`, `tests/`, `pyproject.toml` | 783 |
| settings, segmenter registries and allow-lists: `configs/` | 15 |
| the two pre-registrations | 2 |
| result documents and tables: `runs/20261119-hb-result/`, `runs/20261122-lpba-result/`, `runs/20261122-ms-d4/`, `runs/20261123-qf/` | 50 |
| the method contract and the two run books | 3 |
| check V1: the note and `runs/20261122-v1/` | 12 |
| the archived diagnostic of the evaluation script | 1 |
| the subject lists: `splits/` | 14 |
| total copied | 880 |

## Files edited for this release

**Wording.** In 53 files, wording that refers to internal tools was rewritten. Replacements per file:

| File | Replacements |
|---|---|
| `archive/seq_vp_3d/docs/performance_diagnosis_3d_E83.md` | 2 |
| `bayesreg/backends/torch64/device.py` | 1 |
| `bayesreg/backends/torch64/model.py` | 1 |
| `bayesreg/cli/run_pair.py` | 2 |
| `bayesreg/eval/cli.py` | 1 |
| `bayesreg/eval/fields.py` | 1 |
| `bayesreg/eval/truth.py` | 1 |
| `bayesreg/seg/unet/data.py` | 1 |
| `configs/bayesreg/README.md` | 1 |
| `docs/bayesreg/01_合同.md` | 98 |
| `docs/bayesreg/notes/LPBA40运行手册_20261120.md` | 12 |
| `docs/bayesreg/notes/V1_浮点前提核对_20261122.md` | 5 |
| `docs/bayesreg/notes/留出集解封与运行手册_20261115.md` | 18 |
| `experiments/d0/make_dev_subjects.py` | 1 |
| `experiments/d0/run_queue.sh` | 2 |
| `experiments/ef/check_repair_gradient.py` | 1 |
| `experiments/eta/sweep.py` | 1 |
| `experiments/hb/hb_configc.py` | 1 |
| `experiments/hb/hb_method.py` | 1 |
| `experiments/hb/hb_review.py` | 1 |
| `experiments/he/he_core.py` | 3 |
| `experiments/hi/hi_core.py` | 2 |
| `experiments/holdout/holdout_common.py` | 3 |
| `experiments/holdout/prep_stage.py` | 1 |
| `experiments/lpba/devcheck/lpdev_common.py` | 2 |
| `experiments/lpba/prep_lpba.py` | 1 |
| `experiments/m2/gpu_smoke.sh` | 1 |
| `experiments/mc0/decide_mc0p.py` | 1 |
| `experiments/nn/prepare_nn.py` | 1 |
| `experiments/pq/pqh_replay.py` | 1 |
| `experiments/pq/run_pq.py` | 3 |
| `experiments/s2/decide_s2.py` | 1 |
| `experiments/seed/decide_seed19.py` | 1 |
| `experiments/spd/profile_native.py` | 1 |
| `experiments/swl/fixed_state_memory.py` | 2 |
| `experiments/sy/decide_sy.py` | 1 |
| `experiments/unv2/chain.sh` | 1 |
| `experiments/unv2/eval_seg.py` | 1 |
| `experiments/xb/v4seg_convention.py` | 1 |
| `runs/20261122-ms-d4/posthoc.md` | 2 |
| `runs/20261122-v1/v1_existing_tests.out` | 1 |
| `runs/20261123-qf/result.md` | 2 |
| `tests/test_bayesreg/_golden/golden_11409eb_sch_records.json` | 1 |
| `tests/test_bayesreg/hb/test_hb_lpba.py` | 3 |
| `tests/test_bayesreg/he/test_he_tools.py` | 1 |
| `tests/test_bayesreg/io_seg/test_code_identity.py` | 4 |
| `tests/test_bayesreg/learn/test_cli.py` | 1 |
| `tests/test_bayesreg/pq/test_pq_core.py` | 1 |
| `tests/test_bayesreg/seg/test_decode.py` | 1 |
| `tests/test_bayesreg/solve/test_run_pair.py` | 1 |
| `tests/test_bayesreg/torch64/test_torch64.py` | 2 |
| `tests/test_bayesreg/torch64/test_torch64_closed27.py` | 1 |
| `tests/test_bayesreg/torch64/test_torch64_corners.py` | 1 |
| total | 202 |

In the `.py` files only comments, docstrings and string constants were changed: for each edited file, the abstract syntax trees before and after are equal once every string constant is replaced by one placeholder. Two edits change code: in `experiments/he/he_core.py` the set `REPO_TOP_LEVEL` of top-level names no longer lists two internal files of the private repository, and in `tests/test_bayesreg/he/test_he_tools.py` the test of that set checks that it is a subset of the listed commit's top level (`>=` for `==`), which is a weaker test than the original equality. In two test files (`tests/test_bayesreg/hb/test_hb_lpba.py`, `tests/test_bayesreg/io_seg/test_code_identity.py`) the string constants are path names used as test inputs; they were renamed in 6 places, and as we read the code under test, it does not depend on these names. In `experiments/d0/run_queue.sh`, `experiments/m2/gpu_smoke.sh` and `experiments/unv2/chain.sh` only comments were changed. One string constant of `experiments/xb/build_frozen_config.py` that refers to internal tools is kept: the script writes it into `configs/frozen/holdout_B_v1_baselines.json`, a pinned file.

**Other edits.** The host name in the first line of the outputs of check V1 became `<host>`; in the method contract, the subject numbers of the earlier project's sealed set H1 became `####` (see `SPLITS.md`).

| File | Edit | Replacements |
|---|---|---|
| `runs/20261122-v1/v1_codegen.out` | host name | 1 |
| `runs/20261122-v1/v1_existing_tests.out` | host name | 1 |
| `runs/20261122-v1/v1_fpstate.out` | host name | 1 |
| `runs/20261122-v1/v1_identity_lpba.out` | host name | 1 |
| `runs/20261122-v1/v1_numerics.out` | host name | 1 |
| `runs/20261122-v1/v1_oasis_spacing.out` | host name | 1 |
| `docs/bayesreg/01_合同.md` | H1 number | 3 |

## Files kept byte-identical (pinned)

These 158 files are byte-identical to the source commit; nothing in them was edited. A file is pinned when it is one of the six files of the table in `README.md`, or when another file here writes its sha256 (in full or its first 8 hex digits) or its git blob id.

- `bayesreg/eval/` (2): `holdout_tet24.py`, `metrics.py`
- `bayesreg/seg/unet/` (2): `cli.py`, `large.py`
- `configs/bayesreg/` (7): `colour_table_a5_v1.json`, `dev_subjects.json`, `lpba40_test_v1_allowlist.json`, `seg4_of_seg35_v1.json`, `segmenters.json`, `segmenters_hs_v1.json`, `structure_groups_v1.json`
- `configs/frozen/` (3): `holdout_B_v1_arm_table.json`, `holdout_B_v1_baselines.json`, `lpba40_test_v1_arm_table.json`
- `docs/bayesreg/notes/` (2): `LPBA40第二测试集预登记_20261119.md`, `留出集预登记_20261107.md`
- `experiments/d0/` (1): `configC_reference.json`
- `experiments/fl/` (1): `run_fl.py`
- `experiments/hs/` (4): `decide_hs.py`, `hs_core.py`, `hs_seg.py`, `run_hs.py`
- `experiments/lpba/` (1): `c_record.py`
- `experiments/p2/` (1): `reference_unet35_v4_ens.json`
- `experiments/split/` (1): `draw_split.py`
- `experiments/xb/harness/` (4): `m_convexadam.py`, `m_fireants.py`, `m_unigradicon.py`, `m_voxelmorph.py`
- `jd_curl/` (9): `__init__.py`, `boundary.py`, `cell_q1.py`, `differences.py`, `energy_comparison.py`, `interpolation.py`, `jacobian_barrier.py`, `types.py`, `validation.py`
- `jd_curl/three_d/` (28): `__init__.py`, `active_set.py`, `barrier.py`, `bernstein_barrier.py`, `boundary.py`, `cell_q1.py`, `certificate_prescreen.py`, `compiled_enclosure.py`, `compiled_fallback.py`, `curvature.py`, `data_weight.py`, `energy_comparison.py`, `fast_checks.py`, `feasibility.py`, `geometry.py`, `grid.py`, `interpolation.py`, `interval.py`, `linearization.py`, `nifti.py`, `poisson.py`, `problem.py`, `residuals.py`, `soa_enclosure.py`, `thread_chunks.py`, `tile_execution.py`, `validation.py`, `work_control.py`
- `runs/20261119-hb-result/` (11): `ablation.csv`, `arms_absolute.csv`, `f1_f2_main_family.csv`, `f3_described.csv`, `f4_pq.csv`, `per_label_dice.csv`, `pq_described.csv`, `result.md`, `summarize.out`, `summarize.py`, `tet24_described.csv`
- `runs/20261122-lpba-result/` (1): `result.md`
- `runs/20261122-ms-d4/` (8): `independent_cornerj.py`, `lpba_per_label_delta.csv`, `minj_by_position.csv`, `posthoc.py`, `posthoc3.py`, `posthoc_jloc.py`, `posthoc_minj.py`, `posthoc_minj_counts.py`
- `runs/20261122-v1/` (1): `v1_oasis_spacing.py`
- `runs/20261123-qf/` (1): `fig_qual_counts.json`
- `runs/20261123-qf/records/` (4): `counts_ants_s.json`, `counts_certwarp_final.json`, `counts_certwarp_pre.json`, `selection.json`
- `scripts/` (1): `folding_metrics.py`
- `splits/bayesreg_v1/` (10): `README.md`, `SHA256SUMS`, `draw_log.json`, `eb_pairs.tsv`, `folds.json`, `folds_full_v2.json`, `holdout_B_pairs.tsv`, `holdout_B_subjects.txt`, `pool_206.txt`, `train_subjects.txt`
- `splits/lpba40_test_v1/` (4): `SHA256SUMS`, `feasibility_pairs.tsv`, `test_pairs.tsv`, `test_subjects.txt`
- `tests/test_bayesreg/` (1): `__init__.py`
- `tests/test_bayesreg/_golden/` (13): `golden_481e33b.json`, `golden_481e33b_def.json`, `golden_7579645_eb.json`, `golden_7579645_eb_def.json`, `golden_7579645_eb_ebd.json`, `golden_8a4d060_def_explicit.json`, `golden_8a4d060_run_pair.json`, `golden_9747553_mem.json`, `golden_9747553_mem_paths.json`, `golden_d046b77_mem_gf.json`, `golden_ddcbfce_swli.json`, `golden_ddcbfce_swli_keys.json`, `make_golden_mem_paths.py`
- `tests/test_bayesreg/aud/` (1): `__init__.py`
- `tests/test_bayesreg/eval/` (1): `__init__.py`
- `tests/test_bayesreg/geom/` (1): `__init__.py`
- `tests/test_bayesreg/hb/` (1): `__init__.py`
- `tests/test_bayesreg/he/` (1): `__init__.py`
- `tests/test_bayesreg/hi/` (1): `__init__.py`
- `tests/test_bayesreg/holdout/` (1): `__init__.py`
- `tests/test_bayesreg/hp/` (1): `__init__.py`
- `tests/test_bayesreg/infer/` (1): `__init__.py`
- `tests/test_bayesreg/io_seg/` (1): `__init__.py`
- `tests/test_bayesreg/learn/` (2): `__init__.py`, `_eb_digest.py`
- `tests/test_bayesreg/lpba/` (1): `__init__.py`
- `tests/test_bayesreg/model/` (2): `__init__.py`, `_logvol_digest.py`
- `tests/test_bayesreg/p1/` (1): `__init__.py`
- `tests/test_bayesreg/p2/` (1): `__init__.py`
- `tests/test_bayesreg/pq/` (1): `__init__.py`
- `tests/test_bayesreg/pqh/` (1): `__init__.py`
- `tests/test_bayesreg/r2/` (1): `__init__.py`
- `tests/test_bayesreg/r3/` (1): `__init__.py`
- `tests/test_bayesreg/r3b/` (1): `__init__.py`
- `tests/test_bayesreg/r4/` (1): `__init__.py`
- `tests/test_bayesreg/seg/` (1): `__init__.py`
- `tests/test_bayesreg/segnet/` (1): `__init__.py`
- `tests/test_bayesreg/segprobe/` (1): `__init__.py`
- `tests/test_bayesreg/sg/` (1): `__init__.py`
- `tests/test_bayesreg/solve/` (8): `__init__.py`, `_jdti_digest.py`, `_mem_digest.py`, `_mem_paths_digest.py`, `_run_pair_digest.py`, `_sch_records.py`, `_swli_digest.py`, `_swli_keys.py`
- `tests/test_bayesreg/torch64/` (1): `__init__.py`
- `tests/test_bayesreg/xb/` (1): `__init__.py`

## White-listed files not taken

These 7 test files write, as subject names, numbers of the sealed set H1 of an earlier project (`SPLITS.md`). Code and tests with such a number are not taken whole. sha256 of the private file at the source commit:

| File | sha256 |
|---|---|
| `tests/test_bayesreg/he/test_he_pipeline.py` | `029e92e43c05b6c8e56031b8abf76ace04c51116ad51246841d5be3827643c7d` |
| `tests/test_bayesreg/io_seg/_helpers.py` | `3b970d93a5438b0b7713e81ecf138211a6c5f2708d2d7b8ee276499e0d898b55` |
| `tests/test_bayesreg/io_seg/test_guard.py` | `7c8cd3625ad948de1b7db748de969440315150de6a44b7dcaf6256f4b3ce6ef5` |
| `tests/test_bayesreg/solve/test_fd2_tools.py` | `143caba48d7667f2b77f0096c45322dc75d700bffd476a1942881f007adbe2c3` |
| `tests/test_bayesreg/solve/test_fd_tools.py` | `e6c97d92cc44c3edf5b308260a30360e3a501119cee94c184d691d8cdb068e74` |
| `tests/test_bayesreg/solve/test_fl_tools.py` | `844ebe1dfe3cb7e5900765ba840f5496f0db8da82434ce877b7af998a17cdfe0` |
| `tests/test_bayesreg/solve/test_fx_tools.py` | `8c4ed329eb0a3d5f3cc0a6198b862e17f258746f2fc2c009521b3702bfe06a95` |

## Code and the commits pinned by the two test-set runs

The OASIS held-out run pinned the commit P1, the LPBA40 run the commit P_L. The tables compare the files at the source commit with each of them by git blob id (equal blob ids mean byte-identical files). A file counted as "identical but for the release edits" is identical to the pin at the source commit and differs from it here only by the edits listed above. "Not in snapshot" counts tracked files of the directory at the source commit that were not copied.

### P1 = `3005a57eb06ac820a620138aa5a09cebebfcdf45` (an ancestor of the source commit)

| directory | files in snapshot | identical | identical but for the release edits | changed | added | not in snapshot | `git diff --shortstat` (whole directory, source commit) |
|---|---|---|---|---|---|---|---|
| `bayesreg/` | 79 | 65 | 6 | 5 | 3 | 0 | 8 files changed, 2893 insertions(+), 43 deletions(-) |
| `jd_curl/` | 37 | 37 | 0 | 0 | 0 | 0 | no difference |
| `experiments/` | 303 | 252 | 20 | 14 | 17 | 0 | 31 files changed, 9338 insertions(+), 264 deletions(-) |
| `scripts/` | 2 | 2 | 0 | 0 | 0 | 0 | no difference |
| `configs/` | 15 | 12 | 1 | 0 | 2 | 0 | 2 files changed, 868 insertions(+) |
| `tests/` | 361 | 308 | 9 | 15 | 29 | 7 | 44 files changed, 10224 insertions(+), 43 deletions(-) |

- `bayesreg/` identical but for the release edits (6): `bayesreg/backends/torch64/device.py`, `bayesreg/backends/torch64/model.py`, `bayesreg/cli/run_pair.py`, `bayesreg/eval/cli.py`, `bayesreg/eval/fields.py`, `bayesreg/seg/unet/data.py`
- `bayesreg/` changed (5): `bayesreg/eval/truth.py` (also edited for the release), `bayesreg/io/code_identity.py`, `bayesreg/io/guard.py`, `bayesreg/io/pair.py`, `bayesreg/io/rejudgment.py`
- `bayesreg/` added (3): `bayesreg/eval/holdout_tet24.py`, `bayesreg/eval/lpba_decide.py`, `bayesreg/eval/lpba_eval.py`
- `experiments/` identical but for the release edits (20): `experiments/d0/make_dev_subjects.py`, `experiments/d0/run_queue.sh`, `experiments/ef/check_repair_gradient.py`, `experiments/eta/sweep.py`, `experiments/he/he_core.py`, `experiments/hi/hi_core.py`, `experiments/holdout/prep_stage.py`, `experiments/m2/gpu_smoke.sh`, `experiments/mc0/decide_mc0p.py`, `experiments/nn/prepare_nn.py`, `experiments/pq/pqh_replay.py`, `experiments/pq/run_pq.py`, `experiments/s2/decide_s2.py`, `experiments/seed/decide_seed19.py`, `experiments/spd/profile_native.py`, `experiments/swl/fixed_state_memory.py`, `experiments/sy/decide_sy.py`, `experiments/unv2/chain.sh`, `experiments/unv2/eval_seg.py`, `experiments/xb/v4seg_convention.py`
- `experiments/` changed (14): `experiments/hb/hb_configc.py` (also edited for the release), `experiments/hb/hb_core.py`, `experiments/hb/hb_method.py` (also edited for the release), `experiments/hb/hb_review.py` (also edited for the release), `experiments/hb/hb_store.py`, `experiments/he/firewall_c.py`, `experiments/he/he_holdout.py`, `experiments/he/run_hx.py`, `experiments/holdout/holdout_common.py` (also edited for the release), `experiments/xb/delete_verified.sh`, `experiments/xb/spy/xb_policy.py`, `experiments/xb/xb_chain.py`, `experiments/xb/xb_core.py`, `experiments/xb/xb_drive.py`
- `experiments/` added (17): `experiments/hi/hi_lpba.py`, `experiments/lpba/admission_dry_run.py`, `experiments/lpba/atropos_lpba.py`, `experiments/lpba/c_record.py`, `experiments/lpba/devcheck/lpdev_a.py`, `experiments/lpba/devcheck/lpdev_b.py`, `experiments/lpba/devcheck/lpdev_c.py`, `experiments/lpba/devcheck/lpdev_common.py` (also edited for the release), `experiments/lpba/dxg_monitor.py`, `experiments/lpba/feas_chain.py`, `experiments/lpba/lpba_arm_table.py`, `experiments/lpba/lpba_arms.py`, `experiments/lpba/lpba_common.py`, `experiments/lpba/make_eval_manifest_lpba.py`, `experiments/lpba/make_lpba40_allowlist.py`, `experiments/lpba/prep_lpba.py` (also edited for the release), `experiments/lpba/实现说明_LP-B.md`
- `configs/` identical but for the release edits (1): `configs/bayesreg/README.md`
- `configs/` added (2): `configs/bayesreg/lpba40_test_v1_allowlist.json`, `configs/frozen/lpba40_test_v1_arm_table.json`
- `tests/` identical but for the release edits (9): `tests/test_bayesreg/_golden/golden_11409eb_sch_records.json`, `tests/test_bayesreg/he/test_he_tools.py`, `tests/test_bayesreg/learn/test_cli.py`, `tests/test_bayesreg/pq/test_pq_core.py`, `tests/test_bayesreg/seg/test_decode.py`, `tests/test_bayesreg/solve/test_run_pair.py`, `tests/test_bayesreg/torch64/test_torch64.py`, `tests/test_bayesreg/torch64/test_torch64_closed27.py`, `tests/test_bayesreg/torch64/test_torch64_corners.py`
- `tests/` changed (15): `tests/test_bayesreg/_p1_decisions.py`, `tests/test_bayesreg/eval/_holdout_world.py`, `tests/test_bayesreg/eval/test_holdout_entry.py`, `tests/test_bayesreg/hi/test_hi_tools.py`, `tests/test_bayesreg/holdout/test_pinned_code.py`, `tests/test_bayesreg/hp/test_hp_code_identity.py`, `tests/test_bayesreg/io_seg/test_code_identity.py` (also edited for the release), `tests/test_bayesreg/io_seg/test_pair.py`, `tests/test_bayesreg/pqh/test_pqh_replay.py`, `tests/test_bayesreg/pqh/test_pqh_to_entry.py`, `tests/test_bayesreg/sg/_census_inventory.py`, `tests/test_bayesreg/sg/test_sg_behaviour.py`, `tests/test_bayesreg/sg/test_sg_census.py`, `tests/test_bayesreg/solve/test_hs_tools.py`, `tests/test_bayesreg/xb/test_xb_chain.py`
- `tests/` added (29): `tests/test_bayesreg/eval/test_holdout_tet24.py`, `tests/test_bayesreg/hb/_lpba_world.py`, `tests/test_bayesreg/hb/test_hb_lpba.py` (also edited for the release), `tests/test_bayesreg/hb/test_hb_lpg.py`, `tests/test_bayesreg/hb/test_hb_lph.py`, `tests/test_bayesreg/lpba/__init__.py`, `tests/test_bayesreg/lpba/_eval_world.py`, `tests/test_bayesreg/lpba/test_c_record.py`, `tests/test_bayesreg/lpba/test_dxg_monitor.py`, `tests/test_bayesreg/lpba/test_lpba_admission.py`, `tests/test_bayesreg/lpba/test_lpba_admission_dry_run.py`, `tests/test_bayesreg/lpba/test_lpba_arm_table.py`, `tests/test_bayesreg/lpba/test_lpba_authorization.py`, `tests/test_bayesreg/lpba/test_lpba_census.py`, `tests/test_bayesreg/lpba/test_lpba_code_identity.py`, `tests/test_bayesreg/lpba/test_lpba_decide.py`, `tests/test_bayesreg/lpba/test_lpba_devcheck.py`, `tests/test_bayesreg/lpba/test_lpba_entries.py`, `tests/test_bayesreg/lpba/test_lpba_eval.py`, `tests/test_bayesreg/lpba/test_lpba_eval_census.py`, `tests/test_bayesreg/lpba/test_lpba_eval_lph.py`, `tests/test_bayesreg/lpba/test_lpba_feas_chain.py`, `tests/test_bayesreg/lpba/test_lpba_splits.py`, `tests/test_bayesreg/lpba/test_lpba_spy_no_dir_fd.py`, `tests/test_bayesreg/lpba/test_lpbab_census.py`, `tests/test_bayesreg/lpba/test_make_eval_manifest_lpba.py`, `tests/test_bayesreg/xb/test_xb_lpba.py`, `tests/test_bayesreg/xb/test_xb_lpg.py`, `tests/test_bayesreg/xb/test_xb_lph.py`

### P_L = `d839b23f39b27a8a86eaa7eb22e1dc5ea527a53e` (an ancestor of the source commit)

| directory | files in snapshot | identical | identical but for the release edits | changed | added | not in snapshot | `git diff --shortstat` (whole directory, source commit) |
|---|---|---|---|---|---|---|---|
| `bayesreg/` | 79 | 72 | 7 | 0 | 0 | 0 | no difference |
| `jd_curl/` | 37 | 37 | 0 | 0 | 0 | 0 | no difference |
| `experiments/` | 303 | 277 | 26 | 0 | 0 | 0 | no difference |
| `scripts/` | 2 | 2 | 0 | 0 | 0 | 0 | no difference |
| `configs/` | 15 | 14 | 1 | 0 | 0 | 0 | no difference |
| `tests/` | 361 | 350 | 11 | 0 | 0 | 7 | no difference |

- `bayesreg/` identical but for the release edits (7): `bayesreg/backends/torch64/device.py`, `bayesreg/backends/torch64/model.py`, `bayesreg/cli/run_pair.py`, `bayesreg/eval/cli.py`, `bayesreg/eval/fields.py`, `bayesreg/eval/truth.py`, `bayesreg/seg/unet/data.py`
- `experiments/` identical but for the release edits (26): `experiments/d0/make_dev_subjects.py`, `experiments/d0/run_queue.sh`, `experiments/ef/check_repair_gradient.py`, `experiments/eta/sweep.py`, `experiments/hb/hb_configc.py`, `experiments/hb/hb_method.py`, `experiments/hb/hb_review.py`, `experiments/he/he_core.py`, `experiments/hi/hi_core.py`, `experiments/holdout/holdout_common.py`, `experiments/holdout/prep_stage.py`, `experiments/lpba/devcheck/lpdev_common.py`, `experiments/lpba/prep_lpba.py`, `experiments/m2/gpu_smoke.sh`, `experiments/mc0/decide_mc0p.py`, `experiments/nn/prepare_nn.py`, `experiments/pq/pqh_replay.py`, `experiments/pq/run_pq.py`, `experiments/s2/decide_s2.py`, `experiments/seed/decide_seed19.py`, `experiments/spd/profile_native.py`, `experiments/swl/fixed_state_memory.py`, `experiments/sy/decide_sy.py`, `experiments/unv2/chain.sh`, `experiments/unv2/eval_seg.py`, `experiments/xb/v4seg_convention.py`
- `configs/` identical but for the release edits (1): `configs/bayesreg/README.md`
- `tests/` identical but for the release edits (11): `tests/test_bayesreg/_golden/golden_11409eb_sch_records.json`, `tests/test_bayesreg/hb/test_hb_lpba.py`, `tests/test_bayesreg/he/test_he_tools.py`, `tests/test_bayesreg/io_seg/test_code_identity.py`, `tests/test_bayesreg/learn/test_cli.py`, `tests/test_bayesreg/pq/test_pq_core.py`, `tests/test_bayesreg/seg/test_decode.py`, `tests/test_bayesreg/solve/test_run_pair.py`, `tests/test_bayesreg/torch64/test_torch64.py`, `tests/test_bayesreg/torch64/test_torch64_closed27.py`, `tests/test_bayesreg/torch64/test_torch64_corners.py`

The snapshot holds one version of each file, the one at the source commit with the edits above, which for these directories is the version of P_L. The contents that the files listed as changed had at P1, the commit pinned by the OASIS held-out run, are not in this snapshot.
