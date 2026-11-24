# CertWarp

Code, pre-registrations and result documents for the paper "Certificate-Gated Brain MRI Registration:
Certified Output from an Uncertified Optimizer" (working title; the title of the submitted paper governs).

**Name.** CertWarp is the method that the pre-registrations, the method contract, the two test-set result
documents, the run books and the code in `experiments/` call `Hx24-RF05-F1200-v4`; the post hoc files in
`runs/20261122-ms-d4/` and `runs/20261123-qf/` also use the name CertWarp. `bayesreg` is only an internal code name, kept
as the name of the Python package. "Bayesian tissue-aware registration" in `pyproject.toml` and in the
docstring of the package is the name the project had at its start. CertWarp is not a Bayesian registration
method. It contains one posterior predictive quality-control component (PQ); the ensemble that PQ uses is
an approximate posterior, and that reading is debated. See section 1.4 of the OASIS result document.

**Language.** These documents are written in Chinese and are published as written: the method contract, the
two pre-registrations, the two result documents, the two run books, the check note V1, the post hoc note
(`runs/20261122-ms-d4/posthoc.md`), the note on the qualitative figure (`runs/20261123-qf/result.md`), the
archived diagnostic, the implementation note `experiments/lpba/实现说明_LP-B.md` and the README files of
`configs/bayesreg/` and `splits/bayesreg_v1/`. The verdict column of
`runs/20261119-hb-result/f1_f2_main_family.csv`, parts of `runs/20261119-hb-result/summarize.out`, fields of
the two arm tables (`configs/frozen/holdout_B_v1_arm_table.json`, `configs/frozen/lpba40_test_v1_arm_table.json`)
and a few fields of `configs/frozen/holdout_B_v1_baselines.json` and of `colour_table_a5_v1.json`,
`segmenters.json`, `segmenters_hs_v1.json`, `segmenters_v2.json`, `structure_groups_v1.json` and
`sv_fstar_v1.json` in `configs/bayesreg/` are in Chinese too. Code comments are mostly in English, some in
Chinese.

## Contents

| Path | What it holds |
|---|---|
| `bayesreg/` | the package: input guards, segmenter, solver parts, evaluation and decision code |
| `jd_curl/` | low-level library: cell-wise Jacobian certificate, Q1 geometry, interpolation, grids |
| `experiments/` | entry points and tools of the development runs and of the two test-set runs |
| `scripts/` | Learn2Reg-style evaluation, folding metrics |
| `tests/` | tests of the package (`test_bayesreg/`) and of the library (`three_d/`) |
| `configs/` | settings, segmenter registries and allow-lists; the machine-readable arm tables are in `configs/frozen/` |
| `splits/` | the subject lists of the OASIS split and of the LPBA40 test set (see `SPLITS.md`) |
| `docs/bayesreg/01_合同.md` | the method contract |
| `docs/bayesreg/notes/` | the two pre-registrations, the two run books, the floating-point check V1 |
| `runs/20261119-hb-result/` | OASIS held-out set: result document, summary script, tables |
| `runs/20261122-lpba-result/` | LPBA40 test set: result document |
| `runs/20261122-ms-d4/` | post hoc, descriptive readouts: scripts, outputs, tables (revisions 3 to 5 of `posthoc.md` with `posthoc3`, `posthoc_minj`, `posthoc_minj_counts` and the second implementation `independent_cornerj`) |
| `runs/20261122-v1/` | scripts and outputs of check V1 |
| `runs/20261123-qf/` | the counts behind the qualitative figure (one development pair): result document and records |
| `archive/seq_vp_3d/docs/performance_diagnosis_3d_E83.md` | archived diagnostic: the per-case metric functions against the Learn2Reg code |
| `SPLITS.md`, `SNAPSHOT.md` | the subject lists; where this snapshot comes from and what was edited |

## Pre-registrations and results

Each test set was run once per pair and arm, after its pre-registration was fixed. The files below are
copied byte for byte. Each result document quotes the first 8 hex digits of the sha256 of its
pre-registration and of its arm table; the full values, and those of the result documents, are here.

| File | sha256 |
|---|---|
| `docs/bayesreg/notes/留出集预登记_20261107.md` (OASIS held-out set) | `df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493` |
| `configs/frozen/holdout_B_v1_arm_table.json` | `69cb1ddf3e2006116c3d2237f3a4912e68a7238f2ce813f7d67723e5eeb8d661` |
| `runs/20261119-hb-result/result.md` | `8a61f36e8fecd542a2ab748dd5654f7180effa96125bface2781595d2a02a954` |
| `docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md` (LPBA40 test set) | `06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643` |
| `configs/frozen/lpba40_test_v1_arm_table.json` | `c47bbbe25d748fa14115c251ea1ceb93514eae9136ae8319ba669eb5453a29ad` |
| `runs/20261122-lpba-result/result.md` | `8104dc342e9b095a21d25d480c4a9668c923c1efc6ead1571a3cf36b3f05f706` |

Numbers copied from the two result documents:

- OASIS held-out set, 50 pairs: Dice of the method on 35 / 4 structures (seg35 / seg4) .8782 / .9225;
  judged superior to each of the five main comparators on both. "Superior" means superior to these
  baselines in their frozen configurations only: the hyper-parameters of the method were chosen on the
  development set, the external baselines were not tuned. Decision certificate passed on 50/50 outputs.
- LPBA40, 17 pairs: Dice of the method on 54 / 56 structures (lpba54 / lpba56) .6747 / .6804; on both,
  judged inferior to three of the five comparators, inconclusive against one, equivalent to one. The
  pre-registered guard on SDlogJ holds against none of the five comparators: the official SDlogJ of the
  method is higher than theirs by .015 to .038. Decision certificate passed on 17/17 outputs.

## Not included

- Images, labels and segmentations: the data use agreements of OASIS and LPBA40 do not allow
  redistribution. Displacement fields and model weights are left out too; that they fall under the
  agreements is the authors' reading, not the wording of the agreements.
- Another 7 test files contain subject numbers of the sealed sets of an earlier project and are not included;
  another 31 tests import one of them, `tests/test_bayesreg/io_seg/_helpers.py`, and so do 3 scripts that are
  not tests (`experiments/{eta,m2,m2v1}/dry_inputs.py`); 3 scripts that make golden files
  (`tests/test_bayesreg/_golden/make_golden_{run_pair,sch,swli}.py`) take that file from a commit.
  `SNAPSHOT.md` lists the 7.
- The code of the OASIS run as it was pinned. That run pinned an earlier commit (P1). 19 files of
  `bayesreg/` and `experiments/` and 15 test files were changed after P1, and their P1 contents are not
  in this snapshot. `SNAPSHOT.md` names them.
- The files that the documents and code comments cite from the internal repository: the decision log,
  review records, task briefs, other run directories. There are many such references and they cannot be
  followed here.
- The manuscript and the working notes of the project.

**Running the code.** This snapshot has not been tested as a runnable release; we have not run code from this
snapshot.

## How this repository was produced

This is a one-time snapshot exported from the authors' internal repository by a white list; the internal
history is not published. The documents are working files of the project and keep its internal shorthand
(BZ is the initials of the corresponding author) and references to internal files that are not published.
For this release, wording that refers to internal tools was changed in 53 files (listed in `SNAPSHOT.md`);
the six files of the table above are the versions at the source commit (sha256 in the table): the
pre-registrations and arm tables are byte-identical to the versions that were adopted, and the result documents
include revisions appended after they were finalised (OASIS: revisions 4, 4a, 4b and 5; LPBA40: revision 2).

## License

MIT, see `LICENSE`.
