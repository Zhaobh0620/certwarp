# Subject lists

The subject lists of this project are in this repository, under `splits/`. The files are those of the private
repository at the commit named in `SNAPSHOT.md`, unchanged. Line counts are `wc -l`.

## OASIS split of this project (`splits/bayesreg_v1/`)

| File | Lines | sha256 | Content |
|---|---|---|---|
| `pool_206.txt` | 206 | `c4b4257b231a3ba17ef1a37c4c7964a2d1a8674e0c476d33811bd1f7613ed8fa` | candidate pool, one subject per line |
| `holdout_B_subjects.txt` | 100 | `8ed5c06eb782db3a04afdb885c3e8047fc299c632d00cf74006a70b14efe0c50` | held-out subjects |
| `holdout_B_pairs.tsv` | 51 | `8088916b103a4ddd8b5589164a62ea6048bac1c1799fd367f1deec154151a9ee` | header + the 50 held-out pairs (fixed, moving), in draw order |
| `train_subjects.txt` | 106 | `81aefd1a52c1676e6450232cd76bae88dc977e63b533bfbc55b50f4aeda1280a` | training subjects of the segmenter |
| `folds.json` | 430 | `98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245` | fold (A/B) and role (train/calib) of each training subject |
| `folds_full_v2.json` | 234 | `4573b7900b5ae4e7e3e02ff41c0a756f2ff48a8f84a0e7743965ec17b14558dc` | derived from `folds.json`: per fold, its calibration subjects and every other training subject |
| `eb_pairs.tsv` | 31 | `65431f6e5ef61062224d09df0ea61eccf898635c172470cef7cba47deab62b22` | header + 30 same-fold training pairs |
| `draw_log.json` | 102 | `90a34dc3a93e73013d09dd0677e78ad2fe79b53679ddb94dcd2ead81b4610567` | log of the draw (no subject number; machine paths) |
| `README.md` | 22 | `85ad280a86769b7b3288be264e190e28d09843005829b5ba2d5806a948d386f7` | description of the directory (in Chinese) |
| `SHA256SUMS` | 8 | `5901662875edeb039521aba4e7b83652678e9c7520b47121333eee49fc6dc817` | sha256 of the files above but `folds_full_v2.json` |

**The draw.** One run of `experiments/split/draw_split.py` (sha256
`ba7443ded09ad32df380fb55ed0d49415530c7e30380585644413dd2c283bcbe`) on 2026-10-30, committed to the
private repository as `044426fd`. Seed `20261030`; `numpy.random.default_rng(seed)`, calls in order
`permutation(206)`, `permutation(106)`, `permutation(53)` four times; numpy 2.0.2, Python 3.10.20.
From the pool of 206 subjects it drew 100 held-out subjects in 50 pairs and 106 training subjects in
two folds of 53, each with 8 calibration subjects, and 15 training pairs per fold.

## LPBA40 test set (`splits/lpba40_test_v1/`)

| File | Lines | sha256 | Content |
|---|---|---|---|
| `test_subjects.txt` | 34 | `161f379290b6d3b0e52118570d9bc23e14197e34e77e5263c87f22a84fd5deb9` | the 34 test subjects, one per line |
| `test_pairs.tsv` | 18 | `1b0c27b54384ac15f010f0c9ebaf6577d77386e9166ce4b04e21da05ab289038` | header + the 17 test pairs |
| `feasibility_pairs.tsv` | 3 | `2f20a742942cb9538b7a60f1502db9f932b6b606bfbcad5a3e0a733366829ff7` | header + 2 pairs used only to check that the pipeline runs, without labels |
| `SHA256SUMS` | 3 | `dcb529b8b8252f39aabf8ce535f73c538f8dc46fbc1fa036f7e99ad4d5ae10f0` | sha256 of the three lists |

The design of the LPBA40 pairs is written in the LPBA40 pre-registration
(`docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md`), which is published unchanged.

## Sealed sets of an earlier project

The rules of this project also kept sealed the held-out sets fixed in an earlier project of the same authors.
The lists of that project's sealed sets H1 and H4 (OASIS) are not in this repository.
H2 has no list file: it is LPBA40 subjects S37 to S40. H3 is all 18 subjects of IBSR18; this project did not use
IBSR18.

## Other subject numbers in this repository

- The development pairs (Learn2Reg OASIS validation pairs, subjects 0438 to 0457) and the LPBA40 subject numbers
  appear in the pre-registrations, result documents, code and tests as they were written. They are not masked:
  the sha256 of a pre-registration is the evidence that it was fixed before the run, and any edit would break it.
- 48 further OASIS numbers that the earlier project used for development (from its tuning set T0, its set E11
  and its analyses) appear as subject names, pair names, quoted four-digit strings or run labels.
- 35 four-digit numbers that no OASIS subject has (0000, or above 0457) appear in the synthetic subject names of
  tests and dry runs.
- `IBSR_<nn>` names: 17 occurrences in 8 test files.
- No subject number of H1 or H4 is written in this repository as a subject name, a pair name or a four-digit
  number. Five places in code and tests build synthetic subject names from integer ranges (for example every
  number from 1 to 457); such a range covers listed numbers without naming any of them.

The earlier project used the 414 subjects of the neurite-OASIS release. Of these 414, the subjects written in this
repository are exactly those outside the union of that project's two sealed sets H1 and H4, so the union can be
inferred; which subjects are in H1 and which are in H4 is not given here. Both sets have already been used for the
evaluation of that project's paper.
