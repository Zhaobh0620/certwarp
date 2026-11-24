"""Structure-layer segmenter P_S: self-trained 3D U-Net ensemble with 2-fold cross-fitting and
temperature calibration (docs/bayesreg/04_分割器.md; contract C03.1-2/3, C20 D24, D25, D37).

Modules (torch is imported only by the modules that need it):
  groups     structure groups (configs/bayesreg/structure_groups_v1.json), class -> group sums
  model      UNet3D (4 down-sampling stages, base 16, InstanceNorm + LeakyReLU)
  data       folds file, label guard, image/label reading, normalization, augmentation
  train      Dice + CE training of one (fold, seed) member with early stopping on calib
  large      the profile v3_large: base 32, patches, deep supervision, left-right flip with the labels swapped (work package UL)
  calibrate  temperature per member on its fold's calib brain voxels -> member.pt
  registry   member checks against folds / allowlist / records; fold models in the registry
             (schema bayesreg.segmenters.v1)
  infer      cross-fitted ensemble inference, float16 outputs, P_S provenance
  crosscal   cross-fitted check of the ensemble calibration (NLL / ECE; information only)
  snapshot   git-archive code snapshot of a real run; the code record of every output
  pins       pinned inputs of real runs (folds sha256, repository allowlist, training source); --synthetic
  synthetic  synthetic NIfTI subjects / folds / allowlist / pair for the dry run and the tests
  cli        python -m bayesreg.seg.unet.cli {train,calibrate,check-members,register,evaluate-calibration,
             infer,check-tissue4} | --dry-run
"""

SEGMENTER_ID = "unet35_v1_ens"
MODEL_PREFIX = "unet35_v1"
