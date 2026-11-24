"""Synthetic split / allowlist / registry files for the EB leak-guard tests (ids OASIS_OAS1_09xx_MR1, which do not
exist in OASIS; nothing under data/ or splits/ is read)."""
import json

from bayesreg.io import load_dev_allowlist, load_segmenter_registry, subject_list_sha256
from bayesreg.seg.unet.data import load_folds
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, write_json

FOLD_A_TRAIN = [oasis(n) for n in (901, 902, 903, 904)]
FOLD_A_CALIB = [oasis(905)]
FOLD_B_TRAIN = [oasis(n) for n in (906, 907, 908, 909, 911)]     # 911 is also on the T0-like list (defence test)
FOLD_B_CALIB = [oasis(910)]
T0 = [oasis(911), oasis(912)]
DEV = [oasis(913), oasis(914)]
TRAINING = FOLD_A_TRAIN+FOLD_A_CALIB+FOLD_B_TRAIN+FOLD_B_CALIB
EB_ROWS = [
    (oasis(901), oasis(902), "A"),      # admissible (fold A)
    (oasis(906), oasis(907), "B"),      # admissible (fold B)
    (oasis(903), oasis(908), "A"),      # cross-fold (903 in A, 908 in B)
    (oasis(913), oasis(914), "A"),      # development subjects, not training
    (oasis(904), oasis(905), "B"),      # tsv fold differs from the folds file (both A)
    (oasis(909), oasis(911), "B"),      # 911 never_for_training
]


def write_files(root, *, training_source="bayesreg_train_v1", rows=EB_ROWS):
    root.mkdir(parents=True, exist_ok=True)
    folds = write_json(root/"folds.json", folds_document(FOLD_A_TRAIN, FOLD_A_CALIB, FOLD_B_TRAIN, FOLD_B_CALIB))
    allow = write_json(root/"dev_subjects.json", allowlist_document([
        ("stage_L2R_val_v1", DEV, False), ("stage_T0_tune_v1", T0, True), (training_source, TRAINING, False)]))
    tsv = root/"eb_pairs.tsv"
    tsv.write_text("fixed\tmoving\tfold\n"+"".join(f"{f}\t{m}\t{d}\n" for f, m, d in rows), encoding="utf-8")
    contact_a, contact_b = sorted(FOLD_A_TRAIN+FOLD_A_CALIB), sorted(FOLD_B_TRAIN+FOLD_B_CALIB)
    registry = dict(schema="bayesreg.segmenters.v1", models={
        "unet35_v1_foldA": dict(segmenter_id="unet35_v1_ens", fold=0, member_model_sha256=["a"*64, "b"*64],
                                label_contact_subjects=contact_a,
                                label_contact_subjects_sha256=subject_list_sha256(contact_a)),
        "unet35_v1_foldB": dict(segmenter_id="unet35_v1_ens", fold=1, member_model_sha256=["c"*64, "d"*64],
                                label_contact_subjects=contact_b,
                                label_contact_subjects_sha256=subject_list_sha256(contact_b))})
    reg = root/"segmenters.json"
    reg.write_text(json.dumps(registry, indent=1)+"\n", encoding="utf-8")
    return dict(folds=folds, allowlist=allow, eb_pairs=tsv, registry=reg)


def loaded(root, **kwargs):
    paths = write_files(root, **kwargs)
    return paths, load_folds(paths["folds"]), load_dev_allowlist(paths["allowlist"]), \
        load_segmenter_registry(paths["registry"])
