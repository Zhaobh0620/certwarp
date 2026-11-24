"""The EB leak guard (contract C15-1, D23, D24): only rows of eb_pairs.tsv, same fold, B-line training subjects
of the allowlist source bayesreg_train_v1, never_for_training refused; structure layer only from the other fold's
models.  Synthetic files only."""
import json

import pytest

from bayesreg.io import AcceptedProvenance, load_segmenter_registry, subject_list_sha256
from bayesreg.learn import EBPairRefused, admit_eb_pairs, check_structure_models, read_eb_pairs
from bayesreg.seg.unet.synthetic import oasis

from . import _guard_fixtures as gf


def admit(tmp_path, requested, **kwargs):
    paths, split, allowlist, _ = gf.loaded(tmp_path, **kwargs)
    return admit_eb_pairs(requested, eb_pairs_path=paths["eb_pairs"], split=split, allowlist=allowlist)


def test_admits_listed_same_fold_training_pairs(tmp_path):
    pairs, record = admit(tmp_path, [f"{oasis(901)}__{oasis(902)}", (oasis(906), oasis(907))])
    assert [(p.fixed, p.moving, p.fold) for p in pairs] == [(oasis(901), oasis(902), "A"), (oasis(906), oasis(907), "B")]
    assert pairs[0].pair_id == f"{oasis(901)}__{oasis(902)}"
    for key in ("eb_pairs_sha256", "folds_sha256", "allowlist_sha256", "training_source_sha256"):
        assert len(record[key]) == 64 and pairs[0].admission[key] == record[key]
    assert record["admitted"] == [p.pair_id for p in pairs]


@pytest.mark.parametrize("requested, message", [
    ((oasis(903), oasis(908)), "cross-fold"),
    ((oasis(913), oasis(914)), "not a B-line training subject"),
    ((oasis(904), oasis(905)), "eb_pairs.tsv says fold B"),
    ((oasis(909), oasis(911)), "never_for_training"),
    ((oasis(902), oasis(901)), "not a row"),                 # order matters (fixed, moving)
    ((oasis(901), oasis(906)), "not a row"),
])
def test_refusals(tmp_path, requested, message):
    with pytest.raises(EBPairRefused, match=message):
        admit(tmp_path, [requested])


def test_all_rows_are_checked_and_the_training_source_is_required(tmp_path):
    with pytest.raises(EBPairRefused):
        admit(tmp_path/"a", None)                              # the tsv holds bad rows: the first one refuses
    good = gf.EB_ROWS[:2]
    pairs, _ = admit(tmp_path/"b", None, rows=good)
    assert len(pairs) == 2
    with pytest.raises(EBPairRefused, match="no source"):
        admit(tmp_path/"c", None, rows=good, training_source="other_source")


def test_eb_pairs_file_format(tmp_path):
    path = tmp_path/"eb.tsv"
    path.write_text("fixed\tmoving\tfold\n"+f"{oasis(901)}\t{oasis(902)}\tA\n", encoding="utf-8")
    rows, sha = read_eb_pairs(path)
    assert rows == ((oasis(901), oasis(902), "A"),) and len(sha) == 64
    for text in ("a\tb\tc\n", f"fixed\tmoving\tfold\n{oasis(901)}\t{oasis(902)}\tC\n",
                 f"fixed\tmoving\tfold\n{oasis(901)}\t{oasis(902)}\tA\n{oasis(902)}\t{oasis(903)}\tA\n",
                 f"fixed\tmoving\tfold\nS01\t{oasis(902)}\tA\n", "fixed\tmoving\tfold\n"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(EBPairRefused):
            read_eb_pairs(path)


def accepted(models, pair, folds=None):
    folds = {"unet35_v1_foldA": 0, "unet35_v1_foldB": 1} if folds is None else folds
    return AcceptedProvenance(dict(layer="structure", model_ids=list(models), folds={m: folds[m] for m in models},
                                   pair_subjects_checked=[pair.fixed, pair.moving]))


def test_structure_models_must_be_of_the_other_fold(tmp_path):
    paths, split, allowlist, registry = gf.loaded(tmp_path)
    (fold_a, fold_b), _ = admit_eb_pairs([(oasis(901), oasis(902)), (oasis(906), oasis(907))],
                                         eb_pairs_path=paths["eb_pairs"], split=split, allowlist=allowlist)
    record = check_structure_models(accepted(["unet35_v1_foldB"], fold_a), fold_a, registry)
    assert record["models"] == {"unet35_v1_foldB": 1} and record["pair_fold_index"] == 0
    check_structure_models(accepted(["unet35_v1_foldA"], fold_b), fold_b, registry)
    with pytest.raises(EBPairRefused, match="own fold"):
        check_structure_models(accepted(["unet35_v1_foldA"], fold_a), fold_a, registry)     # same-fold model
    with pytest.raises(EBPairRefused, match="own fold"):
        check_structure_models(accepted(["unet35_v1_foldA", "unet35_v1_foldB"], fold_a), fold_a, registry)  # mean
    with pytest.raises(EBPairRefused, match="registry fold"):
        check_structure_models(accepted(["unet35_v1_foldB"], fold_a, folds={"unet35_v1_foldB": 0}), fold_a, registry)
    with pytest.raises(EBPairRefused, match="different pair"):
        check_structure_models(accepted(["unet35_v1_foldB"], fold_b), fold_a, registry)
    with pytest.raises(EBPairRefused, match="not registered"):
        check_structure_models(accepted(["unet35_v1_foldC"], fold_a, folds={"unet35_v1_foldC": 1}), fold_a, registry)
    with pytest.raises(EBPairRefused, match="structure-layer"):
        check_structure_models(AcceptedProvenance(dict(layer="tissue")), fold_a, registry)


def test_structure_models_without_label_contact(tmp_path):
    paths, split, allowlist, _ = gf.loaded(tmp_path)
    (pair,), _ = admit_eb_pairs([(oasis(901), oasis(902))], eb_pairs_path=paths["eb_pairs"], split=split,
                                allowlist=allowlist)
    document = json.loads(paths["registry"].read_text())
    contact = sorted(document["models"]["unet35_v1_foldB"]["label_contact_subjects"]+[oasis(901)])
    document["models"]["unet35_v1_foldB"].update(label_contact_subjects=contact,
                                                 label_contact_subjects_sha256=subject_list_sha256(contact))
    leaky = tmp_path/"leaky.json"
    leaky.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EBPairRefused, match="label contact"):
        check_structure_models(accepted(["unet35_v1_foldB"], pair), pair, load_segmenter_registry(leaky))
