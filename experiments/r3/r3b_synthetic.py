"""A synthetic world for R3b's tools (dry run and tests): no real subject, no real root.

    build(work) -> Ctx      experiments/unv2/synthetic_tree.build (ids OASIS_OAS1_09xx_MR1, folds 5 train + 1 calib per fold of folds_full_v2), plus four tiny
                            members fold{A,B}_s{0,1} with member.pt, member.json, train_record.json, a registry of unet35_v4_ens, and ctx.json
    load(work) -> Ctx       the Ctx of a built world (the CLI's --synthetic)
The results root, the lock and the outputs live under <work>/results; the plan, the code and the sealed lists the guard reads are the repository's.
"""
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT/"experiments"/"unv2", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import r3b_common as rc  # noqa: E402
import synthetic_tree  # noqa: E402
from bayesreg.io.pair import subject_list_sha256  # noqa: E402
from bayesreg.seg.unet.calibrate import MEMBER_FORMAT  # noqa: E402
from bayesreg.seg.unet.data import load_full_folds  # noqa: E402
from bayesreg.seg.unet.model import build_model  # noqa: E402
from bayesreg.seg.unet.synthetic import write_json  # noqa: E402
from bayesreg.seg.unet.train import file_sha256  # noqa: E402

TINY = dict(base_channels=4, depth=2)
CRITERION = dict(rc.CRITERION, n_subjects=2, small6_positive_min=1)


def _member(models, fold, seed, split, contact):
    torch.manual_seed(300+2*"AB".index(fold)+seed)
    model = build_model(TINY)
    directory = Path(models)/f"fold{fold}_s{seed}"
    directory.mkdir(parents=True)
    train, calib = split.subjects(fold, "train"), split.subjects(fold, "calib")
    record = dict(format="bayesreg.unet35.train_record.v1", fold=fold, seed=seed, recipe="synthetic", train_subjects=train, calib_subjects=calib,
                  label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact))
    write_json(directory/"train_record.json", record)
    temperature = .5+.1*seed
    torch.save(dict(format=MEMBER_FORMAT, state_dict=model.state_dict(), model_config=dict(TINY), temperature=temperature, fold=fold, seed=seed, best_iteration=1,
                    label_contact_subjects=list(contact), label_contact_subjects_sha256=subject_list_sha256(contact), model_best_sha256="0"*64, percentile=99.5,
                    window=None, amp=False), directory/"member.pt")
    sha = file_sha256(directory/"member.pt")
    write_json(directory/"member.json", dict(format="bayesreg.unet35.member_record.v1", member_file="member.pt", member_sha256=sha, fold=fold, seed=seed,
                                             temperature=temperature, label_contact_subjects=list(contact),
                                             train_record_sha256=file_sha256(directory/"train_record.json")))
    return sha


def build(work):
    work = Path(work)
    synthetic_tree.build(work)
    split = load_full_folds(work/"folds_full_v2.json", expect_counts=(5, 1))
    contact = sorted(split.training_subjects)
    shas = {(f, s): _member(work/"members", f, s, split, contact) for f in rc.FOLDS for s in rc.SEEDS}
    write_json(work/"segmenters.json", dict(schema="bayesreg.segmenters.v1", comment="synthetic", models={
        f"unet35_v4_ens_fold{f}": dict(segmenter_id=rc.SEGMENTER_ID, fold=i, member_model_sha256=[shas[(f, 0)], shas[(f, 1)]], label_contact_subjects=contact,
                                       label_contact_subjects_sha256=subject_list_sha256(contact)) for i, f in enumerate(rc.FOLDS)}))
    document = dict(work=str(work), calib={f: split.subjects(f, "calib") for f in rc.FOLDS}, member_sha={f: [shas[(f, 0)], shas[(f, 1)]] for f in rc.FOLDS})
    write_json(work/"ctx.json", document)
    return load(work)


def load(work):
    import r3b_calib
    work = Path(work)
    document = json.loads((work/"ctx.json").read_text(encoding="utf-8"))
    results = work/"results"
    return r3b_calib.Ctx(repo=ROOT, out=results/"r3b_calib_v1", results=results, images=work/"data", data_link=work/"data", models=work/"members",
                         registry=work/"segmenters.json", folds=work/"folds_full_v2.json", allowlist=work/"dev_subjects.json", seg4_map=ROOT/rc.SEG4_MAP_REL,
                         lock=results/"r3b_calib_truth_read.lock", calib={f: tuple(v) for f, v in document["calib"].items()},
                         member_sha={f: tuple(v) for f, v in document["member_sha"].items()}, criterion=dict(CRITERION), synthetic=True,
                         training_source=synthetic_tree.SOURCE, expect_counts=(5, 1), source_root=work)
