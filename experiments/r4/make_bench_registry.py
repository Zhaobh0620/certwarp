"""D134：写台的登记 configs/bayesreg/segmenters_r4_bench_v1.json 与附属文件 configs/bayesreg/segmenters_r4_bench_v1_contact.json（只写一次；已有就拒绝）。
登记照 bayesreg/io/pair.py _parse_registry 的严格键：segmenter_id unet35_v4_uf2_bench，模型 _foldA（fold 0）、_foldB（fold 1），成员 sha256 与 segmenters_hs_v1.json 的同折模型相同，
label_contact_subjects 照旧是 106 人；label contact 的两部分（训练 98 人、温度标定与监视 8 人）放在附属文件。只读钉住的 folds_full_v2.json 与 HS 的登记（id 与散列），不读任何受试者文件。
不加进 bayesreg/seg/structure_layer.py 的 FORMAL_REGISTRIES（C25.12-5）。

    python -B experiments/r4/make_bench_registry.py [--repo DIR]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r4_common as rc  # noqa: E402


def documents(repo):
    from bayesreg.io.pair import load_segmenter_registry, subject_list_sha256
    from bayesreg.seg.unet.data import load_full_folds
    import r4_bench as rb
    repo = Path(repo)
    if rc.sha256_file(repo/rc.FOLDS_REL) != rc.FOLDS_SHA256 or rc.sha256_file(repo/rc.HS_REGISTRY_REL) != rc.HS_REGISTRY_SHA256:
        raise SystemExit('folds_full_v2.json 或 segmenters_hs_v1.json 不是钉住的那份')
    split = load_full_folds(repo/rc.FOLDS_REL, expect_counts=(rc.N_TRAIN, rc.N_CALIB))
    hs = load_segmenter_registry(repo/rc.HS_REGISTRY_REL)
    models = {}
    for f in rc.FOLDS:
        entry = hs.models[rc.HS_FOLD_MODELS[f]]
        contact = sorted(split.subjects(f))
        if list(entry['label_contact_subjects']) != contact:
            raise SystemExit(f'{rc.HS_FOLD_MODELS[f]} 的 label_contact_subjects 不是 folds_full_v2.json 折 {f} 的 train + calib')
        models[rc.BENCH_MODELS[f]] = dict(segmenter_id=rc.BENCH_SEGMENTER_ID, fold=rc.FOLD_INDEX[f], member_model_sha256=list(entry['member_model_sha256']),
                                          label_contact_subjects=contact, label_contact_subjects_sha256=subject_list_sha256(contact))
    registry = dict(schema='bayesreg.segmenters.v1', comment=('R4 bench (contract C25.6-6, -7, -8; D134): UF2 by fold, the members of segmenters_hs_v1.json\'s same-fold model; used only by the bench reader '
                                                             '(experiments/r4/r4_bench.py) for the calibration subjects of the model\'s own fold; not a formal registry; the label contact in two parts is in '
                                                             'segmenters_r4_bench_v1_contact.json; append-only'), models=models)
    contact = rb.contact_document(rc.HS_REGISTRY_SHA256, split)
    return registry, contact


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo', type=Path, default=rc.ROOT)
    o = ap.parse_args(argv)
    registry, contact = documents(o.repo)
    for rel, doc in ((rc.BENCH_REGISTRY_REL, registry), (rc.BENCH_CONTACT_REL, contact)):
        path = o.repo/rel
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(doc, indent=1, ensure_ascii=False)+'\n')
        print(f'{rel}: {rc.sha256_file(path)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
