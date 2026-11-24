"""R4 的筛选台（合同 C25.6）：受试者与配对、台守卫 bench_admission、台的登记与附属文件、结构层读取器（照做 validate_segmentation_provenance 与 StructureLayer.load
的全部核对，只换四处）、oracle 标记、台的 stage、Atropos、推断。

D24 的窄例外（C25.6-7，D129）只在这里成立：只许折内配对，只许本折两个 UF 成员（UF2）给本折的 calib 受试者分割。
现有入口（a4_admission、StructureLayer、select_fold_models、HS、HE、HI）一个字不改；它们见到台的登记或台的受试者照旧拒绝（测试）。
每个读受试者文件的函数都先过 C23.4-2（bayesreg.io.guard.check_subject，经 bench_admission），并登记在静态普查里（tests/test_bayesreg/sg/_census_inventory.py）。
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for _d in ('mc0', 'ef', 'fl', 'he', 'd0', 'hs'):
    _p = str(HERE.parent/_d)
    if _p not in sys.path:
        sys.path.append(_p)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import r4_common as rc  # noqa: E402


class BenchRefused(rc.Refused):
    """台守卫或台的读取器拒绝：读真值之前的失败（C25.7-3）。"""
    status = 'bench_refused'


@dataclass(frozen=True)
class BenchWorld:
    """台的输入。默认是真实的（仓库里的钉住文件）；测试给合成的（synthetic=True，受试者 id 照 bayesreg/seg/unet/synthetic.py 的约定）。"""
    repo: Path = rc.ROOT
    folds_path: Path = rc.ROOT/rc.FOLDS_REL
    folds_sha256: str = rc.FOLDS_SHA256
    train_list_path: Path = rc.ROOT/rc.TRAIN_LIST_REL
    allowlist_path: Path = rc.ROOT/rc.ALLOWLIST_REL
    hs_registry_path: Path = rc.ROOT/rc.HS_REGISTRY_REL
    hs_registry_sha256: str = rc.HS_REGISTRY_SHA256
    training_source: str = rc.TRAINING_SOURCE
    dev_source: str = rc.DEV_SOURCE
    expect_counts: tuple = (rc.N_TRAIN, rc.N_CALIB)
    synthetic: bool = False

    @property
    def registry_path(self):
        return Path(self.repo)/rc.BENCH_REGISTRY_REL          # 固定路径（C25.6-6 (b) 第四处）

    @property
    def contact_path(self):
        return Path(self.repo)/rc.BENCH_CONTACT_REL


@dataclass(frozen=True)
class Bench:
    world: BenchWorld
    split: object
    allowlist: object
    subjects: dict                                          # {fold: [8 个，规范 id 升序]}
    pairs: tuple                                            # ((name, fold, i, fixed, moving), ...)，A 折 8 对再 B 折 8 对
    record: dict = field(default_factory=dict)

    @property
    def whitelist(self):
        return frozenset(s for f in rc.FOLDS for s in self.subjects[f])

    @property
    def pair_names(self):
        return tuple(p[0] for p in self.pairs)

    def pair_of(self, name):
        for row in self.pairs:
            if row[0] == name:
                return row
        return None

    def fold_of(self, subject):
        for f in rc.FOLDS:
            if subject in self.subjects[f]:
                return f
        return None

    def pair_subjects(self):
        return {p[0]: (p[3], p[4]) for p in self.pairs}

    def ordered_subjects(self):
        return [s for f in rc.FOLDS for s in self.subjects[f]]


def pair_name(fixed, moving):
    return f'{fixed}__{moving}'


def ring_pairs(subjects):
    """第 i 对 = (fixed s_i, moving s_{(i+1) mod 8})，s 按规范 id 升序（C25.6-3）。"""
    s = sorted(subjects)
    return [(s[i], s[(i+1) % len(s)]) for i in range(len(s))]


def load_bench(world=None):
    """钉住的 folds_full_v2.json 算出 16 人与 16 对，并核 C25.6-2 的每一条；任何一条不符 BenchRefused（读真值之前的失败，归整台）。"""
    from bayesreg.io.guard import check_subject, load_dev_allowlist
    from bayesreg.seg.unet.data import load_full_folds
    world = BenchWorld() if world is None else world
    problems = []
    got = rc.sha256_file(world.folds_path) if Path(world.folds_path).is_file() else None
    if got != world.folds_sha256:
        raise BenchRefused(f'{world.folds_path}：sha256 {got} 不是钉住的 {world.folds_sha256}（未钉住或改过的 folds 文件）')
    split = load_full_folds(world.folds_path, expect_counts=None if world.synthetic else world.expect_counts)
    if split.sha256 != world.folds_sha256:
        raise BenchRefused('folds 文件在读的时候变了')
    allowlist = load_dev_allowlist(world.allowlist_path)
    subjects = {f: sorted(split.subjects(f, 'calib')) for f in rc.FOLDS}
    train = {f: set(split.subjects(f, 'train')) for f in rc.FOLDS}
    for f in rc.FOLDS:
        if not world.synthetic and (len(subjects[f]) != rc.N_CALIB or len(train[f]) != rc.N_TRAIN):
            problems.append(f'折 {f}：calib {len(subjects[f])}、train {len(train[f])}，应为 {rc.N_CALIB}、{rc.N_TRAIN}')
        if set(subjects[f]) & train[f]:
            problems.append(f'折 {f}：calib 与本折 train 相交')
    if not set(subjects['A']) <= train['B'] or not set(subjects['B']) <= train['A']:
        problems.append('A 折 calib ⊄ B 折 train 或 B 折 calib ⊄ A 折 train')
    if set(subjects['A']) & set(subjects['B']):
        problems.append('两折的 calib 相交')
    everyone = [s for f in rc.FOLDS for s in subjects[f]]
    listed = {line.strip() for line in Path(world.train_list_path).read_text(encoding='utf-8').splitlines() if line.strip()}
    if not set(everyone) <= listed:
        problems.append(f'不在 {world.train_list_path.name} 里：{sorted(set(everyone)-listed)}')
    sources = {s['name']: s for s in allowlist.sources}
    if world.training_source not in sources or not set(everyone) <= set(sources[world.training_source]['subjects']):
        problems.append(f'不全在白名单的 {world.training_source} 来源里')
    dev = set(sources.get(world.dev_source, {}).get('subjects', []))
    if not world.synthetic and not dev:
        problems.append(f'白名单没有开发集来源 {world.dev_source}')
    for s in everyone:
        verdict = check_subject(s, allowlist)                  # 封存守卫（C23.4-2）：B 线留出集、H1–H4 在这里拒
        if not verdict.admitted:
            problems.append(f'{s}：封存守卫拒绝（{verdict.step}）')
        if s in dev:
            problems.append(f'{s} 在开发集里')
        if s in allowlist.never_for_training:
            problems.append(f'{s} 是 T0（never_for_training）')
    if problems:
        raise BenchRefused('C25.6-2：'+'；'.join(problems))
    pairs = tuple((pair_name(f_, m_), f, i, f_, m_) for f in rc.FOLDS for i, (f_, m_) in enumerate(ring_pairs(subjects[f])))
    record = dict(folds=str(world.folds_path), folds_sha256=world.folds_sha256, subjects=subjects, pairs=[p[0] for p in pairs], allowlist_sha256=allowlist.sha256,
                  checks='C25.6-2: 8 calib / 98 train per fold, own calib and train disjoint, A calib ⊂ B train, B calib ⊂ A train, all in train_subjects.txt and the training source, '
                         'none in the development set, T0, a sealed list (check_subject)')
    return Bench(world=world, split=split, allowlist=allowlist, subjects=subjects, pairs=pairs, record=record)


def pairs_table_sha256(bench):
    text = ''.join(f'{name}\t{fold}\t{i}\t{f}\t{m}\n' for name, fold, i, f, m in bench.pairs)
    return hashlib.sha256(text.encode()).hexdigest()


# ---- 台守卫（C25.6-6 (a)） -------------------------------------------------------------------------------------------------------------------
def bench_admission(bench, *, subjects=(), pair=None, model_id=None):
    """在打开任何受试者文件之前调用。先过封存守卫（check_subject，C23 不放宽），再核：受试者在 16 人白名单里；对在折内配对表里；给受试者分割的模型恰是它本折的台模型。
    BenchRefused 否则。返回记录。"""
    from bayesreg.io.guard import check_subject
    named = list(subjects)
    row = None
    if pair is not None:
        if not isinstance(pair, str) or pair.count('__') != 1:
            raise BenchRefused(f'对的名字 {pair!r} 不是 <fixed>__<moving>')
        named += pair.split('__')
    for s in named:
        verdict = check_subject(s, bench.allowlist)
        if not verdict.admitted:
            raise BenchRefused(f'{s!r}：封存守卫拒绝（{verdict.step}：{verdict.message or "不在白名单上"}）')
    for s in named:
        if s not in bench.whitelist:
            raise BenchRefused(f'{s}：不是台的 16 人（开发集、T0、B 线留出集、H1–H4、训练集其余 90 人一律拒绝）')
    if pair is not None:
        row = bench.pair_of(pair)
        if row is None:
            raise BenchRefused(f'{pair}：不在折内配对表里（跨折、颠倒或不在表里的对一律拒绝）')
    if model_id is not None:
        for s in named:
            want = rc.BENCH_MODELS[bench.fold_of(s)]
            if model_id != want:
                raise BenchRefused(f'{s}：模型 {model_id!r} 不是它本折的台模型 {want}')
    return dict(rule='bench_admission (C25.6-6 (a)): check_subject, the 16-subject whitelist, the within-fold pair table, the own-fold bench model', subjects=named, pair=pair,
                fold=None if row is None else row[1], index=None if row is None else row[2], model_id=model_id, allowlist_sha256=bench.allowlist.sha256)


# ---- 台的登记与附属文件（D134；C25.6-6 (b) 第四处） --------------------------------------------------------------------------------------------
def git_state(repo, rel):
    """(tracked, clean)：文件已入 git、工作树里没有改动。"""
    tracked = subprocess.run(['git', '-C', str(repo), 'ls-files', '--error-unmatch', '--', rel], capture_output=True, text=True).returncode == 0
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', rel], capture_output=True, text=True)
    return tracked, status.returncode == 0 and not status.stdout.strip()


def contact_document(registry_sha256_of_hs, split):
    """附属文件的内容（make_bench_registry.py 用它写文件；测试用它造合成文件）。"""
    models = {}
    for f in rc.FOLDS:
        models[rc.BENCH_MODELS[f]] = dict(fold=rc.FOLD_INDEX[f], label_contact_training=sorted(split.subjects(f, 'train')), label_contact_calibration=sorted(split.subjects(f, 'calib')),
                                          training_note='weights trained on these subjects\' labels (98)', calibration_note='temperature calibration and monitoring every 500 steps (8); no step selection (UF: no early stopping, step 40000)')
    return dict(schema=rc.CONTACT_SCHEMA, comment='label contact of the bench models in two parts (contract C25.6-7, D134); the union is the registry\'s label_contact_subjects', exception='C25.6-7 (D129)',
                source_registry=rc.HS_REGISTRY_REL, source_registry_sha256=registry_sha256_of_hs, models=models)


def load_bench_registry(bench, pins):
    """只用 load_segmenter_registry 读固定路径 configs/bayesreg/segmenters_r4_bench_v1.json；已入 git、工作树干净，sha256 等于 batch.json 钉的值；不走暂存登记。
    附属文件同样钉住；两部分之并等于登记的 label_contact_subjects；training = 本折 train，calibration = 本折 calib。返回 (registry, contact, record)。"""
    from bayesreg.io.pair import ProvenanceRejected, load_segmenter_registry
    world = bench.world
    problems = []
    for rel, key in ((rc.BENCH_REGISTRY_REL, 'registry_sha256'), (rc.BENCH_CONTACT_REL, 'contact_sha256')):
        path = Path(world.repo)/rel
        tracked, clean = git_state(world.repo, rel)
        if not tracked or not clean:
            problems.append(f'{rel}：没有入 git 或工作树里有改动（tracked {tracked}，clean {clean}）')
        got = rc.sha256_file(path) if path.is_file() else None
        if got is None or got != pins.get(key):
            problems.append(f'{rel}：sha256 {got} 不是 batch.json 钉的 {pins.get(key)}')
    if problems:
        raise BenchRefused('台的登记：'+'；'.join(problems))
    try:
        registry = load_segmenter_registry(world.registry_path)
        hs = load_segmenter_registry(world.hs_registry_path)
    except (ProvenanceRejected, ValueError, OSError) as exc:
        raise BenchRefused(f'台的登记读不了：{type(exc).__name__}: {exc}') from None
    if registry.sha256 != pins.get('registry_sha256'):
        raise BenchRefused('台的登记在读的时候变了')
    if hs.sha256 != world.hs_registry_sha256:
        problems.append(f'HS 的登记 sha256 {hs.sha256} 不是钉住的 {world.hs_registry_sha256}')
    if sorted(registry.models) != sorted(rc.BENCH_MODELS.values()):
        problems.append(f'台的登记的模型 {sorted(registry.models)} 不是 {sorted(rc.BENCH_MODELS.values())}')
    raw = Path(world.contact_path).read_bytes()
    contact = json.loads(raw.decode('utf-8'))
    if hashlib.sha256(raw).hexdigest() != pins.get('contact_sha256'):
        raise BenchRefused('附属文件在读的时候变了')
    if contact.get('schema') != rc.CONTACT_SCHEMA or sorted(contact.get('models') or {}) != sorted(rc.BENCH_MODELS.values()):
        problems.append('附属文件的 schema 或模型不对')
    for f in rc.FOLDS:
        m = rc.BENCH_MODELS[f]
        entry, c = registry.models.get(m), (contact.get('models') or {}).get(m) or {}
        if entry is None:
            continue
        if entry['segmenter_id'] != rc.BENCH_SEGMENTER_ID or entry['fold'] != rc.FOLD_INDEX[f]:
            problems.append(f'{m}：segmenter_id {entry["segmenter_id"]}、fold {entry["fold"]} 不是台的')
        hs_entry = hs.models.get(rc.HS_FOLD_MODELS[f])
        if hs_entry is None or tuple(entry['member_model_sha256']) != tuple(hs_entry['member_model_sha256']):
            problems.append(f'{m}：成员 sha256 不是 HS 登记里同折模型 {rc.HS_FOLD_MODELS[f]} 的')
        training, calibration = list(c.get('label_contact_training') or []), list(c.get('label_contact_calibration') or [])
        if sorted(set(training) | set(calibration)) != sorted(entry['label_contact_subjects']) or set(training) & set(calibration):
            problems.append(f'{m}：附属文件的两部分之并不等于登记的 label_contact_subjects')
        if sorted(training) != sorted(bench.split.subjects(f, 'train')) or sorted(calibration) != sorted(bench.split.subjects(f, 'calib')):
            problems.append(f'{m}：training 不是本折 train 或 calibration 不是本折 calib')
        if c.get('fold') != rc.FOLD_INDEX[f]:
            problems.append(f'{m}：附属文件的 fold 不对')
    if problems:
        raise BenchRefused('台的登记：'+'；'.join(problems))
    record = dict(path=rc.BENCH_REGISTRY_REL, sha256=registry.sha256, contact_path=rc.BENCH_CONTACT_REL, contact_sha256=pins['contact_sha256'], status='bench_registry', staging=False,
                  git=dict(tracked=True, clean=True), rule='load_segmenter_registry on the fixed path, committed and clean, the sha256 pinned in batch.json; no staging registry (C25.6-6 (b), revision 10e)')
    return registry, contact, record


# ---- 结构层读取器（C25.6-6 (b)） ------------------------------------------------------------------------------------------------------------------
def _reject(reason):
    from bayesreg.io.pair import ProvenanceRejected
    raise ProvenanceRejected(reason)


def bench_validate_provenance(prov, pair, *, side, registry, allowlist, bench):
    """validate_segmentation_provenance（bayesreg/io/pair.py:344）的结构层路径，全部核对照做，只换三处：
    (1) test_labels_seen 恰为 'calibration_only'；(3) 配对级 label contact 的 overlap 换成 C25.6-7（模型是本折的、两人都在本折 calib、contact ⊆ 训练集 106 人）。
    第二处（选模型与 policy）在 bench_check_models。"contact 全在白名单上""不含 never_for_training"两条照做。另核登记的 segmenter_id 等于台的 id。"""
    from bayesreg.io.guard import DevAllowlist, is_safe_id
    from bayesreg.io.pair import (FORBIDDEN_ID, SHA256_FIELDS, AcceptedProvenance, PairInput, SegmenterRegistry, _class_names_problem, _id_list, _sha_map, recheck_sealed_pair)
    from jd_curl.validation import InputValidationError
    if not isinstance(pair, PairInput):
        raise InputValidationError('pair: the PairInput returned by load_pair is required')
    recheck_sealed_pair(pair, allowlist, where='bench structure layer')
    if not isinstance(registry, SegmenterRegistry):
        raise InputValidationError('registry: a SegmenterRegistry from load_segmenter_registry is required')
    if not isinstance(allowlist, DevAllowlist) or allowlist.sha256 != pair.provenance['subject_guard']['allowlist_sha256']:
        raise InputValidationError('allowlist: the DevAllowlist that load_pair used is required')
    if not isinstance(prov, dict):
        _reject('provenance is not a JSON object')
    if prov.get('layer') != 'structure':
        _reject(f"provenance layer {prov.get('layer')!r} given for the 'structure' slot")
    sid = prov.get('segmenter_id')
    if not is_safe_id(sid):
        _reject('segmenter_id missing or not an identifier')
    if FORBIDDEN_ID.search(sid):
        _reject(f'segmenter_id {sid!r} is a label source')
    if prov.get('oracle'):
        _reject('input carries the oracle marker')
    if prov.get('test_labels_seen') != rc.BENCH_TEST_LABELS_SEEN or not isinstance(prov.get('test_labels_seen'), str):      # 换的第一处
        _reject(f"test_labels_seen must be exactly {rc.BENCH_TEST_LABELS_SEEN!r} on the bench (D134), got {prov.get('test_labels_seen')!r}")
    if prov.get('trained_with_labels') is not True:
        _reject('trained_with_labels must be True for the structure layer')
    model_ids = _id_list(prov.get('model_ids'), 'model_ids')
    class_names = prov.get('class_names')
    problem = _class_names_problem(class_names)
    if problem:
        _reject(problem)
    inputs, outputs = (_sha_map(prov.get(name), name) for name in SHA256_FIELDS)
    if len(inputs) != 1 or next(iter(inputs.values())) != pair.source_sha256[side]:
        _reject(f'input_sha256 must be exactly the {side} source image (manifest sources.{side}.sha256)')
    subject = pair.fixed_subject if side == 'fixed' else pair.moving_subject
    if sid != rc.BENCH_SEGMENTER_ID:
        _reject(f'segmenter_id {sid!r} is not the bench id {rc.BENCH_SEGMENTER_ID!r}')
    entries = {}
    for model_id in model_ids:
        entry = registry.models.get(model_id)
        if entry is None:
            _reject(f'model {model_id!r} is not registered')
        if entry['segmenter_id'] != sid:
            _reject(f"model {model_id!r} is registered for {entry['segmenter_id']!r}, not {sid!r}")
        entries[model_id] = entry
    hashes = prov.get('model_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(model_ids) or any(not isinstance(hashes[m], list) or tuple(hashes[m]) != entries[m]['member_model_sha256'] for m in model_ids):
        _reject('model_sha256 must map every model id to its registered member sha256 list')
    for model_id, entry in entries.items():
        contact = set(entry['label_contact_subjects'])
        off = sorted(s for s in contact if not allowlist.allows(s))
        if off:
            _reject(f'model {model_id!r} had label contact with subjects off the allowlist {off}')
        tuning = sorted(contact & allowlist.never_for_training)
        if tuning:
            _reject(f'model {model_id!r} had label contact with never_for_training subjects {tuning}')
        fold = next((f for f in rc.FOLDS if rc.FOLD_INDEX[f] == entry['fold']), None)              # 换的第三处（C25.6-7）
        calib = set(bench.subjects.get(fold, ()))
        if model_id != rc.BENCH_MODELS.get(fold) or not {pair.fixed_subject, pair.moving_subject} <= calib:
            _reject(f'D24 narrow exception (C25.6-7): model {model_id!r} is not the own-fold bench model of both pair subjects, or the pair is not within one fold\'s calibration subjects')
        if not contact <= set(bench.split.training_subjects):
            _reject(f'model {model_id!r}: label contact outside the 106 training subjects')
    record = dict(check='bench: validate_segmentation_provenance with three substitutions (C25.6-6 (b))', layer='structure', side=side, subject=subject, segmenter_id=sid, model_ids=model_ids,
                  trained_with_labels=True, test_labels_seen=rc.BENCH_TEST_LABELS_SEEN, class_names=list(class_names), input_sha256=inputs, output_sha256=outputs)
    return AcceptedProvenance(record, model_sha256={m: list(entries[m]['member_model_sha256']) for m in model_ids}, folds={m: entries[m]['fold'] for m in model_ids},
                              label_contact_subjects_sha256={m: entries[m]['label_contact_subjects_sha256'] for m in model_ids}, registry_path=registry.path, registry_sha256=registry.sha256,
                              allowlist_sha256=allowlist.sha256, pair_subjects_checked=[pair.fixed_subject, pair.moving_subject], d24_exception='C25.6-7')


class BenchLayer:
    """StructureLayer 的台版本：同样的属性（pair、grouping、registry、registry_record、allowlist、subjects、expected），load(side) 照做 StructureLayer.load 的核对，只换选模型与 policy（第二处）。"""

    def __init__(self, pair, bench, registry, contact, registry_record, *, pred_root, models_root=None, groups_path=None, grouping=None):
        from bayesreg.io.pair import PairInput, recheck_sealed_pair
        from bayesreg.seg.structure_layer import DEFAULT_GROUPING, DEFAULT_GROUPS_PATH
        from bayesreg.seg.unet.groups import GroupingUnverified, load_grouping, require_inference_grouping
        from jd_curl.validation import InputValidationError
        if not isinstance(pair, PairInput):
            raise InputValidationError('pair: the PairInput returned by load_pair is required')
        recheck_sealed_pair(pair, bench.allowlist, where='bench structure layer')
        self.pair, self.bench, self.registry, self.contact, self.registry_record = pair, bench, registry, contact, registry_record
        self.root = Path(pred_root)
        self.models_root = None if models_root is None else Path(models_root)
        try:
            self.grouping = load_grouping(DEFAULT_GROUPS_PATH if groups_path is None else groups_path, DEFAULT_GROUPING if grouping is None else grouping)
            self.grouping_check = require_inference_grouping(self.grouping)
        except GroupingUnverified as exc:
            _reject(f'grouping_unverified: {exc}')
        except (ValueError, OSError) as exc:
            _reject(f'grouping: {exc}')
        self.allowlist = bench.allowlist
        if self.allowlist.sha256 != pair.provenance['subject_guard']['allowlist_sha256']:          # 允许名单与 load_pair 用的是同一份（structure_layer.py:295–296）
            raise InputValidationError('bench structure layer: the allowlist differs from the one load_pair used')
        self.subjects = dict(fixed=pair.fixed_subject, moving=pair.moving_subject)
        self.expected = {side: ([rc.BENCH_MODELS[bench.fold_of(s)]] if bench.fold_of(s) else [], rc.BENCH_POLICY) for side, s in self.subjects.items()}
        self.preflight = self._bench_contact_check()

    def _bench_contact_check(self):
        """换的第三处（C25.6-7）：所用模型是本折的；配对的两人都在本折 calib；模型的 label contact ⊆ 训练集 106 人。"""
        folds = {self.bench.fold_of(s) for s in self.subjects.values()}
        if len(folds) != 1 or None in folds:
            _reject(f'D24 narrow exception (C25.6-7): {sorted(self.subjects.values())} are not the calibration subjects of one fold: a cross-fold pair is refused')
        fold = folds.pop()
        model = rc.BENCH_MODELS[fold]
        entry = self.registry.models.get(model)
        if entry is None or not set(entry['label_contact_subjects']) <= set(self.bench.split.training_subjects):
            _reject(f'{model}: not registered, or its label contact is not within the 106 training subjects')
        return dict(check='C25.6-7 D24 narrow exception (preflight)', fold=fold, model=model, pair_subjects=[self.pair.fixed_subject, self.pair.moving_subject], within_fold_calibration=True,
                    contact_within_training=True, label_contact=dict(training=len((self.contact['models'][model])['label_contact_training']),
                                                                     calibration=len(self.contact['models'][model]['label_contact_calibration'])))

    def load(self, side):
        from bayesreg.io.pair import verify_segmentation_files
        from bayesreg.seg.structure_layer import GROUPS_FILE, PROVENANCE_FILE, StructureLayer, read_structure_groups
        from jd_curl.validation import InputValidationError
        if side not in ('fixed', 'moving'):
            raise InputValidationError('side: fixed or moving')
        subject = self.subjects[side]
        directory = self.root/subject
        path = directory/PROVENANCE_FILE
        if not path.is_file():
            _reject(f'{side}: {path} missing (the bench never infers here)')
        raw = path.read_bytes()
        try:
            prov = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _reject(f'{side}: {path} is not UTF-8 JSON ({exc})')
        accepted = bench_validate_provenance(prov, self.pair, side=side, registry=self.registry, allowlist=self.allowlist, bench=self.bench)
        models = self.bench_check_models(side, subject, prov, accepted)
        grouping = StructureLayer._check_grouping(self, side, prov)                                # 照做（structure_layer.py:378）
        StructureLayer._check_grid(self, side, prov)                                               # 照做（:395）
        self._check_contact_record(side, prov)
        verified = verify_segmentation_files(prov, directory)
        if GROUPS_FILE not in verified:
            _reject(f'{side}: {GROUPS_FILE} is not among the hash-verified outputs')
        array, read_back = read_structure_groups(directory/GROUPS_FILE, self.grouping.k+1, self.pair.grid.shape)
        environment = prov.get('environment') if isinstance(prov.get('environment'), dict) else {}
        record = dict(accepted=dict(accepted), directory=str(directory), provenance_sha256=hashlib.sha256(raw).hexdigest(), verified_outputs=verified, generated_in_this_run=False, inference=None,
                      models=models, grouping=grouping, read_back=read_back, prediction_registry=dict(path=prov.get('registry_path'), sha256=prov.get('registry_sha256')),
                      prediction_code=environment.get('code'), prediction_runtime_seconds=prov.get('runtime_seconds'))
        return array, accepted, record

    def bench_check_models(self, side, subject, prov, accepted):
        """StructureLayer._check_models（:353）换第二处：模型恰是本折的台模型，policy 恰为 bench_own_fold_calib。"""
        ids, policy = self.expected[side]
        if prov.get('segmenter_id') != rc.BENCH_SEGMENTER_ID:
            _reject(f"{side}: segmenter_id {prov.get('segmenter_id')!r} != {rc.BENCH_SEGMENTER_ID!r}")
        if prov.get('subject') != subject:
            _reject(f"{side}: the prediction is of {prov.get('subject')!r}, not {subject!r}")
        if not ids or sorted(accepted['model_ids']) != ids:
            _reject(f"{side}: bench model selection: {subject} must be segmented by {ids} ({policy}), the prediction used {sorted(accepted['model_ids'])}")
        if prov.get('policy') != policy:
            _reject(f"{side}: policy {prov.get('policy')!r} != {policy!r} (D134)")
        members = prov.get('members')
        if not isinstance(members, list) or not all(isinstance(m, dict) for m in members):
            _reject(f'{side}: members: a list of member records required')
        got = sorted((m.get('model_id'), m.get('member_sha256')) for m in members)
        want = sorted((m, sha) for m in ids for sha in self.registry.models[m]['member_model_sha256'])
        if got != want:
            _reject(f'{side}: members must be every registered seed member of {ids} (model id, sha256)')
        folds = {m: self.registry.models[m]['fold'] for m in ids}
        if prov.get('folds') != folds:
            _reject(f"{side}: folds {prov.get('folds')} != the registry {folds}")
        temperatures = self._check_temperatures(side, members)
        return dict(check='bench model selection and member sha256 against the bench registry (C25.6-6 (b) second substitution)', model_ids=ids, policy=policy, folds=folds,
                    members=[dict(model_id=m, member_sha256=sha) for m, sha in want], temperatures=temperatures)

    def _check_temperatures(self, side, members):
        """C25.6-6 (b)（复核次要 9）：每个成员的温度等于成员文件记的（<models_root>/fold*/member.json 的 temperature，按 member_sha256 找；登记表里没有温度）。"""
        if self.models_root is None:
            _reject(f'{side}: the member files root is required to check the temperatures (C25.6-6 (b))')
        index = {}
        for path in sorted(self.models_root.glob('fold*/member.json')):
            try:
                doc = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            index[doc.get('member_sha256')] = (path, doc.get('temperature'))
        out = {}
        for m in members:
            got = index.get(m.get('member_sha256'))
            if got is None or not isinstance(got[1], (int, float)) or got[1] != m.get('temperature'):
                _reject(f"{side}: member {str(m.get('member_sha256'))[:12]}…: temperature {m.get('temperature')} is not the member file's {None if got is None else got[1]}")
            out[m['member_sha256']] = dict(temperature=got[1], member_json=str(got[0]))
        return out

    def _check_contact_record(self, side, prov):
        """provenance 记附属文件的 sha256 与 label contact 的两部分（C25.6-8）。"""
        c = prov.get('label_contact') or {}
        model = self.expected[side][0][0]
        want = self.contact['models'][model]
        if c.get('side_file') != rc.BENCH_CONTACT_REL or c.get('side_file_sha256') != self.registry_record['contact_sha256'] \
                or list(c.get('training') or []) != list(want['label_contact_training']) or list(c.get('calibration') or []) != list(want['label_contact_calibration']):
            _reject(f'{side}: the provenance does not record the bench label contact (side file and its sha256, the two parts)')

    def record(self):
        return dict(kind='bench structure layer (C25.6-6 (b))', registry=self.registry_record, preflight=self.preflight, expected={s: dict(model_ids=i, policy=p) for s, (i, p) in self.expected.items()},
                    grouping=self.grouping.name)


# ---- oracle 标记（C25.6-6 (c)） -------------------------------------------------------------------------------------------------------------------
def bench_oracle_mark(keys_read, sides, *, segmenter_id=rc.BENCH_SEGMENTER_ID, policy=rc.BENCH_POLICY):
    """run_hs.hs_oracle_mark 的写法，id 与 policy 都换成台的：oracle 只有在 pair.npz 给了两幅图、没有别的成员，两侧预测都是台的 id 与 policy 时才是 False。"""
    image_keys = ('fixed', 'moving', 'initial_u')
    other = sorted(set(keys_read)-set(image_keys))
    images = {'fixed', 'moving'} <= set(keys_read)
    predictions = set(sides) == {'fixed', 'moving'} and all(s.get('segmenter_id') == segmenter_id and s.get('policy') == policy for s in sides.values())
    return dict(oracle=not (images and predictions and not other), truth_labels_read=bool(other))


# ---- 一侧的通道（fast_channels.read_side 的逐语句复制，只换 policy 一句；测试做 AST 对照） ----------------------------------------------------------------
import mc0_core as core  # noqa: E402
import mc0p_core as mc  # noqa: E402
import fast_channels as _fast  # noqa: E402

raw_labels, N_CLASSES = _fast.raw_labels, _fast.N_CLASSES


def bench_read_side(layer, pin, side, seg_root, workers=1):
    """(L, the side's record): make_inputs.unet_posterior and the per-side part of mc0p_core.read_predicted, the same refusals in the same order, on raw_labels; the old path when raw_labels says so."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    mi = core.make_inputs()
    subject = layer.subjects[side]
    ids, policy = layer.expected[side]
    if policy != rc.BENCH_POLICY:
        raise SystemExit(f'R4 bench {side} {subject}: policy {policy} != {rc.BENCH_POLICY} (C25.6-6 (b), D134)')
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise SystemExit(f'A4p {side} {subject}: structure layer refused ({type(exc).__name__}: {exc})')
    directory = Path(rec['directory'])
    if 'probs36.npz' not in rec['verified_outputs']:
        raise SystemExit(f'A4p {side} {subject}: probs36.npz is not among the hash-verified outputs')
    g = layer.grouping
    if mi.UNET_WM_GROUP not in g.group_names or tuple(g.group_labels[g.group_names.index(mi.UNET_WM_GROUP)]) != mi.CEREBRAL_WM:
        raise SystemExit(f'A4p {side} {subject}: grouping {g.name} has no group {mi.UNET_WM_GROUP} of seg35 labels {mi.CEREBRAL_WM} (alignment QC, protocol §12.4 item 5)')
    wm_channel = 1+g.group_names.index(mi.UNET_WM_GROUP)
    fast = raw_labels(directory/'probs36.npz', (N_CLASSES,)+groups.shape[1:], groups, (g.background_labels,)+tuple(g.group_labels), workers)
    if fast is None:                                # the old path decides, and reads everything again
        P, record, u_wm = mi.unet_posterior(layer, side)
        labels, p0 = mc.predicted_labels(P), np.array(P[0])
        del P
    else:
        labels, p0, read_back, deviation = fast
        u_wm = np.array(groups[wm_channel])
        if not deviation <= mi.GROUP_TOL:
            raise SystemExit(f'A4p {side} {subject}: probs36 summed by {g.name} differs from structure_groups.npz by {deviation:.3e} > {mi.GROUP_TOL:g}')
        record = dict(unet_provenance_sha256=rec['provenance_sha256'], unet_output_sha256=rec['verified_outputs'], unet_directory=str(directory), segmenter_id=accepted['segmenter_id'], model_ids=list(ids),
                      policy=policy, members=rec['models']['members'], registry_path=layer.registry.path, registry_sha256=layer.registry.sha256, registry_status=layer.registry_record['status'],
                      registry_git={k: (layer.registry_record.get('git') or {}).get(k) for k in ('tracked', 'clean', 'last_commit')}, read_back_probs36=read_back, read_back_groups=rec['read_back'],
                      group_sum_max_abs_deviation=deviation, group_sum_tol=mi.GROUP_TOL, grouping=rec['grouping'], prediction_code=rec['prediction_code'],
                      qc_wm_group=dict(file='structure_groups.npz', name=mi.UNET_WM_GROUP, channel=wm_channel, labels=list(mi.CEREBRAL_WM)))
    del groups
    support = mi.support_dice_record(getattr(pin, side), p0)
    del p0
    bound = getattr(layer, 'allowlist', None)                   # HB (C23.4-8): a hold-out pair's Atropos provenance is rechecked against the bound allowlist load_pair admitted it through
    a_wm, atropos = mi.atropos_wm(seg_root, pin, side, allowlist=bound) if mi.unsealed_allowlist(bound) else mi.atropos_wm(seg_root, pin, side)
    qc = mi.alignment_qc(u_wm, a_wm)
    del u_wm, a_wm
    if not qc['passed']:
        raise SystemExit(f"EF {pin.fixed_subject} <- {pin.moving_subject}:{side}: alignment QC {qc['dice']} < {qc['threshold']} (protocol §12.4 item 5)")
    return labels, dict(record, subject=getattr(pin, side+'_subject'), alignment_qc=qc, support_dice_record=support, atropos_qc=atropos, read_path='old' if fast is None else 'fast')


def bench_read_predicted(pin, bench, layer, *, seg_root, admission, workers=1):
    """fast_channels.read_predicted 的台版本：不经 a4_admission、pair_structure_layer（它们拒绝训练集受试者），经台守卫与 BenchLayer；返回 mc0p_core.Predicted（soft None）。"""
    each = max(1, workers//2)
    rows = core._pool_map(lambda side: bench_read_side(layer, pin, side, seg_root, each), ('fixed', 'moving'), 2 if workers > 1 else 1)
    labels = dict(fixed=rows[0][0], moving=rows[1][0])
    sides = dict(fixed=rows[0][1], moving=rows[1][1])
    keys_read = list(pin.provenance['keys_read'])
    return mc.Predicted(labels, None, dict(**bench_oracle_mark(keys_read, sides), pair_keys_read=keys_read, label_firewall=pin.provenance['label_firewall'],
                                           segmenter=dict(segmenter_id=rc.BENCH_SEGMENTER_ID, registry=layer.registry_record['path'], registry_sha256=layer.registry.sha256, policy=rc.BENCH_POLICY),
                                           atropos_root=str(seg_root), subject_admission=admission, structure_layer=layer.record(), sides=sides))


# ---- 图像头文件（C25.6-4） -------------------------------------------------------------------------------------------------------------------------
def header_record(path):
    """只读头文件（不读体素）：形状、体素、affine、qform / sform 的码与矩阵。"""
    import nibabel as nib
    img = nib.load(str(path))
    h = img.header
    q, qcode = h.get_qform(coded=True)
    s, scode = h.get_sform(coded=True)
    return dict(shape=[int(x) for x in img.shape], zooms=[float(x) for x in h.get_zooms()[:3]], affine=np.asarray(img.affine, dtype=float).round(6).tolist(),
                qform_code=int(qcode), sform_code=int(scode), qform=None if q is None else np.asarray(q).round(6).tolist(), sform=None if s is None else np.asarray(s).round(6).tolist())


def header_problems(record, reference):
    keys = ('shape', 'zooms', 'affine', 'qform_code', 'sform_code', 'qform', 'sform')
    return [f'{k}: {record.get(k)} != {reference.get(k)}' for k in keys if record.get(k) != reference.get(k)]


# ---- 台的 stage（C25.6-5；experiments/holdout/prep_stage.py write_pair 的写法） ------------------------------------------------------------------------
def write_bench_pair(out, fixed, moving, files, *, bench, reference_header, dev_grid):
    """一对：两幅图经 files.image_path 与 load_scalar（labels=False）读到同一网格；pair.npz 恰好 fixed、moving、initial_u；manifest.json 只有不由标签导出、也不提标签的字段。"""
    from jd_curl.three_d.nifti import file_sha256, load_scalar, require_common_grid          # 修订 7：require_common_grid 在 nifti（不在 grid）；experiments/holdout/prep_stage.py:51 同
    admission = bench_admission(bench, pair=pair_name(fixed, moving))               # 打开任何受试者文件之前
    for subject in (fixed, moving):
        problems = header_problems(header_record(files.image_path(subject)), reference_header)
        if problems:
            raise BenchRefused(f'{subject}：头文件与开发集不同：{problems}')
    arrays, sources, reference, grid = {}, {}, None, None
    for side, subject in zip(('fixed', 'moving'), (fixed, moving)):
        volume = load_scalar(files.image_path(subject))
        reference = volume if reference is None else reference
        grid = require_common_grid(reference, volume)
        arrays[side] = volume.values
        sources[side] = dict(source=str(volume.metadata.get('source')), sha256=volume.metadata['sha256'])
    meta = dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes], unit='mm')
    if {k: meta[k] for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz')} != {k: list(dev_grid[k]) for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz')}:
        raise BenchRefused(f'{fixed}, {moving}：网格 {meta} 与开发集 pair.npz 的网格元数据 {dev_grid} 不同')
    arrays['initial_u'] = np.zeros(grid.shape+(3,))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'pair.npz', fixed=arrays['fixed'], moving=arrays['moving'], initial_u=arrays['initial_u'])
    L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)
    manifest = dict(dataset='OASIS (B-line training set, calibration subjects; R4 bench, contract C25.6)', pair=pair_name(fixed, moving), fixed_subject=fixed, moving_subject=moving, **meta,
                    sources=sources, pair_sha256=file_sha256(out/'pair.npz'), weights=dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001),
                    preprocessing='aligned_norm.nii.gz as stored (neurite-oasis), load_scalar (labels=False); no field derived from labels or naming them', bench=dict(fold=admission['fold'], index=admission['index']))
    rc.write_json(out/'manifest.json', manifest)
    return manifest


def write_bench_stage(stage_root, bench, files, *, reference_header, dev_grid, log=print):
    """台的 stage：<R4>/bench_stage/（新目录），selection.json、16 对、SHA256SUMS。"""
    stage_root = Path(stage_root)
    stage_root.mkdir(parents=True, exist_ok=False)
    rc.write_json(stage_root/'selection.json', dict(rule='R4 bench (C25.6-3): within-fold rings of the calibration subjects, (s_i, s_(i+1) mod 8)', pairs=[dict(pair=p[0], fold=p[1], index=p[2]) for p in bench.pairs],
                                                     pairs_table_sha256=pairs_table_sha256(bench), no_metric_based_selection=True))
    out = {}
    for name, _, _, fixed, moving in bench.pairs:
        out[name] = write_bench_pair(stage_root/name, fixed, moving, files, bench=bench, reference_header=reference_header, dev_grid=dev_grid)['pair_sha256']
        log(f'{name}: written')
    names = ['selection.json']+[f'{p[0]}/{n}' for p in bench.pairs for n in ('pair.npz', 'manifest.json')]
    (stage_root/'SHA256SUMS').write_text(''.join(f'{rc.sha256_file(stage_root/n)}  {n}\n' for n in sorted(names)), encoding='utf-8')
    return dict(stage=str(stage_root), pair_sha256=out, sha256sums=rc.sha256_file(stage_root/'SHA256SUMS'))


def load_bench_pair(bench, stage_root, pair):
    """配准一侧的每个进程读台的 pair.npz 的唯一方式：台守卫，再 load_pair（开发白名单；keys_read = fixed、moving、initial_u）。"""
    from bayesreg.io.pair import load_pair
    admission = bench_admission(bench, pair=pair)
    pin = load_pair(Path(stage_root)/pair/'pair.npz', allowlist=bench.world.allowlist_path)
    if (pin.fixed_subject, pin.moving_subject) != tuple(pair.split('__')):
        raise BenchRefused(f'{pair}：manifest 的受试者 {pin.fixed_subject}, {pin.moving_subject} 不是对的名字')
    return pin, admission


# ---- 台的 Atropos（C25.6-9） -------------------------------------------------------------------------------------------------------------------------
ATROPOS_INPUTS = {'ImageMath': ('t1.nii.gz', 'mask.nii.gz'), 'Atropos': ('t1_trunc.nii.gz', 'mask.nii.gz')}


def atropos_argv_problems(argv, cwd, *, ants, out_root, subject):
    """C25.10-5：子进程看不到，只核 argv 与交给它的路径。只许 ANTs 的 ImageMath 与 Atropos；工作目录是 <out_root>/<subject>；
    参数里没有绝对路径、没有 ..、不提 aligned_norm 或 aligned_seg（Atropos 只拿到 atropos_seg.segment 从台的 stage 的 PairInput 写出的 t1.nii.gz 与 mask.nii.gz）。"""
    problems = []
    tool = Path(str(argv[0])) if argv else None
    if tool is None or tool.parent.resolve() != Path(ants).resolve() or tool.name not in ATROPOS_INPUTS:
        problems.append(f'不是 ANTs 的 ImageMath 或 Atropos：{argv[:1]}')
    if Path(cwd).resolve() != (Path(out_root)/subject).resolve():
        problems.append(f'工作目录 {cwd} 不是 {Path(out_root)/subject}')
    for a in map(str, argv[1:]):
        if a.startswith('/') or '..' in a or 'aligned_norm' in a or 'aligned_seg' in a or '/mnt/' in a or 'pair.npz' in a:
            problems.append(f'参数 {a!r} 指向别的文件')
    if tool is not None and tool.name in ATROPOS_INPUTS:
        for name in ATROPOS_INPUTS[tool.name]:
            if name not in map(str, argv) or not (Path(cwd)/name).is_file():
                problems.append(f'{tool.name} 的输入 {name} 不在工作目录里')
    return problems


class AtroposArgv:
    """atropos_seg.segment 的 subprocess 换成这个：每次调用先核 argv 与工作目录（atropos_argv_problems），不过就 BenchRefused（子进程不起）；记录每次调用。"""

    def __init__(self, real, *, ants, out_root, subject):
        self.real, self.ants, self.out_root, self.subject, self.calls = real, ants, out_root, subject, []

    def run(self, argv, **kwargs):
        problems = atropos_argv_problems(argv, kwargs.get('cwd'), ants=self.ants, out_root=self.out_root, subject=self.subject)
        self.calls.append(dict(argv=[str(a) for a in argv], cwd=str(kwargs.get('cwd')), problems=problems))
        if problems:
            raise BenchRefused(f'{self.subject}：Atropos 的启动串不合规则：{problems}')
        return self.real.run(argv, **kwargs)

    def __getattr__(self, name):
        return getattr(self.real, name)


def bench_atropos(bench, stage_root, out_root, *, threads=8, log=print, segment=None):
    """16 人，照 C22.1-3 的规格（experiments/d0/atropos_seg.py segment，未改）；每人的图像经台守卫与 load_pair 读出（两对里同一幅图核相同）。
    子进程的启动串经 AtroposArgv 核（C25.10-5）；记录每人的 argv。"""
    import atropos_seg
    entries = {}
    for name in bench.pair_names:
        pin, _ = load_bench_pair(bench, stage_root, name)
        for side in ('fixed', 'moving'):
            entries.setdefault(getattr(pin, side+'_subject'), []).append((name, side, pin))
    rows, argv = {}, {}
    real = atropos_seg.subprocess
    for subject in bench.ordered_subjects():
        bench_admission(bench, subjects=[subject])
        guard = AtroposArgv(real, ants=atropos_seg.ANTS, out_root=out_root, subject=subject)
        atropos_seg.subprocess = guard
        try:
            rows[subject] = (atropos_seg.segment if segment is None else segment)(subject, entries[subject], Path(out_root), threads, stage_name=Path(stage_root).name)
        finally:
            atropos_seg.subprocess = real
        argv[subject] = dict(calls=guard.calls, stage_pairs={n: p.provenance['pair_sha256'] for n, _, p in entries[subject]})
        log(f'{subject}: Atropos done')
    return rows, argv


# ---- 台的推断（C25.6-8；HS plan 的路径 (a)，两个本折成员） ---------------------------------------------------------------------------------------
def bench_plan(registry, subject, bench, available):
    """[(台模型 id, 成员 sha256, 路径)]：本折两个成员，次序 s0、s1（登记的次序）。"""
    fold = bench.fold_of(subject)
    model = rc.BENCH_MODELS[fold]
    bench_admission(bench, subjects=[subject], model_id=model)
    plan = []
    for sha in registry.models[model]['member_model_sha256']:
        if sha not in available:
            raise BenchRefused(f'{model} 的成员 {sha[:12]}… 找不到')
        plan.append((model, sha, available[sha]))
    return plan, {model: registry.models[model]}


def bench_provenance(subject, *, bench, registry, registry_record, contact, models, members, meta, norm, grouping, grouping_check, allowlist, code, environment, command, seconds):
    model = sorted(models)[0]
    c = contact['models'][model]
    return dict(layer='structure', segmenter_id=rc.BENCH_SEGMENTER_ID, model_ids=[model], model_sha256={model: list(models[model]['member_model_sha256'])}, folds={model: models[model]['fold']},
                trained_with_labels=True, test_labels_seen=rc.BENCH_TEST_LABELS_SEEN, subject=subject, policy=rc.BENCH_POLICY, members=members, **grouping.record(), grouping_check=grouping_check,
                input_sha256={'aligned_norm.nii.gz': meta['sha256']}, grid=dict(shape_zyx=meta['grid_shape_zyx'], spacing_xyz=meta['spacing_xyz'], origin_xyz=meta['origin_xyz']), normalization=norm,
                label_contact=dict(side_file=rc.BENCH_CONTACT_REL, side_file_sha256=registry_record['contact_sha256'], training=list(c['label_contact_training']), calibration=list(c['label_contact_calibration']),
                                   exception='C25.6-7 (D129): within-fold pairs, own-fold UF2, calibration subjects only', what=('not in the members\' training; no step selection; labels read by the temperature '
                                                                                                                                 'calibration and the monitoring every 500 steps')),
                registry_path=rc.BENCH_REGISTRY_REL, registry_sha256=registry.sha256, allowlist_path=allowlist.path, allowlist_sha256=allowlist.sha256,
                composition=dict(rule='UF2: the mean over the own fold\'s two members of softmax(z/T_m) (ensemble_probabilities), float16 as HS stores probs36', members=[m['member_sha256'] for m in members]),
                version=dict(segmenter=rc.BENCH_SEGMENTER_ID), command=command, environment=dict(environment, code=code), runtime_seconds=seconds)


def bench_infer(bench, registry, registry_record, contact, *, pred_root, files, models_root, device, code, command, log=print):
    """16 人，每人一次（做过的不再做）：hs_seg.uf4_probabilities（未改）对本折两个成员，hs_seg.write_prediction（未改）写 probs36.npz、structure_groups.npz、provenance.json。"""
    import torch
    import hs_seg
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping, require_inference_grouping
    from bayesreg.seg.unet.registry import find_member_files
    from bayesreg.seg.unet.train import environment_record
    import time
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    grouping_check = require_inference_grouping(grouping)
    available = find_member_files(models_root)
    environment = {k: v for k, v in environment_record(torch.device(device)).items() if k != 'code'}
    rows = {}
    for subject in bench.ordered_subjects():
        if (Path(pred_root)/subject/'provenance.json').is_file():
            rows[subject] = dict(subject=subject, skipped='provenance.json exists')
            continue
        started = time.perf_counter()
        plan, models = bench_plan(registry, subject, bench, available)
        probs, meta, norm, members = hs_seg.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)
        p16 = probs.astype(np.float16)
        labels = p16.argmax(axis=0)
        del p16
        prov = bench_provenance(subject, bench=bench, registry=registry, registry_record=registry_record, contact=contact, models=models, members=members, meta=meta, norm=norm, grouping=grouping,
                                grouping_check=grouping_check, allowlist=bench.allowlist, code=code, environment=environment, command=command, seconds=time.perf_counter()-started)
        rows[subject] = hs_seg.write_prediction(subject, probs, labels, out_root=pred_root, grouping=grouping, provenance=prov)
        log(f'{subject}: predicted')
    return rows
