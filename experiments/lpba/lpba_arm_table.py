"""LP-T0 item 2 (runs/20261119-lpba/任务书_T0_名单臂表与c.md): the LPBA40 second test set's machine-readable arm table, configs/frozen/lpba40_test_v1_arm_table.json.

Basis: the preregistration docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md (final, revision 3, sha256 06121278...) §4 ("the content of the arm table is fixed by
this section": the method, configc, ants_s, fireants_greedy, fireants_syn, convexadam_v4seg and the identity reference; arm names, settings and the "written" rule as the
hold-out's), §11 (everything on /home under /home/rmcfz/seq_vp_tag_results/lpba40_test_v1, no move to D:), §12 item 15.

How the table is made (build_table; mechanical, byte for byte reproducible): from the hold-out's frozen arm table configs/frozen/holdout_B_v1_arm_table.json (sha256 69cb1ddf...,
schema bayesreg.holdout.arm_table.v3, BZ-adopted with the hold-out preregistration's revision 2 item 6) the six arms are taken in its order and copied as they are -- the
terminal records and their classes, the complete files, the review record, the decision certificate, the export check, the certificate mode, the authorization tool --
with only the roots replaced by ROOT_MAP (a prefix map, the same for every string of an arm):
  /home/rmcfz/seq_vp_tag_results/holdout_B_v1/   -> /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/        records, queue records, authorization records, stage
  /mnt/d/seq_vp_tag_holdout_B_v1/                -> /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/   the hold-out's D: places were the moved work cells; LPBA40 does not move (§11),
                                                                                                         so a field stays in its work cell (hb_core.cell, holdout_common.work_dir)
  runs/holdout_B_v1/                             -> runs/lpba40_test_v1/                                  the stop records of P19
The hold-out table's top level changes: schema SCHEMA; stage_root LPBA_ROOT/stage; pq null (PQ is not run on LPBA40, §4.5); segmentation null; two keys added: reference
(the identity row, §4.4: d = 0 through the same evaluation entry, in no family) and frozen (the pins §4 names: the preregistration, the hold-out arm table it is derived
from, the method's settings_sha256, the configuration C reference, the external arms' frozen file).  The table's own sha256 is not in the preregistration (§4, §12 item 15):
the lead commits it and writes it into the decision log and the code pins.

validate_lpba40_arm_table: (1) the form (_validate_form): the LPBA40 form (exact top-level keys, every arm an object with exactly the hold-out's ARM_KEYS, the six arms in
order, the families, pq null, the identity reference, the frozen pins, every path under LPBA_ROOT and none of the hold-out's) and then the hold-out's per-arm form, by calling
bayesreg.eval.holdout.validate_arm_table itself on a shim of the table: the hold-out schema name and two placeholder pq roots (_SHIM_PQ) only to satisfy its top-level checks,
which are LPBA40's own here; the shim is never written.  (2) the content (_validate_content; LP-T0 review important 2): the hold-out arm table is read from the code tree and
must have the pinned sha256 HOLDOUT_TABLE_SHA256; the table must equal build_table of it, key for key (the comment included), compared as
JSON text (json.dumps with sort_keys: true is not 1).  A table of the right form whose arms differ
from the hold-out's in anything but the mapped roots is refused.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = 'bayesreg.lpba40.arm_table.v1'
ARM_TABLE_REL = 'configs/frozen/lpba40_test_v1_arm_table.json'
PREREG_REL = 'docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md'
PREREG_SHA256 = '06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643'
HOLDOUT_TABLE_REL = 'configs/frozen/holdout_B_v1_arm_table.json'
HOLDOUT_TABLE_SHA256 = '69cb1ddf3e2006116c3d2237f3a4912e68a7238f2ce813f7d67723e5eeb8d661'      # the hold-out preregistration revision 2 item 6; bayesreg.eval.holdout.ARM_TABLE_SHA256
BASELINES_REL = 'configs/frozen/holdout_B_v1_baselines.json'
BASELINES_SHA256 = '7c4965ebed65cfe55009518abca72c021309c366f96b8452b9c622a5f53b15a2'          # §4, §4.3
CONFIGC_REL = 'experiments/d0/configC_reference.json'
CONFIGC_SHA256 = 'd331386890d90c8e5fc0c53410a8ffbde64293b1dea3851ccba87e2c9a81c1c3'             # §4.2
METHOD_NAME = 'Hx24-RF05-F1200-v4'
METHOD_SETTINGS_REL = 'experiments/hb/hb_method.py'
METHOD_SETTINGS_SHA256 = '414eccd37fe264d081e87bb63235790dc3a1ec40655c57ac64337bf84992f35d'     # §4.1; hb_method.SETTINGS_SHA256

LPBA_ROOT = '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1'                                    # §11
STAGE_ROOT = f'{LPBA_ROOT}/stage'
ROOT_MAP = (('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/', f'{LPBA_ROOT}/'),
            ('/mnt/d/seq_vp_tag_holdout_B_v1/', f'{LPBA_ROOT}/work/'),
            ('runs/holdout_B_v1/', 'runs/lpba40_test_v1/'))
METHOD_ARM = 'method'
MAIN_FAMILY = ('configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')      # §4, the hold-out's main family and its order
ARM_NAMES = (METHOD_ARM,)+MAIN_FAMILY
EXTERNAL_ARMS = ('ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')               # §4.3
IDENTITY = {'name': 'identity', 'field': 'zero', 'family': None}                                 # §4.4: d = 0 on the pair grid, a reference row, in no family, not tested
FROZEN = {
    'prereg': {'path': PREREG_REL, 'sha256': PREREG_SHA256},
    'holdout_arm_table': {'path': HOLDOUT_TABLE_REL, 'sha256': HOLDOUT_TABLE_SHA256},
    'method': {'name': METHOD_NAME, 'settings_sha256': METHOD_SETTINGS_SHA256, 'constant': f'{METHOD_SETTINGS_REL} SETTINGS_SHA256'},
    'configc': {'path': CONFIGC_REL, 'sha256': CONFIGC_SHA256},
    'external': {'path': BASELINES_REL, 'sha256': BASELINES_SHA256, 'arms': list(EXTERNAL_ARMS)},
}
TOP_KEYS = ('schema', 'comment', 'stage_root', 'main_family', 'pq', 'segmentation', 'reference', 'frozen', 'arms')
FORBIDDEN_TEXT = ('holdout_B_v1', '/mnt/')                     # no place of the hold-out, nothing on D: (§11)
_SHIM_PQ = {'small_root': f'{LPBA_ROOT}/no_pq_small', 'big_root': f'{LPBA_ROOT}/no_pq_big'}      # validation only (module docstring); never in the table

COMMENT = ('LPBA40 第二测试集预登记 docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md（定稿修订 3，sha256 06121278…）第 4 节的机器可读臂表；第 12 节第 15 项。'
           '臂是第 4 节的 6 个：方法 1、主族 5（configc、ants_s、fireants_greedy、fireants_syn、convexadam_v4seg，次序照留出集），另加恒等参照（reference：d = 0，'
           '经同一评测入口，不进任何族，不检验）。由 runs/20261119-lpba/make_lpba40_arm_table.py 从留出集冻结臂表 configs/frozen/holdout_B_v1_arm_table.json'
           '（sha256 69cb1ddf…）机械生成：6 臂照它的次序逐字复制（终态记录与类、完整记录、复核记录、判定证书、导出核对、证书模式、授权 tool），只按前缀换根：'
           '/home/rmcfz/seq_vp_tag_results/holdout_B_v1/ → /home/rmcfz/seq_vp_tag_results/lpba40_test_v1/；/mnt/d/seq_vp_tag_holdout_B_v1/ → '
           '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/（第 11 节：全放 /home，不移到 D:，场留在 work 下的格子里）；runs/holdout_B_v1/ → runs/lpba40_test_v1/。'
           'pq 为 null（第 4.5 节：PQ 不跑）。frozen 记第 4 节点名的冻结设置（方法 settings_sha256、配置 C 参照、外部臂冻结文件）与本表所依据的两份文件。'
           '校验：experiments/lpba/lpba_arm_table.py 的 validate_lpba40_arm_table（LPBA40 的顶层与根，再调 bayesreg.eval.holdout.validate_arm_table 核逐臂的格式）。'
           '本表的 sha256 不写进预登记，由总负责人写进 decision_log 与代码钉值。'
           '附注：configc 的场照留出集取终态记录的 output_store.disp.file，不换键（总负责人 2026-11-19 约 1:01 PM CST 裁决 (1)）。留出集里这个键在移到 D: 时才写；'
           'LPBA 的配置 C 入口在不移动时写 output_store（照 xb_drive.py:402），形状与留出集移动之后相同，文件在 work/configc/<pair>/output 下，补跑在 work/configc/<pair>.resume/output 下；'
           '入口本身留给后面的工具包（第 12 节第 10 项）。')


class ArmTableError(ValueError):
    """The LPBA40 arm table does not have the form of this module."""


def remap(text):
    """One string of an arm with the hold-out's roots replaced (ROOT_MAP, first matching prefix)."""
    for old, new in ROOT_MAP:
        if text.startswith(old):
            return new+text[len(old):]
    return text


def _map_strings(value):
    if isinstance(value, str):
        return remap(value)
    if isinstance(value, list):
        return [_map_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v) for k, v in value.items()}
    return value


def build_table(holdout_table):
    """The LPBA40 arm table (a dict) from the hold-out's arm table (a dict, as loaded): the six arms in the hold-out's order, copied with remap applied to every string."""
    by_name = {arm['name']: arm for arm in holdout_table['arms']}
    missing = [n for n in ARM_NAMES if n not in by_name]
    if missing or list(holdout_table['main_family']) != list(MAIN_FAMILY):
        raise ArmTableError(f'the hold-out table lacks {missing} or its main family is not {list(MAIN_FAMILY)}')
    order = [arm['name'] for arm in holdout_table['arms'] if arm['name'] in ARM_NAMES]
    if order != list(ARM_NAMES):
        raise ArmTableError(f'the hold-out table orders the six arms as {order}')
    arms = [_map_strings(copy.deepcopy(by_name[name])) for name in ARM_NAMES]
    return {'schema': SCHEMA, 'comment': COMMENT, 'stage_root': STAGE_ROOT, 'main_family': list(MAIN_FAMILY), 'pq': None, 'segmentation': None,
            'reference': [dict(IDENTITY)], 'frozen': copy.deepcopy(FROZEN), 'arms': arms}


def dumps(table):
    """The file's bytes: the hold-out table's format (json.dumps, ensure_ascii False, indent 1, one trailing newline), keys in TOP_KEYS order."""
    return (json.dumps(table, ensure_ascii=False, indent=1)+'\n').encode('utf-8')


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def _validate_form(table):
    """(1) of the module docstring.  ArmTableError, or bayesreg.eval.holdout.HoldoutRefused('arms') from the per-arm form."""
    from bayesreg.eval import holdout
    if not isinstance(table, dict) or list(table) != list(TOP_KEYS):
        raise ArmTableError(f'arm table: exactly the keys {list(TOP_KEYS)} in this order')
    if table['schema'] != SCHEMA:
        raise ArmTableError(f'schema {SCHEMA} required')
    if not isinstance(table['comment'], str) or not table['comment']:
        raise ArmTableError('comment: a non-empty string')
    if table['stage_root'] != STAGE_ROOT:
        raise ArmTableError(f'stage_root {STAGE_ROOT} required (§11)')
    if table['pq'] is not None or table['segmentation'] is not None:
        raise ArmTableError('pq and segmentation null (§4.5: PQ is not run on LPBA40)')
    if table['reference'] != [IDENTITY]:
        raise ArmTableError(f'reference: exactly [{IDENTITY}] (§4.4)')
    if table['frozen'] != FROZEN:
        raise ArmTableError('frozen: the pins of §4 exactly (FROZEN)')
    if table['main_family'] != list(MAIN_FAMILY):
        raise ArmTableError(f'main_family {list(MAIN_FAMILY)} in this order (§4)')
    arms = table['arms']
    if not isinstance(arms, list) or any(not isinstance(a, dict) or set(a) != set(holdout.ARM_KEYS) for a in arms):
        raise ArmTableError(f'arms: a list of objects, each with exactly the keys {list(holdout.ARM_KEYS)} (review minor 4)')
    if [a['name'] for a in arms] != list(ARM_NAMES):
        raise ArmTableError(f'arms: exactly {list(ARM_NAMES)} in this order (§4; the identity is a reference, not an arm)')
    families = [a['family'] for a in arms]
    if families != ['method']+['main']*len(MAIN_FAMILY):
        raise ArmTableError(f'families {families}: the method arm is family method, the five others main')
    for k, arm in enumerate(arms):
        for text in _strings(arm):
            if any(bad in text for bad in FORBIDDEN_TEXT):
                raise ArmTableError(f'arms[{k}] ({arm["name"]}): {text!r} names a place of the hold-out or of D: (§11)')
            if text.startswith('/') and not text.startswith(LPBA_ROOT+'/'):
                raise ArmTableError(f'arms[{k}] ({arm["name"]}): {text!r} is not under {LPBA_ROOT} (§11)')
        if not isinstance(arm['stop_record'], str) or not arm['stop_record'].startswith('runs/lpba40_test_v1/'):
            raise ArmTableError(f'arms[{k}] ({arm["name"]}): stop_record under runs/lpba40_test_v1/')
    shim = {'schema': holdout.SCHEMA_ARMS, 'stage_root': table['stage_root'], 'main_family': table['main_family'], 'arms': arms, 'pq': dict(_SHIM_PQ), 'segmentation': None}
    holdout.validate_arm_table(shim)
    return table


def holdout_table_of(repo=REPO):
    """The hold-out arm table of the code tree `repo`, its sha256 the pinned one, read strictly (duplicate keys refused).  ArmTableError otherwise."""
    from bayesreg.eval import holdout
    raw = (Path(repo)/HOLDOUT_TABLE_REL).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != HOLDOUT_TABLE_SHA256:
        raise ArmTableError(f'{HOLDOUT_TABLE_REL}: sha256 {digest} is not the pinned {HOLDOUT_TABLE_SHA256}')
    return holdout.strict_json(raw, HOLDOUT_TABLE_REL)


def _validate_content(table, repo=REPO):
    """(2) of the module docstring: table == build_table(the pinned hold-out table).  ArmTableError naming what differs."""
    expected = build_table(holdout_table_of(repo))
    if json.dumps(table, sort_keys=True) != json.dumps(expected, sort_keys=True):      # diff review minor 1: as JSON text, so True is not 1
        top = [k for k in TOP_KEYS if table.get(k) != expected[k]]
        arms = [e['name'] for a, e in zip(table.get('arms', []), expected['arms']) if a != e]
        raise ArmTableError(f'content: the table is not build_table of the pinned hold-out table (top-level keys that differ {top}; arms that differ {arms})')
    return table


def validate_lpba40_arm_table(table, repo=REPO):
    """The form, then the content (module docstring).  Returns the table; ArmTableError, or bayesreg.eval.holdout.HoldoutRefused('arms') from the per-arm form."""
    _validate_form(table)
    return _validate_content(table, repo)


def load_bytes(raw, repo=REPO):
    """The table from its bytes (bayesreg.eval.holdout.strict_json: duplicate keys refused), validated."""
    from bayesreg.eval import holdout
    return validate_lpba40_arm_table(holdout.strict_json(raw, ARM_TABLE_REL), repo)


__all__ = ['ARM_NAMES', 'ARM_TABLE_REL', 'ArmTableError', 'COMMENT', 'EXTERNAL_ARMS', 'FROZEN', 'IDENTITY', 'LPBA_ROOT', 'MAIN_FAMILY', 'ROOT_MAP', 'SCHEMA', 'STAGE_ROOT',
           'build_table', 'dumps', 'holdout_table_of', 'load_bytes', 'remap', 'validate_lpba40_arm_table']
