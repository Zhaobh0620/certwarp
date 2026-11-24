"""R4（runs/20261113-r4/plan.md；合同 C25，修订 10–10e）：第二轮配准侧筛选台与开发集读数的共用常量与小函数。

只放常量、臂的设定（C25.1-1、C25.5）、选项族（C25.7-6）、钉住的 sha256、JSON 与时间的写法。读任何受试者文件的函数都不在这里。
现行文件（fx_core、fl_channels、mc0_core、HE、HS）一个字不改；它们的常量在这里重写一遍，测试逐项对照（tests/test_bayesreg/r4/test_r4_options.py）。
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
R4_ROOT = RESULTS/'r4_v1'                                   # C25.9-1
RESUME_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')   # 修订 6：续跑记录在 main 仓库里核（已入 git、已提交、工作树一致）；它的 HEAD 不钉
CHECKOUT_PREFIX = 'r4_code_'                                # 修订 6：运行检出 <RESULTS>/r4_code_<短 sha>（C25.9-6"从同一个提交的干净检出跑"；R2 的 source_repo 先例）
SCHEMA = 'bayesreg.r4.{}.v1'
UNIT_PREFIX = 'tag-r4-'                                     # C25.9-1；不得用 seq-vp-*
PLAN_REL = 'runs/20261113-r4/plan.md'
# 总负责人在任何台上运行之前把 plan.md 的 sha256 钉进 decision_log（C25.9-6）：main 8cf08b3 已钉（总负责人约 1:08 PM CST 11-13 转来）；工具核它。
PLAN_SHA256 = 'b8e62d8b6af7d1471b133b7d6ccc3be35d5bad8f0512358621cc7303e923de2c'

# ---- 受试者、折、登记（C25.6-2、D134） ------------------------------------------------------------------------------------------------------------
FOLDS_REL = 'splits/bayesreg_v1/folds_full_v2.json'
FOLDS_SHA256 = '4573b7900b5ae4e7e3e02ff41c0a756f2ff48a8f84a0e7743965ec17b14558dc'      # runs/20261105-uf/plan.md:59
TRAIN_LIST_REL = 'splits/bayesreg_v1/train_subjects.txt'
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'
TRAINING_SOURCE = 'bayesreg_train_v1'
DEV_SOURCE = 'stage_L2R_val_v1'
FOLDS = ('A', 'B')
FOLD_INDEX = dict(A=0, B=1)
N_CALIB, N_TRAIN = 8, 98
HS_REGISTRY_REL = 'configs/bayesreg/segmenters_hs_v1.json'
HS_REGISTRY_SHA256 = 'b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'
HS_FOLD_MODELS = dict(A='unet35_v4_ens_foldA', B='unet35_v4_ens_foldB')
BENCH_REGISTRY_REL = 'configs/bayesreg/segmenters_r4_bench_v1.json'
BENCH_CONTACT_REL = 'configs/bayesreg/segmenters_r4_bench_v1_contact.json'
BENCH_SEGMENTER_ID = 'unet35_v4_uf2_bench'
BENCH_MODELS = dict(A='unet35_v4_uf2_bench_foldA', B='unet35_v4_uf2_bench_foldB')
BENCH_POLICY = 'bench_own_fold_calib'
BENCH_TEST_LABELS_SEEN = 'calibration_only'
CONTACT_SCHEMA = 'bayesreg.r4.bench_contact.v1'
DATA_ROOT = Path('/mnt/d/data')
DATA_LINK_TARGET = Path('/home/rmcfz/code/python_projects/seq_vp_3d/data')   # 仓库 data/ 符号链接的目标（C25.10-5 修订 10e）

# ---- 开发集（C25.8） ------------------------------------------------------------------------------------------------------------------------
DEV_STAGE_REL = 'data/three_d/stage_L2R_val_v1'
HS_ROOT = RESULTS/'hs_v1'
HS_PRED_ROOT = RESULTS/'seg'/'unet35_v4_ens'/'pred'
DEV_ATROPOS_ROOT = RESULTS/'seg'/'atropos_v1'
HS_DECISION_SHA256 = '0b85dd06fb2d859174bcefd61c9bd5e6238760b8aa8ea205de31bb62e1c435c7'     # C25.8-4
HS_REFERENCE_REL = 'experiments/p2/reference_unet35_v4_ens.json'                             # = r2_common（测试对照）
HS_REFERENCE_SHA256 = 'fd157a2bc48522d8ee18bcfd4e6425b5551914c2c02d74c7ad8ddc4f152531df'
SMOKE_DEV_PAIR = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'                                  # C25.10-2

# ---- 臂的设定（C25.1-1） --------------------------------------------------------------------------------------------------------------------
SPEC = dict(k=35, sigma_field=.5, native=600, checkpoints=(600,), channels='onehot', weight='Q3h', D_factor=24.)    # = fx_core.SPEC（测试对照）
OPTIONS = dict(
    A1=dict(field_mask='small6', sigma_small=.42),          # 修订 10a，D140
    A2=dict(field_mask='small6', sigma_small=.35),
    B=dict(channels='soft'),
    C1=dict(D_factor=32.),
    C2=dict(native=1200, checkpoints=(1200,)),
    C3=dict(sigma_field=.65))                               # D131
CONTROL = 'CTL'
CANDIDATES = ('A1', 'A2', 'B', 'C1', 'C2', 'C3')            # 并列与排序的次序（C25.7-5）
ARM_ORDER = (CONTROL,)+CANDIDATES                           # 台上的次序（C25.6-10）
FAMILY = dict(A1='A', A2='A', B='B', C1='C1', C2='C2', C3='C3')     # C25.7-6，D132
INCOMPATIBLE = (frozenset(('A1', 'A2')),)                   # C25.5-2
SMALL_SIX = (4, 23, 18, 34, 19, 35)                         # C25.1-4；= pq_stats.SMALL_SIX（测试对照）
MASK_RADIUS, MASK_SIGMA, MASK_TRUNCATE = 2, 1.0, 4.0        # C25.2-1
COMBO_PREFIX = 'COMBO-'

# ---- 门槛（C25.7-2、C25.5-3、C25.8-4） ------------------------------------------------------------------------------------------------------------
SCREEN_MEAN35, SCREEN_COUNT, SCREEN_MEAN4, SDLOGJ_MAX = .0015, 10, -.001, .10
COMBO_MARGIN = .0005
DEV_MEAN35, DEV_LOWER, DEV_MEAN4, BETTER_LOWER = .003, -.003, -.001, 0.
CONTROL_TOL = 2e-4
BENCH_BOOT_N, BENCH_BOOT_SEED = 10000, 20261113
DEV_BOOT_N, DEV_BOOT_SEED, DEV_N_SUBJECTS = 10000, 20261105, 20
N_BENCH_PAIRS, N_DEV_PAIRS = 16, 19

# ---- 线状态（C22.5-4；= he_core） -------------------------------------------------------------------------------------------------------------
HX, FALLBACK_C, FALLBACK_ID, REJECTED, IMPLEMENTATION = 'hx_certified', 'hx_fallback_configC', 'hx_fallback_identity', 'hx_provenance_rejected', 'hx_implementation_error'
OUTPUT_STATUSES = (HX, FALLBACK_C, FALLBACK_ID)
LINE_STATUSES = OUTPUT_STATUSES+(REJECTED, IMPLEMENTATION)

# ---- 机器门（plan.md 第 1 节；R2、R3 的写法） -------------------------------------------------------------------------------------------------
LOAD_MAX, MEM_AVAILABLE_GIB, HOME_FREE_GB, C_FLOOR_GIB = 1.5, 9., 100., 40.
GATE_WAIT_MINUTES, GAP_SECONDS = 30, 10


class Refused(Exception):
    """读真值之前的拒绝：工具停下，交总负责人（C25.7-3 的"读真值之前的失败"）。"""
    status = 'refused'


# ---- 设定 ---------------------------------------------------------------------------------------------------------------------------------------
def is_combo(arm):
    return isinstance(arm, str) and arm.startswith(COMBO_PREFIX)


def combo_name(first, second):
    parts = sorted((first, second), key=CANDIDATES.index)
    return f'{COMBO_PREFIX}{parts[0]}-{parts[1]}'


def combo_parts(arm):
    if not is_combo(arm):
        return (arm,)
    parts = tuple(arm[len(COMBO_PREFIX):].split('-'))
    if len(parts) != 2 or any(p not in CANDIDATES for p in parts) or parts[0] == parts[1]:
        raise ValueError(f'不是组合臂的名字：{arm!r}')
    return parts


def compatible(a, b):
    return a != b and frozenset((a, b)) not in INCOMPATIBLE


def arm_spec(arm):
    """臂的设定 = SPEC 加上它打开的键（C25.1-1）；组合臂 = 两项各自打开的键的并（C25.5-2）；A1 + A2 拒绝。"""
    parts = combo_parts(arm) if arm != CONTROL else ()
    if arm != CONTROL and not is_combo(arm) and arm not in CANDIDATES:
        raise ValueError(f'没有臂 {arm!r}')
    if len(parts) == 2 and not compatible(*parts):
        raise ValueError(f'{parts[0]} 与 {parts[1]} 不相容（C25.5-2）')
    spec = dict(SPEC)
    keys = set()
    for p in parts:
        clash = keys & set(OPTIONS[p])
        if clash:
            raise ValueError(f'{arm}：两项都开了 {sorted(clash)}')
        keys |= set(OPTIONS[p])
        spec.update(OPTIONS[p])
    return spec


def options_of(spec):
    """打开了哪些选项（C25.9-2 r4_options）。"""
    return sorted(name for name, opts in OPTIONS.items() if all(spec.get(k) == v for k, v in opts.items()) and any(SPEC.get(k) != v for k, v in opts.items()))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=_default)


def spec_sha256(spec):
    """键排序的 JSON 的 sha256（C25.1-1）；元组按列表写。"""
    return hashlib.sha256(canonical({k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}).encode()).hexdigest()


def field_file(spec):
    """C25.4-2、-4：field_<native 步数四位>.npz（关着时 field_0600.npz）。"""
    return f"field_{int(spec['native']):04d}.npz"


def families(arm):
    """一个单项的族集合是 {它的族}；组合的族集合是两个成分的族之并（C25.7-6）。"""
    return frozenset(FAMILY[p] for p in combo_parts(arm))


# ---- 小函数 ------------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha256(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def local_time():
    """Texas 本地时间（America/Chicago）。"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:                                       # noqa: BLE001 - 没有时区数据时只记 UTC
        return None


def _default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return str(value)


def write_json(path, value, *, exclusive=True):
    """只写一次：已有的文件不覆盖（硬约束 4）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, ensure_ascii=False, default=_default)+'\n'
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(text)
    return path


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    except ValueError:
        return None


def under(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def check_results_place(path, *, real):
    """真实运行只写在 <R4> 之下；测试只写在结果根之外（R2、R3 的写法）。"""
    inside = under(path, RESULTS)
    if real and not under(path, R4_ROOT):
        raise Refused(f'{path}：真实运行只写在 {R4_ROOT} 之下')
    if not real and inside:
        raise Refused(f'{path}：测试与合成运行不得写在结果根 {RESULTS} 之下')
    return path


def plan_problem(plan_sha256=None, repo=None):
    """None，或 plan.md 的 sha256 与钉住的不符（C25.9-6）。"""
    want = PLAN_SHA256 if plan_sha256 is None else plan_sha256
    path = Path(ROOT if repo is None else repo)/PLAN_REL
    if want is None:
        return 'plan.md 的 sha256 还没有钉住（r4_common.PLAN_SHA256 是 None）：总负责人钉进 decision_log 并由经复核的提交填上之后才能真实运行'
    if not path.is_file():
        return f'{path} 不存在'
    got = sha256_file(path)
    return None if got == want else f'plan.md 的 sha256 {got} 不是钉住的 {want}'


def pid():
    return os.getpid()
