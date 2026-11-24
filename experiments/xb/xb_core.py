"""XB (runs/20261107-xb/任务书.md): the external baseline runner of the B-line hold-out and the same-kind competitor ConvexAdam-v4seg.  The rules, the arms and the pieces of the driver; the driver itself
is xb_drive.py, the chain xb_chain.py, the regression check xb_regress.py.

Ported from archive/seq_vp_3d/results/verification/3d_stage_BL2VAL_v1/harness/drive.py (sha256 48e447ca…; "drive.py" below).  What is the same: the arms' commands and environments, the method scripts
(experiments/xb/harness/m_*.py, byte-identical copies; a test compares their sha256 with the archive's), the prechecks (MemAvailable >= 6 GB, the GPU idle, 15 minutes), the memory guard (1 GB), the export
to our convention (d (3, z, y, x), voxels, warped(p) = moving(p + d(p))), the export check (map_coordinates order 1, mode constant, cval 0, inside fixed > 0, corr >= .999), the timing definitions.
What changed, each because the preregistration (docs/bayesreg/notes/留出集预登记_20261107.md §2.3 (a)-(h)) requires it:
  (a) the method gets a derived npz holding fixed and moving only, copied byte for byte (derive_inputs); the driver reads the original through bayesreg.io.pair's loader hook (_open_npz), the two members
      only, never listing the archive; drive.py's load_pair (all members) and arm_py's `target = NPZ` are gone;
  (b) sanity() is gone, and with it the Dice in the driver log and the summary;
  (c) the attempt count and BL2_ATTEMPT are gone: one lock per (arm, pair) (holdout_common.take_lock, after every check), a rerun only through a resume record;
  (d) the chain is new (xb_chain.py);
  (e) the spies (xb_spy.py);
  (f) the failure classes (classify, the terminal record's status);
  (g) the roots (holdout_common) and the unit's settings (UNIT_PROPERTIES);
  (h) the records (xb_drive.result_document).
The terminal record's status (prereg §4.5, §5.4; the lead's addition of 2026-11-07):
  ok                  the export written and checked (the check's suspicious flag is a field: an export check below .999 is not a failure, prereg §4.5; P19 stops the arm at the third; on LPBA40 at the
                      second: p19_stop_at)
  method_failure      the method itself: an error exit, an incomplete export (no disp.npz, a non-finite or misshapen field, a missing transform)
  unit_limit          the unit's RuntimeMaxSec without any OOM evidence (rule (乙), the default); under rule (甲) also the unit's own memory limit; a method failure for the analysis, not rerun
  infra_failure       prereg §5.4's list and revision 1c item 1 (乙): any OOM evidence of the unit or of a step (memory.events oom / oom_kill, a dmesg oom-kill line of its task, the journal, systemd's
                      Result oom-kill, a step's exit -9; read on every exit), a CONSTRAINT_NONE line in the window, the driver's memory guard, a precheck that did not pass in 15 minutes, a disk
                      guard, a move that failed or whose sha256 differ, a spy breach (before the lock: a refusal record; after it: the status), an input refused after the lock; the batch stops
  wrapper_error       our code raised (the driver's or the ConvexAdam-v4seg wrapper's own code, not the method's); WRAPPER_ERROR_POLICY says whether the batch goes on (the cell counts as identity,
                      prereg §4.5 (2)) or stops for a resume after the fix; the lead and BZ decide which
  admission_rejected  (no longer written into a cell: every refusal before the lock is a refusal record in the batch directory, I1; a rerun needs a committed authorization)
  needs_lead_rejudgment  a kill or a limit whose evidence could not be read (holdout_common.classify_unit_end: never read as 0); the batch stops, the lead decides
Unit limits and machine failures: holdout_common.classify_unit_end (prereg revision 1c item 1), rule (乙) by default: a step's exit -9 or any OOM evidence of the unit is infra_failure, a timeout
without OOM evidence unit_limit; rule (甲) (holdout_common.LIMIT_RULE) reads the unit's own memory limit as unit_limit.
A wrapper error, by default (the lead, prereg revision 1b): the batch stops; the implementation is fixed; a rerun only with a committed resume record and the frozen configuration's sha256 unchanged
(the first run's lock line has it); a second wrapper error is "no output (wrapper)", no third run.  An infrastructure failure that BZ does not let rerun gets a committed no_resume authorization
("未批补跑", I3), which makes the cell final.  A record the unit's evidence contradicts keeps its file; the chain writes overridden.json beside it, which supersedes it (I2: effective_record).

LP-B (contract C26.5-6, C26.4 L33; LPBA40 preregistration 06121278... §4.3, §6.6, §8.2 steps 6, 9, §12 item 11; brief item 3): the pair sets 'lpba40_test' (formal) and 'lpba40_feas' (feasibility).
  arms       only LPBA_ARMS (ants_s, fireants_greedy, fireants_syn, convexadam_v4seg; prereg §4.3), their frozen configuration that of configs/frozen/holdout_B_v1_baselines.json (7c4965eb..., checked as
             for the hold-out, check_frozen); the harness and m_*.py unchanged
  roots      the mode's root (experiments/lpba/lpba_arms.root): records <root>/<arm>/<pair>, work <root>/work/<arm>/<pair>, locks, batches; no store (nothing moves to D:, prereg §11): the kept files stay
             in the work cell, disp.file names them there (the arm table's field root <root>/work/<arm>)
  the stage  lpba_common.roots(pair_set).stage/<pair>/pair.npz (pair names LPBA40_<f>__LPBA40_<m>, subjects S<nn>: subjects_of)
  HI         lpba_common.roots(pair_set).pred_root (hi_pred_root; HI's record subjects/<s>.json beside pred/, hi_lpba.py's)
  method     the method's t88/run.json of an LPBA pair: its cell's (<root>/work/method/<pair>/t88/run.json, nothing moves) at the sha256 of the mode's finalized_outputs.json (step 5 first; method_t88_lookup)
  P19        P19_STOP_AT stays 3 for the hold-out and the development; an LPBA40 arm stops at its second suspicious pair (p19_stop_at: P19_STOP_AT_LPBA; prereg §6.6 裁决 (8); contract revision 11d)
  units      tag-lp-x-<arm>-<Sff>-<Smm> (tag-lp-feas-x-...), the chain tag-lp-x-chain (tag-lp-feas-x-chain); C26.2-5
A rerun, resume or no_resume record of an LPBA mode is read only when holdout_common knows its authorizations (C26.8-3, D148: LP-C2's; the feasibility mode has none): authorizations_problem.
"""
from dataclasses import dataclass
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
for _p in (EXPERIMENTS/'holdout', HERE, EXPERIMENTS/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
if str(EXPERIMENTS/'lpba') not in sys.path:     # LP-B: lpba_arms (imported only when an LPBA mode is met, lpba()) findable; appended last
    sys.path.append(str(EXPERIMENTS/'lpba'))

import holdout_common as hc  # noqa: E402
import xb_spy  # noqa: E402

SCHEMA = 'bayesreg.xb.{}.v1'
ARCHIVE_HARNESS = 'archive/seq_vp_3d/results/verification/3d_stage_BL2VAL_v1/harness'
ARCHIVE_DRIVE_SHA256 = '48e447caa40b5657d32377961486dfb9122d7f7bba452a29682935291731e4e5'
BL2_PROTOCOL = 'archive/seq_vp_3d/results/verification/3d_stage_BL2_v1/plan/protocol.md'
BL2_PROTOCOL_SHA256 = '8dc597ee6d8b895d6210a39434b51d66a00e0eea850f83b4518c2e3558a6acf2'
FROZEN_CONFIG_REL = 'configs/frozen/holdout_B_v1_baselines.json'
FROZEN_CONFIG_SHA256 = '7c4965ebed65cfe55009518abca72c021309c366f96b8452b9c622a5f53b15a2'     # configs/frozen/holdout_B_v1_baselines.json as build_frozen_config.py wrote it; a test checks the pin
ENV_RECORDS_REL = 'archive/seq_vp_3d_bl2_env_20261107'
PREREG_REL = 'docs/bayesreg/notes/留出集预登记_20261107.md'
PREREG_SHA256 = 'df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493'     # I5; P0 (11-14), after BZ adopted the preregistration's revision 2: None or another sha256 refuses a hold-out run (a dry run reports it)
SMOKE_CMD_REL = 'archive/seq_vp_3d/results/verification/3d_stage_BL2_v1/env'
SMOKE_CMD_SHA256 = dict(                        # m11: prereg §5.1 item 9, the archive's smoke_cmd.md of each environment record the runner uses (sha256 of the repository's copies)
    ants='fc9476b6138dfa24404003ce5d41478fee30088d62d0a252be2c0c72c1a38566', bl_fireants='b52b74ade654a7521f4589649de664f4adc42779900d73877ee3ac6b953737cf',
    bl_unigradicon='22c5e65d82b7c3c34ebe317b5300f6d06f80d31b97e226d0eee8bdca79fdd1d2', convexadam='973178ab79f518b718cf2b36497cd7876f4ff919b1a23288a5d2749aca440998',
    deedsBCV='aed99bb645330197a361cb1260010f84b5d9986eb63ed0ebd5db834831e184d2', niftyreg='4d7be97dc719a3ba49fbc2abaab5b84af8bc52f5e0a9199c92af0cf1aa41c97f',
    voxelmorph='9ca1748602ab4f5f2a82a4019aed9027b8ca85e0ee36057592f5b19aa0933201')
SMOKE_CMD_OF_GROUP = dict(fireants='bl_fireants', ants='ants', deeds='deedsBCV', convexadam='convexadam', unigradicon='bl_unigradicon', voxelmorph='voxelmorph')


def prereg_problem(repo=None):
    """None when the preregistration file's sha256 is the pinned PREREG_SHA256; the problem otherwise (I5)."""
    path = Path(REPO if repo is None else repo)/PREREG_REL
    if PREREG_SHA256 is None:
        return 'PREREG_SHA256 is None: the preregistration is not pinned in this code (the lead fills it after BZ adopts revision 2)'
    digest = sha256(path) if path.is_file() else None
    if digest != PREREG_SHA256:
        return f'{PREREG_REL}: sha256 {digest} is not the pinned {PREREG_SHA256}'
    return None


PY = Path('/home/rmcfz/miniconda3/envs/sci310/bin/python')
CONDA_BIN = Path('/home/rmcfz/miniconda3/bin/conda')
ENVS = Path('/home/rmcfz/miniconda3/envs')
NVSMI = '/usr/lib/wsl/lib/nvidia-smi'
BASELINES = Path('/home/rmcfz/opt/baselines')
ANTS_BIN = BASELINES/'ants/ants-2.6.5/bin'
NR_BIN = ENVS/'bl_niftyreg/bin'
DEEDS_BIN = BASELINES/'deedsBCV/bin'
MEM_MIN_GB = 6.0
MEM_GUARD_GB = 1.0                              # drive.py: kill the method's process group below this MemAvailable
WAIT_MAX_S = 15*60
GPU_IDLE_MEM_MIB = 2500
GPU_IDLE_UTIL = 10
EXPORT_CORR_MIN = .999
P19_STOP_AT = 3                                 # prereg §4.5 P19: the third pair of an arm whose export check is suspicious stops the arm (the hold-out's; LP-B: per pair set, p19_stop_at)
LPBA_PAIR_SETS = ('lpba40_test', 'lpba40_feas')                      # LP-B (C26.6-2: one value per mode; lpba_common.PAIR_SETS, a test compares)
P19_STOP_AT_LPBA = 2                            # LP-B: LPBA40 prereg §6.6 (裁决 (8)), §12 item 11: the second suspicious pair stops an LPBA40 arm; contract C26.13 revision 11d (the hold-out keeps 3)
LPBA_ARMS = ('ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')                  # LP-B: LPBA40 prereg §4.3 (the main family's four external arms)
UNIT_PROPERTIES = ('MemoryMax=10G', 'MemorySwapMax=0', 'OOMPolicy=continue', 'RuntimeMaxSec=1800')     # prereg §2.3 (g); BL2VAL protocol_notes.md:23
CHAIN_UNIT = 'tag-hb-x-chain'
UNIT_PREFIX = 'tag-hb-x-'
WRAPPER_ERROR_POLICIES = ('stop', 'identity')
WRAPPER_ERROR_POLICY = 'stop'                   # the lead and BZ decide (the lead's addition 11-07): stop = the batch stops, a resume after the fix; identity = recorded, counted as identity, the batch goes on
V4SEG_WRAPPER_ERROR_EXIT = 3                    # m_convexadam_v4seg.py: its own code raised (not convexAdam's)
SPYRUN_BREACH_EXIT = xb_spy.SPYRUN_BREACH_EXIT
DEV_ROOT = hc.RESULTS/'xb_dev_v1'               # development runs (the regression check, ConvexAdam-v4seg's 19 pairs): records, work and store all on /home; never D:
STATUSES = ('ok', 'method_failure', 'unit_limit', 'infra_failure', 'wrapper_error', 'needs_lead_rejudgment')       # a refusal before the lock is a refusal record, not a status (I1)
EXIT = dict(ok=0, method_failure=1, unit_limit=1, infra_failure=20, wrapper_error=23, admission_rejected=2, refused=2, needs_lead_rejudgment=25)
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'ANTS_RANDOM_SEED', 'PYTHONPATH', 'PYTHONHOME', 'CUDA_VISIBLE_DEVICES',
               'TF_FORCE_GPU_ALLOW_GROWTH')     # drive.py, word for word
TIMING_DEFINITIONS = """seconds_registration      the registration call itself, first call in a fresh process ("cold"), CUDA context created
                            before the timer where the method's API allows it; CLI methods: the registration command
  seconds_registration_warm the same call repeated in the same process (GPU Python-API arms only; timing information only,
                            the exported field is the cold run's)
  seconds_total             fresh process launch (interpreter start, imports, input writing, model/weight loading) to the
                            method's own outputs being available (in memory or written by the method's own tools);
                            excludes this harness's conversion to our convention, the export check and compressed saving."""     # drive.py's docstring, word for word

# ---- the arms ------------------------------------------------------------------------------------------------------------------------------------
# role: main (prereg §4.4 F1/F2), descriptive (F3), tier90 (the 90 s tier, §2.6: F3).  store: where the kept big files of a hold-out run go: D: for every arm (decision_log 2026-11-10 ~1:18 AM CST
# ruling (7), prereg revision 2 item 8's allocation: the descriptive arms and the 90 s tier moved from /home to D: as well, A5a; the C: side could not hold them); a change of this table is a reviewed
# change of the code.  baselines: the groups of /home/rmcfz/opt/baselines files the frozen configuration pins for the arm.
ARMS = {
    'fireants_greedy':    dict(kind='py', device='cuda', env='bl_fireants', script='harness/m_fireants.py', args=['greedy'], post_args=[], role='main', store='d', baselines=['fireants'], clones=['fireants']),
    'fireants_syn':       dict(kind='py', device='cuda', env='bl_fireants', script='harness/m_fireants.py', args=['syn'], post_args=[], role='main', store='d', baselines=['fireants'], clones=['fireants']),
    'ants_s':             dict(kind='ants', device='cpu', variant='s', role='main', store='d', baselines=['ants'], clones=[]),
    'ants_so':            dict(kind='ants', device='cpu', variant='so', role='descriptive', store='d', baselines=['ants'], clones=[]),
    'niftyreg':           dict(kind='niftyreg', device='cpu', env='bl_niftyreg', role='descriptive', store='d', baselines=[], clones=[]),
    'deeds':              dict(kind='deeds', device='cpu', role='descriptive', store='d', baselines=['deeds'], clones=['deeds']),
    'convexadam':         dict(kind='py', device='cuda', env='bl_convexadam', script='harness/m_convexadam.py', args=[], post_args=[], role='descriptive', store='d', baselines=['convexadam'], clones=['convexadam']),
    'unigradicon_io':     dict(kind='py', device='cuda', env='bl_unigradicon', script='harness/m_unigradicon.py', args=['io'], post_args=[], role='descriptive', store='d', baselines=['unigradicon'], clones=['unigradicon']),
    'unigradicon_noio':   dict(kind='py', device='cuda', env='bl_unigradicon', script='harness/m_unigradicon.py', args=['noio'], post_args=[], role='descriptive', store='d', baselines=['unigradicon'], clones=['unigradicon']),
    'voxelmorph':         dict(kind='py', device='cuda', env='bl_synthmorph', script='harness/m_voxelmorph.py', args=[], post_args=[], role='descriptive', store='d', baselines=['voxelmorph'], clones=[],
                               extra_env=dict(TF_FORCE_GPU_ALLOW_GROWTH='true')),
    # the 90 s tier (prereg §2.6, P10): the factor fixed here, as BL2VAL passed it by BL2_FIREANTS_K / BL2_UNIGRADICON_N (drive.py:479-484): k = 22, 27 (Amendment 6), N* = 123 (BL2VAL README)
    'fireants_greedy_eq': dict(kind='py', device='cuda', env='bl_fireants', script='harness/m_fireants.py', args=['greedy'], post_args=['22'], role='tier90', store='d', baselines=['fireants'], clones=['fireants'],
                               factor=dict(name='k', value=22)),
    'fireants_syn_eq':    dict(kind='py', device='cuda', env='bl_fireants', script='harness/m_fireants.py', args=['syn'], post_args=['27'], role='tier90', store='d', baselines=['fireants'], clones=['fireants'],
                               factor=dict(name='k', value=27)),
    'unigradicon_io_eq':  dict(kind='py', device='cuda', env='bl_unigradicon', script='harness/m_unigradicon.py', args=['io'], post_args=['123'], role='tier90', store='d', baselines=['unigradicon'],
                               clones=['unigradicon'], factor=dict(name='N', value=123)),
    # the same-kind competitor (prereg §2.4, P3): convexAdam.convex_adam_nnUNet with its argparse defaults on the two v4 argmax label maps
    'convexadam_v4seg':   dict(kind='v4seg', device='cuda', env='bl_convexadam', script='m_convexadam_v4seg.py', args=[], post_args=[], role='main', store='d', baselines=['convexadam'], clones=['convexadam']),
}
ORDER = list(ARMS)
NOT_RUN = dict(transmorph_large='P8: trained with Dice on the L2R OASIS training set, which holds every hold-out subject (prereg §2.5)', transmorph_base='P8', synthmorph_deform='not runnable on this machine (BL2 Amendment 2)',
               synthmorph_joint='not runnable on this machine', vpreg='P9')
KEEP_GIB = dict(fireants_greedy=.1, fireants_syn=.1, ants_s=.7, ants_so=.7, niftyreg=.2, deeds=.11, convexadam=.07, unigradicon_io=.45, unigradicon_noio=.45, voxelmorph=.1, fireants_greedy_eq=.1,
                fireants_syn_eq=.1, unigradicon_io_eq=.27, convexadam_v4seg=.1)          # GiB per run before the intermediates are removed: ceil-ish 1.2 x BL2's smoke big directories (protocol_notes.md:59-72)
KEEP_FILES = ('disp.npz', 'own_warped.npz', 'own_warped_alt.npz', 'method_out.json', 'method_log.txt', 'maps.npz', 'spy_method.jsonl', 'spy_policy.json')
V4SEG_ANCHOR = dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                    registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1', test_labels_seen=False)      # C22.1-2 修订 4 (HI's anchor)
V4SEG_NNUNET_SHA256 = '9dab109177c98cd9ecda72c8b1b114a46c0a7f30ce860c32d4fcf493943d7f58'          # src/convexAdam/convex_adam_nnUNet.py @ b229e52 (prereg §2.4)
HI_PRED_ROOTS = dict(holdout=hc.RESULTS/'seg'/'unet35_v4_ens_holdout_B_v1'/'pred',                  # hi_core.HOLDOUT_OUT_SUGGESTED/pred: the hold-out entry's one root (the first run and its resume; a test compares)
                     dev=hc.RESULTS/'seg'/'unet35_v4_ens'/'pred')                                     # hi_core.REGISTERED_PRED: the 20 development subjects (HS's seg phase)
N_CLASSES = 36
RAW_SUM_TOL_FLOAT16 = 1e-3                      # bayesreg.seg.channels.RAW_SUM_TOL['float16'] (C03.2-4a, D39); a test compares


class MethodFailure(Exception):
    """The method itself failed (prereg §4.5 (1))."""


class UnitLimit(Exception):
    """A unit limit stopped a step (prereg §4.5 (3))."""


class WrapperError(Exception):
    """Our wrapper's own code raised (prereg §4.5 (2)); also the class of any unexpected exception after the lock."""


class NeedsRejudgment(Exception):
    """A killed step whose evidence could not be read (holdout_common.classify_unit_end)."""


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def sha256(path):
    return hc.sha256_file(path)


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def is_lpba(pair_set):
    return pair_set in LPBA_PAIR_SETS


def lpba():
    """experiments/lpba/lpba_arms.py (and lpba_common through it), imported when an LPBA mode is first met (its directory on sys.path from this module's import)."""
    import lpba_arms
    return lpba_arms


def p19_stop_at(pair_set):
    """P19's stop: the hold-out's and the development's third suspicious pair (P19_STOP_AT, unchanged, read at call time), an LPBA40 mode's second (P19_STOP_AT_LPBA; LPBA40 prereg §6.6)."""
    return P19_STOP_AT_LPBA if is_lpba(pair_set) else P19_STOP_AT


def subjects_of(pair_set, pair):
    """(fixed, moving) subject ids of a pair name: <fixed>__<moving> (holdout_common.split_pair), an LPBA40 pair LPBA40_<f>__LPBA40_<m> -> (S<f>, S<m>) (lpba_arms.split)."""
    return lpba().split(pair) if is_lpba(pair_set) else hc.split_pair(pair)


def authorizations_problem(pair_set):
    """LP-B: None for the hold-out and the development; for an LPBA mode whose committed authorizations holdout_common does not know (AUTH_DIRS; C26.8-3, D148: LP-C2's; the feasibility mode has
    none), why a rerun, resume or no_resume record cannot be read."""
    if not is_lpba(pair_set) or (pair_set in hc.AUTH_DIRS and pair_set in hc.AUTH_PREFIX):
        return None
    return (f'{pair_set}: no committed authorizations of this mode are known to holdout_common (AUTH_DIRS; C26.8-3, D148: the formal mode\'s are LP-C2\'s, the feasibility mode has none -- a '
            'feasibility failure is fixed in the code and run again on a new root, C26.8-5)')


def roots(pair_set):
    """(record root, work root, store root of 'd' arms or None) of a pair set."""
    if pair_set == 'holdout':
        return hc.HOLDOUT_ROOT, hc.HOLDOUT_ROOT, hc.BIG_ROOT
    if pair_set == 'dev':
        return DEV_ROOT, DEV_ROOT, None
    if is_lpba(pair_set):                       # LP-B: the mode's root; no store (nothing moves to D:, prereg §11)
        root = lpba().root(pair_set)
        return root, root, None
    raise hc.Refused('admission', f'pair set {pair_set!r}: holdout, dev, {LPBA_PAIR_SETS[0]} or {LPBA_PAIR_SETS[1]}')


def cell_paths(arm, pair_set, pair):
    record_root, work_root, big = roots(pair_set)
    record = hc.record_dir(arm, pair, record_root)
    store = hc.big_dir(arm, pair, big) if (big is not None and ARMS[arm]['store'] == 'd') else None
    return dict(record=record, work=hc.work_dir(arm, pair, work_root), store=store, lock=Path(record_root)/'locks'/f'xb__{arm}__{pair}.lock',
                result=record/'result.json', resume_result=record/'resume'/'result.json', arm_stop=hc.arm_dir(arm, record_root)/'arm_stopped.json',
                overridden=record/'overridden.json', resume_overridden=record/'resume'/'overridden.json', batches=hc.batch_root(record_root))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def effective_record(paths, resume=False):
    """I2: the terminal record of an attempt as the cell reads it: overridden.json (written by the chain when the unit's evidence contradicted the driver's record; it supersedes that record, whose
    file stays) when it is there, else result.json; None when neither is."""
    over = read_json(paths['resume_overridden' if resume else 'overridden'])
    return over if over is not None else read_json(paths['resume_result' if resume else 'result'])


def attempt_complete(paths, resume=False):
    """Prereg §5.4 "完整记录" for an attempt, the overriding record counted (I2)."""
    doc = effective_record(paths, resume)
    return doc is not None and complete(doc)


# I4: the method's T88 run.json of a pair (prereg §2.4: ConvexAdam-v4seg's maps must be the method's labels).  Development: HS's T88 fields of Hx24-RF05-F1200-v4 (hs_v1, the method without C:
# the labels are the v4 argmax maps either way).  Hold-out (HB pins it, runs/20261109-hb/实现说明.md): the method arm's cell on D: after step 5's finalization and the move (experiments/hb/hb_store.py
# move: the whole working cell holdout_B_v1/work/method/<pair>/ to /mnt/d/seq_vp_tag_holdout_B_v1/method/<pair>/, sha256 before and after), so ConvexAdam-v4seg (step 6) runs after the move; before
# it the file is not there and the cell is refused before its lock (method_run_check).
HS_RUNS = hc.RESULTS/'hs_v1'/'runs'
HB_METHOD_ARM = 'method'                        # experiments/he/he_holdout.py ARM (a test compares)


HB_FINALIZED = ('review', 'finalized_outputs.json')     # HB's step-5 record under holdout_common.HOLDOUT_ROOT (runs/20261109-hb/实现说明.md section 6.1)
HB_T88_RUN = 't88/run.json'


def method_t88_lpba(pair_set, pair):
    """LP-B: (found, why) of an LPBA40 pair: the mode's finalized record <root>/review/finalized_outputs.json (step 5) names pairs[pair].part.t88_repair['t88/run.json'] by sha256 and its cell, which must be
    the method cell <root>/work/method/<pair> (nothing moves, prereg §11); found = (that cell's t88/run.json, the sha256).  Before step 5 None (ConvexAdam-v4seg is refused until then)."""
    la = lpba()
    root = la.root(pair_set)
    final_path = Path(root).joinpath(*HB_FINALIZED)
    final = read_json(final_path)
    row = ((final or {}).get('pairs') or {}).get(pair) or {}
    want = ((row.get('part') or {}).get('t88_repair') or {}).get(HB_T88_RUN)
    if not want:
        return None, (f"{final_path}: no part.t88_repair['{HB_T88_RUN}'] for {pair} (step 5 not done, or the method arm has no output on the pair)" if final is not None
                      else f'{final_path}: not there (step 5 not done)')
    cell = la.work_dir(HB_METHOD_ARM, pair, pair_set)
    if Path(row.get('cell') or '') != Path(cell):
        return None, f"{final_path}: the cell of {pair} is {row.get('cell')}, not the method cell {cell}"
    return (Path(cell)/HB_T88_RUN, want), None


def method_t88_lookup(pair, pair_set='holdout'):
    """(found, why): method_t88_holdout's answer and, when there is none, why (A5b diff review minor 3: a move row without dst is named as the move record's format).  LP-B: an LPBA pair's
    (method_t88_lpba)."""
    if is_lpba(pair_set):
        return method_t88_lpba(pair_set, pair)
    final_path = Path(hc.HOLDOUT_ROOT).joinpath(*HB_FINALIZED)
    final = read_json(final_path)
    row = ((final or {}).get('pairs') or {}).get(pair) or {}
    want = ((row.get('part') or {}).get('t88_repair') or {}).get(HB_T88_RUN)
    if not want:
        return None, (f"{final_path}: no part.t88_repair['{HB_T88_RUN}'] for {pair} (step 5 not done, or the method arm has no output on the pair)" if final is not None
                      else f'{final_path}: not there (step 5 not done)')
    record = Path(hc.record_dir(HB_METHOD_ARM, pair))
    why = f"{record}: no ok row of {HB_T88_RUN} at the finalized sha256 {want} in move.json or move_resume.json (the move not done)"
    for name in ('move.json', 'move_resume.json'):
        move = read_json(record/name) or {}
        for i, r in enumerate(move.get('files') or []):
            if isinstance(r, dict) and r.get('file') == HB_T88_RUN and r.get('ok') is True and r.get('sha256_after') == want:
                if isinstance(r.get('dst'), str):
                    return (Path(r['dst']), want), None
                why = (f"{record/name}: files[{i}] ({HB_T88_RUN}) has no dst: the move record's format (holdout_common.move_files writes file, src, dst, ok, sha256_before, "
                       "sha256_after)")
    return None, why


def method_t88_holdout(pair):
    """HB review m13 (the alignment's revision 1 item 1): the method's t88/run.json on a hold-out pair is known by HB's records, never by a path template: the finalized record's
    pairs[pair].part.t88_repair['t88/run.json'] sha256, and the move record's row of that file (holdout_B_v1/method/<pair>/move.json, or a passed move_resume.json): ok, its sha256_after that sha256,
    its dst the path.  (path, sha256), or None while either record or the row is not there (step 5 and the move not done: ConvexAdam-v4seg is refused until then); a row without dst is no row
    (method_t88_lookup says why)."""
    return method_t88_lookup(pair)[0]


def method_t88_run(pair_set, pair):
    """The path of the method's t88/run.json for this pair: the development pairs' pinned path (HS's runs); a hold-out pair's from HB's finalized and move records (method_t88_holdout); None
    when there is none."""
    if pair_set == 'dev':
        short = hc.short_pair(pair)
        first5 = hc.DEV_PAIRS[:5]
        return Path(HS_RUNS)/('SHx24_first5' if pair in first5 else 'SHx24_rest600')/short/'run.json'
    if pair_set == 'holdout':
        found = method_t88_holdout(pair)
        return None if found is None else found[0]
    if is_lpba(pair_set):
        found = method_t88_lpba(pair_set, pair)[0]
        return None if found is None else found[0]
    return None


def method_t88_sha256(pair_set, pair):
    """The sha256 the file must have (a hold-out pair: the finalized record's), or None (the development pairs: no record pins it).  LP-B: an LPBA pair's finalized record's."""
    found = method_t88_holdout(pair) if pair_set == 'holdout' else method_t88_lpba(pair_set, pair)[0] if is_lpba(pair_set) else None
    return None if found is None else found[1]


def pair_path(pair_set, pair):
    if is_lpba(pair_set):                       # LP-B: the mode's stage (lpba_common.roots)
        stage = lpba().lc.roots(pair_set).stage
        lpba().lc.guard_write(stage)
        return Path(stage)/hc.safe_name(pair, 'pair')/'pair.npz'
    return hc.stage_pair_path(pair) if pair_set == 'holdout' else hc.DEV_STAGE/hc.safe_name(pair, 'pair')/'pair.npz'


def hi_pred_root(pair_set):
    """HI's predictions of a pair set: HI_PRED_ROOTS (the hold-out's and the development's, read at call time); an LPBA mode's lpba_common.roots(pair_set).pred_root (LP-B)."""
    if is_lpba(pair_set):
        return Path(lpba().lc.roots(pair_set).pred_root)
    return HI_PRED_ROOTS[pair_set]


def unit_name(arm, pair, pair_set='holdout'):
    if is_lpba(pair_set):                       # LP-B (C26.2-5): tag-lp-x-<arm>-<Sff>-<Smm>, tag-lp-feas-x-...
        return lpba().unit_name(f'x-{arm}', pair_set, pair)
    return f'{UNIT_PREFIX}{arm}-{hc.short_pair(pair)}'


def chain_unit(pair_set='holdout'):
    """The chain's own unit: tag-hb-x-chain (CHAIN_UNIT, the hold-out's and the development's); an LPBA mode's tag-lp-x-chain (tag-lp-feas-x-chain)."""
    return lpba().unit_prefix('x-chain', pair_set) if is_lpba(pair_set) else CHAIN_UNIT


# ---- the frozen configuration -------------------------------------------------------------------------------------------------------------------
def frozen_config(repo=None, pinned=None):
    """configs/frozen/holdout_B_v1_baselines.json, its sha256 against the pin.  Refused (stage frozen) otherwise."""
    repo = Path(REPO if repo is None else repo)
    pinned = FROZEN_CONFIG_SHA256 if pinned is None else pinned
    path = repo/FROZEN_CONFIG_REL
    if pinned is None:
        raise hc.Refused('frozen', 'the frozen configuration is not pinned in this code (FROZEN_CONFIG_SHA256 is None)')
    if not path.is_file():
        raise hc.Refused('frozen', f'{FROZEN_CONFIG_REL} is missing')
    digest = sha256(path)
    if digest != pinned:
        raise hc.Refused('frozen', f'{FROZEN_CONFIG_REL}: sha256 {digest} is not the pinned {pinned}')
    return json.loads(path.read_text(encoding='utf-8')), dict(path=FROZEN_CONFIG_REL, sha256=digest)


EXECUTABLE_KEYS = ('kind', 'device', 'env', 'script', 'args', 'post_args', 'variant', 'role', 'factor', 'extra_env')


def check_frozen(arm, *, repo=None, baselines=None, config=None, pinned=None):
    """The frozen configuration item by item, only files of this repository and of /home/rmcfz/opt/baselines (prereg §2.3): the configuration's own sha256; this arm's executable entries equal
    ARMS; the archive's drive.py and the BL2 protocol; the four method scripts and their copies; the 15 environment records; this arm's baseline files.  Returns the record; Refused (frozen) otherwise."""
    repo = Path(REPO if repo is None else repo)
    baselines = Path(BASELINES if baselines is None else baselines)
    if config is None:
        config, pin = frozen_config(repo, pinned)
    else:
        pin = dict(path='(given)', sha256=None)
    problems, checked = [], []

    def file_is(path, want, what):
        path = Path(path)
        got = sha256(path) if path.is_file() else None
        checked.append(dict(file=str(path), sha256=got, want=want, what=what))
        if got != want:
            problems.append(f'{what} {path}: sha256 {got} != {want}')
    entry = (config.get('arms') or {}).get(arm)
    if entry is None:
        problems.append(f'{arm} is not an arm of the frozen configuration')
    else:
        for key in EXECUTABLE_KEYS:
            if entry.get(key) != ARMS[arm].get(key):
                problems.append(f'{arm}.{key}: frozen {entry.get(key)!r}, code {ARMS[arm].get(key)!r}')
    file_is(repo/ARCHIVE_HARNESS/'drive.py', ARCHIVE_DRIVE_SHA256, 'archive drive.py')
    file_is(repo/BL2_PROTOCOL, BL2_PROTOCOL_SHA256, 'BL2 protocol.md')
    groups = list(ARMS[arm].get('baselines', []))+(['niftyreg'] if ARMS[arm]['kind'] == 'niftyreg' else [])
    for group in groups:
        record = SMOKE_CMD_OF_GROUP.get(group, group)
        file_is(repo/SMOKE_CMD_REL/record/'smoke_cmd.md', SMOKE_CMD_SHA256[record], 'archive smoke_cmd.md')
    for name, want in (config.get('harness', {}).get('copies') or {}).items():
        file_is(repo/ARCHIVE_HARNESS/name, want, 'archive method script')
        file_is(HERE/'harness'/name, want, 'method script copy')
    for rel, want in (config.get('env_records', {}).get('sha256') or {}).items():
        file_is(repo/ENV_RECORDS_REL/rel, want, 'environment record')
    for group in ARMS[arm].get('baselines', []):
        for rel, want in ((config.get('baseline_files') or {}).get(group) or {}).items():
            file_is(baselines/rel, want, f'baseline file ({group})')
        if not (config.get('baseline_files') or {}).get(group):
            problems.append(f'baseline group {group} has no pinned files')
    record = dict(config=pin, arm=arm, files_checked=checked, problems=problems, passed=not problems)
    if problems:
        raise hc.Refused('frozen', f'{problems}')
    return record


# ---- the environments (prereg §5.1 item 9, the baselines) ---------------------------------------------------------------------------------------
def conda_rows(text):
    """conda list: the package lines as (name, version, build, channel?) tuples, the comment lines dropped (the header names the prefix and the conda version's spacing may differ)."""
    return [tuple(line.split()) for line in text.splitlines() if line.strip() and not line.startswith('#')]


def pip_rows(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def check_environment(arm, *, repo=None, config=None, run=None, envs=None, baselines=None):
    """The arm's conda environment against the archived records (only reads: conda list -p, pip freeze, file sha256, git of the clones): conda_list.txt and pip_freeze.txt line for line where the
    record has them, the installed files and binaries the frozen configuration pins, the clones' commit and a clean tree.  Returns the record; Refused (environment) otherwise."""
    repo = Path(REPO if repo is None else repo)
    run = subprocess.run if run is None else run
    envs = Path(ENVS if envs is None else envs)
    baselines = Path(BASELINES if baselines is None else baselines)
    config = frozen_config(repo)[0] if config is None else config
    cfg = ARMS[arm]
    problems, record = [], dict(arm=arm, env=cfg.get('env'))
    env_name = cfg.get('env')
    spec = (config.get('environments') or {}).get(env_name) if env_name else None
    if env_name and spec is None:
        problems.append(f'environment {env_name} is not in the frozen configuration')
    if spec:
        prefix = envs/env_name
        record['prefix'] = str(prefix)
        for kind, argv, parse in (('conda_list', [str(CONDA_BIN), 'list', '-p', str(prefix)], conda_rows), ('pip_freeze', [str(prefix/'bin/python'), '-I', '-m', 'pip', 'freeze'], pip_rows)):
            rel = spec.get(kind)
            if not rel:
                record[kind] = 'no record (the archive has only versions.json for this environment)'
                continue
            want = parse((repo/ENV_RECORDS_REL/rel).read_text(encoding='utf-8'))
            try:
                done = run(argv, capture_output=True, text=True, timeout=300)
                got, rc, err = parse(done.stdout), done.returncode, (done.stderr or '').strip()[:300]
            except (OSError, subprocess.SubprocessError) as exc:
                got, rc, err = [], 127, f'{type(exc).__name__}: {exc}'
            missing, extra = sorted(set(want)-set(got)), sorted(set(got)-set(want))
            record[kind] = dict(record=rel, n_record=len(want), n_now=len(got), returncode=rc, stderr=err, missing=[' '.join(r) if isinstance(r, tuple) else r for r in missing][:40],
                                extra=[' '.join(r) if isinstance(r, tuple) else r for r in extra][:40], same=rc == 0 and got == want)
            if not record[kind]['same']:
                problems.append(f'{env_name} {kind} differs from {rel} (exit {rc}; missing {len(missing)}, extra {len(extra)})')
        files = []
        for rel, want in (spec.get('files') or {}).items():
            path = prefix/rel
            got = sha256(path) if path.is_file() else None
            files.append(dict(file=str(path), sha256=got, want=want))
            if got != want:
                problems.append(f'{path}: sha256 {got} != {want}')
        record['files'] = files
    clones = []
    for name in cfg.get('clones', []):
        clone = (config.get('clones') or {}).get(name)
        if not clone:
            problems.append(f'clone {name} is not in the frozen configuration')
            continue
        directory = baselines/clone['dir']
        try:
            head = run(['git', '--literal-pathspecs', '-C', str(directory), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=60)
            status = run(['git', '--literal-pathspecs', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, timeout=60)
            row = dict(name=name, dir=str(directory), head=head.stdout.strip(), want=clone['commit'], clean=status.returncode == 0 and not status.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            row = dict(name=name, dir=str(directory), head=None, want=clone['commit'], clean=False, error=f'{type(exc).__name__}: {exc}')
        clones.append(row)
        if row['head'] != clone['commit'] or not row['clean']:
            problems.append(f"clone {name}: HEAD {row['head']} (want {clone['commit']}), clean {row['clean']}")
    record.update(clones=clones, problems=problems, passed=not problems,
                  rule='read only: conda list -p and pip freeze against the archived records, installed files and binaries by sha256, the clones by git; nothing installed or changed')
    if problems:
        raise hc.Refused('environment', f'{problems}')
    return record


def archive_reads(arm, *, repo=None, config=None):
    """LP-G revision 1 (decision_log 2026-11-21 ~2:42 AM CST, the LP-G review's important 1): the files under the code tree's archive/ that check_frozen(arm) and check_environment(arm) open,
    built as those two functions build them (repo/<constant>/...): the archive's drive.py and the BL2 protocol; the smoke_cmd.md of each of the arm's baseline groups (and niftyreg's for a
    niftyreg arm); the method scripts of harness.copies; the environment records of env_records.sha256; the conda_list and pip_freeze records of the arm's environment.  Paths only:
    nothing of archive/ is opened here.  The frozen configuration is read when not given (frozen_config, its sha256 checked); one that does not hold gives the constant files only --
    check_frozen then refuses (stage frozen) before it opens any of the others.  An LPBA driver's policy names these files one by one, read only (xb_drive.policies; xb_policy rule 1a);
    tests/test_bayesreg/xb/test_xb_lpg.py runs the real check_frozen and check_environment under that policy's spy."""
    repo = Path(REPO if repo is None else repo)
    paths = [repo/ARCHIVE_HARNESS/'drive.py', repo/BL2_PROTOCOL]
    groups = list(ARMS[arm].get('baselines', []))+(['niftyreg'] if ARMS[arm]['kind'] == 'niftyreg' else [])
    paths += [repo/SMOKE_CMD_REL/SMOKE_CMD_OF_GROUP.get(group, group)/'smoke_cmd.md' for group in groups]
    if config is None:
        try:
            config = frozen_config(repo)[0]
        except hc.Refused:
            config = {}
    paths += [repo/ARCHIVE_HARNESS/name for name in (config.get('harness', {}).get('copies') or {})]
    paths += [repo/ENV_RECORDS_REL/rel for rel in (config.get('env_records', {}).get('sha256') or {})]
    env_name = ARMS[arm].get('env')
    spec = (config.get('environments') or {}).get(env_name) if env_name else None
    paths += [repo/ENV_RECORDS_REL/spec[kind] for kind in ('conda_list', 'pip_freeze') if spec and spec.get(kind)]
    return list(dict.fromkeys(paths))


# ---- (a) the derived npz ------------------------------------------------------------------------------------------------------------------------
IMAGE_MEMBERS = ('fixed', 'moving')


def check_pair_input(path, *, fixed, moving, allowlist):
    """Before the lock, nothing decoded: the manifest through bayesreg.io.pair's whitelisting decoder (its ids must be the pair's), the general guard (guard.check_pair_subjects with the allowlist the
    admission gave: the bound one for a hold-out pair), the container sha256 against manifest.pair_sha256 (a container hash, inside the spy's reading()).  Returns the record; Refused otherwise."""
    from bayesreg.io import pair as pair_module
    from bayesreg.io.guard import SubjectSealed, check_pair_subjects
    path = Path(path)
    manifest_path = path.parent/'manifest.json'
    try:
        grid, metadata, manifest = pair_module._read_manifest(manifest_path)
    except Exception as exc:                        # noqa: BLE001
        raise hc.Refused('input', f'{manifest_path}: {type(exc).__name__}: {exc}') from None
    if (manifest['fixed_subject'], manifest['moving_subject']) != (fixed, moving):
        raise hc.Refused('input', f"{manifest_path} names {manifest['fixed_subject']}, {manifest['moving_subject']}, not the pair {fixed}, {moving}")
    try:
        guard = check_pair_subjects(fixed, moving, allowlist)
    except SubjectSealed as exc:
        raise hc.Refused('admission', f'{exc}') from None
    with xb_spy.original_context():
        digest = sha256(path)
    if digest != manifest['pair_sha256']:
        raise hc.Refused('input', f'{path}: sha256 {digest} is not the manifest\'s pair_sha256 {manifest["pair_sha256"]}')
    return dict(pair_npz=str(path), pair_sha256=digest, manifest=str(manifest_path), manifest_sha256=sha256(manifest_path), grid=metadata, guard=hc._plain(guard))


def read_image_members(path):
    """{fixed, moving}: the raw arrays as stored, one key at a time through the loader's hook (bayesreg.io.pair._open_npz; tests replace it with a guard); nothing else of the archive is asked for,
    the archive is never listed.  Inside the spy's reading()."""
    from bayesreg.io import pair as pair_module
    out = {}
    with xb_spy.original_context():
        with pair_module._open_npz(path) as npz:
            for key in IMAGE_MEMBERS:
                out[key] = np.array(npz[key])
    return out


def derive_inputs(path, out_path, *, shape):
    """The derived npz: fixed and moving of the original, byte for byte (dtype, shape and bytes), np.savez_compressed (lossless); read back and compared.  Returns (arrays, record)."""
    raw = read_image_members(path)
    for key in IMAGE_MEMBERS:
        if tuple(raw[key].shape) != tuple(shape):
            raise hc.Refused('input', f'{path}:{key} has shape {raw[key].shape}, the manifest says {tuple(shape)}')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(out_path)
    np.savez_compressed(out_path, **{k: raw[k] for k in IMAGE_MEMBERS})
    with np.load(out_path, allow_pickle=False) as npz:
        if sorted(npz.files) != sorted(IMAGE_MEMBERS):
            raise WrapperError(f'{out_path} holds {npz.files}')
        back = {k: np.array(npz[k]) for k in IMAGE_MEMBERS}
    for key in IMAGE_MEMBERS:
        if back[key].dtype != raw[key].dtype or back[key].shape != raw[key].shape or back[key].tobytes() != raw[key].tobytes():
            raise WrapperError(f'{out_path}:{key} is not the original member byte for byte')
    record = dict(file=str(out_path), sha256=sha256(out_path), keys=list(IMAGE_MEMBERS), source=str(path),
                  members={k: dict(sha256=array_sha256(raw[k]), dtype=str(raw[k].dtype), shape=list(raw[k].shape)) for k in IMAGE_MEMBERS},
                  rule='prereg §2.3 (a): only fixed and moving, byte for byte, read through bayesreg.io.pair._open_npz one key at a time; the method processes get this file, never the original')
    return back, record


# ---- ConvexAdam-v4seg: the v4 argmax maps (C22.1-4) ---------------------------------------------------------------------------------------------
def check_prediction(pred_root, subject, *, shape, anchor=V4SEG_ANCHOR):
    """Before the lock: the subject's structure-layer prediction is the anchor's (unet35_v4_ens, all_folds_mean, the registry b7682421…, test_labels_seen false, layer structure, this subject), and its
    probs36.npz is the file its provenance names (sha256).  Reads provenance.json and hashes probs36.npz; decodes nothing.  Returns the record; Refused (provenance) otherwise."""
    directory = Path(pred_root)/subject
    path = directory/'provenance.json'
    try:
        provenance = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise hc.Refused('provenance', f'{path}: {exc}') from None
    problems = []
    for key, want in (('subject', subject), ('layer', 'structure'), ('segmenter_id', anchor['segmenter_id']), ('policy', anchor['policy']), ('registry_sha256', anchor['registry_sha256'])):
        if provenance.get(key) != want:
            problems.append(f'{key} {provenance.get(key)!r} != {want!r}')
    if provenance.get('test_labels_seen') is not False:
        problems.append(f"test_labels_seen {provenance.get('test_labels_seen')!r} is not false")
    want = (provenance.get('output_sha256') or {}).get('probs36.npz')
    probs = directory/'probs36.npz'
    got = sha256(probs) if probs.is_file() else None
    if got is None or got != want:
        problems.append(f'probs36.npz sha256 {got} != the provenance\'s {want}')
    shape_recorded = ((provenance.get('outputs') or {}).get('probs36') or {}).get('shape')
    if shape_recorded is not None and list(shape_recorded) != [N_CLASSES, *shape]:
        problems.append(f'probs36 shape {shape_recorded} != {[N_CLASSES, *shape]}')
    if problems:
        raise hc.Refused('provenance', f'{directory}: {problems}')
    return dict(subject=subject, directory=str(directory), provenance_sha256=sha256(path), probs36_sha256=got, hi_block=bool(provenance.get('hi')), anchor=dict(anchor))


def hi_record_path(pred_root, subject):
    """HI's record of a written subject: <root>/subjects/<subject>.json beside <root>/pred (hi_core.infer_holdout)."""
    return Path(pred_root).parent/'subjects'/f'{subject}.json'


def check_hi_record(pred_root, subject, probs36_sha256, provenance_sha256=None):
    """A5a review I2 point 3 (HI's union of the first run and its resume, in the one root): a hold-out subject's prediction counts only when HI wrote its record subjects/<s>.json, naming this subject and
    the probs36.npz that check_prediction hashed (output_sha256), and provenance.json when the record has its sha256.  A prediction in pred/ without a record is a killed run's residue (the resume moves
    it out, hi_core.keep_residue), never read.  Reads the record; Refused (provenance) otherwise."""
    path = hi_record_path(pred_root, subject)
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise hc.Refused('provenance', f'{path}: {exc}: HI has not written this subject (its record subjects/<s>.json is written after the prediction; A5a review I2)') from None
    problems = []
    if not isinstance(row, dict) or row.get('subject') != subject:
        problems.append(f"subject {row.get('subject') if isinstance(row, dict) else row!r} != {subject!r}")
    else:
        want = (row.get('output_sha256') or {}).get('probs36.npz')
        if want != probs36_sha256:
            problems.append(f'probs36.npz sha256 {probs36_sha256} != the record\'s {want}')
        if provenance_sha256 is not None and row.get('provenance_sha256') is not None and row['provenance_sha256'] != provenance_sha256:
            problems.append(f"provenance.json sha256 {provenance_sha256} != the record's {row['provenance_sha256']}")
    if problems:
        raise hc.Refused('provenance', f'{path}: {problems}')
    return dict(record=str(path), record_sha256=sha256(path), run=row.get('run'))


def argmax_labels(p, *, slab=8):
    """C22.1-4: L = argmax_k p^k / sum_j p^j over the 36 classes (0 background), ties to the smallest k; the read-back normalization first, after the raw sum check |sum - 1| <= 1e-3 on every voxel
    (C03.2-4a, float16 storage).  The reference rule of mc0p_core.predicted_labels on the read-back P; slabs along z keep the float64 stack small.  int16."""
    if p.dtype != np.float16 or p.ndim != 4 or p.shape[0] != N_CLASSES:
        raise hc.InfrastructureFailure('input_rejected', f'probs36 is {p.dtype} {p.shape}, float16 (36, nz, ny, nx) required')
    labels = np.empty(p.shape[1:], dtype=np.int16)
    worst = 0.
    for z0 in range(0, p.shape[1], slab):
        P = p[:, z0:z0+slab].astype(np.float64)
        total = P.sum(axis=0)
        if not np.all(np.isfinite(P)) or np.any(total <= 0):
            raise hc.InfrastructureFailure('input_rejected', 'probs36: a non-finite value or a voxel with zero total probability')
        worst = max(worst, float(np.max(np.abs(total-1))))
        labels[z0:z0+slab] = np.argmax(P/total, axis=0)
    if not worst <= RAW_SUM_TOL_FLOAT16:
        raise hc.InfrastructureFailure('input_rejected', f'probs36: raw channel sum |sum - 1| = {worst:.3e} > {RAW_SUM_TOL_FLOAT16} (C03.2-4a)')
    return labels, worst


def labels_digest(labels):
    """experiments/ef/run_ef.py labels_digest: sha256 of the int16 map's bytes per side (a T88 run.json's labels_sha256)."""
    return {side: hashlib.sha256(np.ascontiguousarray(labels[side]).tobytes()).hexdigest() for side in ('fixed', 'moving')}


def write_maps(pred_root, fixed, moving, out_path, *, shape, checked, method_run=None):
    """After the lock: probs36 of both subjects (sha256 again), the argmax maps, maps.npz (fixed_map, moving_map, int16); with a T88 run.json of the method on this pair, its labels_sha256 must be the
    maps' (prereg §2.4).  Returns the record."""
    labels, rows = {}, {}
    for side, subject in (('fixed', fixed), ('moving', moving)):
        path = Path(pred_root)/subject/'probs36.npz'
        if sha256(path) != checked[side]['probs36_sha256']:
            raise hc.InfrastructureFailure('input_rejected', f'{path} changed since the check')
        with np.load(path, allow_pickle=False) as npz:
            p = npz['p']
        if tuple(p.shape[1:]) != tuple(shape):
            raise hc.InfrastructureFailure('input_rejected', f'{path}: shape {p.shape} for the grid {tuple(shape)}')
        labels[side], deviation = argmax_labels(p)
        rows[side] = dict(checked[side], raw_max_abs_sum_deviation=deviation, labels_present=int(np.unique(labels[side]).size))
        del p
    digest = labels_digest(labels)
    method = None
    if method_run is not None:
        run_json = json.loads(Path(method_run).read_text(encoding='utf-8'))
        method = dict(file=str(method_run), sha256=sha256(method_run), labels_sha256=run_json.get('labels_sha256'), equal=run_json.get('labels_sha256') == digest)
        if not method['equal']:
            raise hc.InfrastructureFailure('input_rejected', f'the maps are not the method\'s labels ({method_run}: {run_json.get("labels_sha256")} vs {digest})')
    out_path = Path(out_path)
    np.savez_compressed(out_path, fixed_map=labels['fixed'], moving_map=labels['moving'])
    return dict(file=str(out_path), sha256=sha256(out_path), labels_sha256=digest, sides=rows, method_labels_check=method,
                rule='C22.1-4: argmax of the read-back probs36 (36 classes, ties to the smallest), the same file and rule as the method\'s channels; int16, our layout (z, y, x)')


# ---- the export check (drive.py check, word for word) ------------------------------------------------------------------------------------------
def check(fixed, moving, d, own, mode='constant'):
    from scipy.ndimage import map_coordinates
    ident = np.meshgrid(*[np.arange(n) for n in moving.shape], indexing='ij')
    ours = map_coordinates(moving.astype(np.float64), ident+d, order=1, mode=mode, cval=0.)
    m = fixed > 0
    a, b = ours[m], own[m]
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), max_abs=float(np.abs(a-b).max()), mean_abs=float(np.abs(a-b).mean()),
                n_voxels=int(m.sum()), mode=mode, cval=0.0, order=1, region='fixed > 0')


def export_check(fixed, moving, d, own, alts):
    rec = dict(check=check(fixed, moving, d, own), check_nearest_mode=check(fixed, moving, d, own, mode='nearest'), check_alt={k: check(fixed, moving, d, v) for k, v in alts.items()},
               threshold=EXPORT_CORR_MIN)
    corr = rec['check']['corr']
    rec['suspicious'] = not (corr >= EXPORT_CORR_MIN)            # NaN counts as suspicious
    rec['rule'] = 'drive.py check: our map_coordinates(moving, identity + d, order 1, constant 0) against the method\'s own warped moving inside fixed > 0; corr < .999 is suspicious (not a failure, prereg §4.5)'
    return rec


# ---- the machine during a step (drive.py, with the cgroup and the fd sampler) ------------------------------------------------------------------
def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    return float('nan')


def gpu_state():
    try:
        out = subprocess.run([NVSMI, '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip().split(', ')
        return dict(name=out[0], mem_used_mib=float(out[1]), mem_total_mib=float(out[2]), util_pct=float(out[3]))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e))


def top_processes(n=6):
    out = subprocess.run(['ps', '-eo', 'pid,pcpu,rss,etime,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout
    return out.strip().splitlines()[:n+1]


def precheck(gpu, *, log=print, mem=None, gpu_reader=None, sleep=time.sleep, wait_max=WAIT_MAX_S):
    """drive.py precheck: wait (up to 15 min) until MemAvailable >= 6 GB and, for GPU arms, the GPU looks idle.  Returns a record; its failure is an infrastructure failure (prereg §5.4)."""
    mem = mem_available_gb if mem is None else mem
    gpu_reader = gpu_state if gpu_reader is None else gpu_reader
    rec = dict(checks=[], waited_s=0.0)
    t0 = time.time()
    while True:
        m = mem(); g = gpu_reader() if gpu else None
        ok_mem = m >= MEM_MIN_GB
        ok_gpu = (not gpu) or ('error' not in g and g['mem_used_mib'] < GPU_IDLE_MEM_MIB and g['util_pct'] < GPU_IDLE_UTIL)
        rec['checks'].append(dict(utc=utc(), mem_available_gb=round(m, 2), gpu=g, ok_mem=ok_mem, ok_gpu=ok_gpu))
        if ok_mem and ok_gpu:
            rec.update(passed=True)
            break
        if time.time()-t0 > wait_max:
            rec.update(passed=False)
            break
        log(f'precheck waiting: mem {m:.2f} GB, gpu {g}')
        sleep(30)
    rec['waited_s'] = round(time.time()-t0, 1)
    if not rec['passed']:
        raise hc.InfrastructureFailure('precheck', f"MemAvailable or the GPU not ready after {rec['waited_s']} s: {rec['checks'][-1]}")
    return rec


def base_env(extra=None):
    """drive.py base_env, word for word."""
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS and not k.startswith('CONDA')}
    env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    env.update(extra or {})
    return env


def cgroup_events():
    """memory.events of this process's cgroup (the unit's when the driver runs in one), or None."""
    import unit_probe
    path, _ = unit_probe.cgroup_path()
    if path is None:
        return None
    files, _ = unit_probe.read_cgroup(path)
    return unit_probe.parse_flat(files.get('memory.events')) if 'memory.events' in files else None


class Monitor(threading.Thread):
    """drive.py Monitor: polls MemAvailable (and the GPU) every second during a method step and enforces the memory guard; XB adds the fd sampler (xb_spy.FdSampler) on the same tick."""

    def __init__(self, proc, gpu, sampler=None, log=print):
        super().__init__(daemon=True)
        self.proc, self.gpu, self.sampler, self.log = proc, gpu, sampler, log
        self.min_mem = float('inf'); self.max_gpu_mem = 0.0; self.max_gpu_util = 0.0; self.killed = False; self.max_threads = 0
        self.breach = None
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set() and self.proc.poll() is None:
            m = mem_available_gb(); self.min_mem = min(self.min_mem, m)
            try:
                for line in open(f'/proc/{self.proc.pid}/status'):
                    if line.startswith('Threads:'):
                        self.max_threads = max(self.max_threads, int(line.split()[1]))
            except OSError:
                pass
            if m < MEM_GUARD_GB and not self.killed:
                self.killed = True
                self.log(f'MEMORY GUARD: MemAvailable {m:.2f} GB < {MEM_GUARD_GB} GB, killing process group {self.proc.pid}')
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if self.sampler is not None and self.breach is None:
                new = self.sampler.sample()
                if new:
                    self.breach = new
                    self.log(f'SPY: the method session opened {new}; killing process group {self.proc.pid}')
                    self.sampler.kill()
            if self.gpu:
                g = gpu_state()
                if 'error' not in g:
                    self.max_gpu_mem = max(self.max_gpu_mem, g['mem_used_mib']); self.max_gpu_util = max(self.max_gpu_util, g['util_pct'])
            self.stop.wait(1.0)


def uptime():
    try:
        return float(Path('/proc/uptime').read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def dmesg():
    """The kernel log, or None when it cannot be read (then the kernel's side of the evidence is missing, never read as no OOM)."""
    try:
        done = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def run_step(name, cmd, env, cwd, logfile, gpu, *, policy=None, log=print, popen=subprocess.Popen, events=cgroup_events):
    """drive.py run_step: one command, stdout+stderr appended to logfile; XB: the fd sampler under `policy`, the cgroup's memory.events before and after and the kernel's oom-kill lines on every exit (I2).
    Returns a step record."""
    with open(logfile, 'a') as f:
        f.write(f'\n===== step {name} {utc()}\n$ (cwd={cwd}) {" ".join(map(str, cmd))}\n'); f.flush()
        env = dict(env); env['BL2_T_LAUNCH'] = repr(time.time())
        import resource
        ev0 = events()
        up0 = uptime()
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        p = popen([str(c) for c in cmd], env=env, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        sampler = xb_spy.FdSampler(policy, p.pid) if policy is not None else None
        mon = Monitor(p, gpu, sampler, log); mon.start()
        rc = p.wait()
        secs = time.perf_counter()-t0
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        mon.stop.set(); mon.join(timeout=5)
        if sampler is not None and sampler.samples == 0:
            sampler.sample()                             # a step shorter than a tick: one sample after it (sees nothing open, but the record says it ran)
        ev1 = events()
        kernel = hc.parse_oom(dmesg(), up0 or 0.)       # I2: every step, whatever its exit (None when dmesg cannot be read)
        f.write(f'===== step {name} exit {rc} after {secs:.3f} s\n')
    rec = dict(name=name, cmd=[str(c) for c in cmd], cwd=str(cwd), returncode=rc, seconds_wall=secs,
               min_mem_available_gb=round(mon.min_mem, 2) if mon.min_mem != float('inf') else None,
               memory_guard_killed=mon.killed, cpu_user_s=round(ru1.ru_utime-ru0.ru_utime, 2), cpu_sys_s=round(ru1.ru_stime-ru0.ru_stime, 2),
               max_threads_sampled=mon.max_threads, max_rss_children_kb=ru1.ru_maxrss, memory_events_before=ev0, memory_events_after=ev1, uptime_start=up0, kernel_oom_lines=kernel,
               fd_spy=sampler.record() if sampler is not None else None, spy_breach=mon.breach)
    if gpu:
        rec.update(max_gpu_mem_used_mib=mon.max_gpu_mem, max_gpu_util_pct=mon.max_gpu_util)
    log(f'step {name}: exit {rc}, {secs:.2f} s, min MemAvailable {rec["min_mem_available_gb"]} GB, cpu {rec["cpu_user_s"]}+{rec["cpu_sys_s"]} s, max threads {mon.max_threads}'
        + (f', max GPU mem {mon.max_gpu_mem:.0f} MiB' if gpu else ''))
    return rec


def step_outcome(step, unit=None):
    """None, or the exception the step's end calls for (prereg §5.4, §4.5; revision 1c item 1 through holdout_common.classify_unit_end): a spy breach is infrastructure; on **every** exit (I2: an
    antsRegistrationSyN.sh whose antsRegistration was OOM-killed exits 0) the unit's memory.events during the step and the kernel's oom-kill lines since its start are read for OOM evidence, and so
    are the driver's memory guard and an exit -9 (rule (乙): infra_failure; rule (甲): the unit's own limit unit_limit, else infra_failure; a -9 whose evidence cannot be read: infra_failure, the
    -9 itself is evidence); a spyrun breach exit is infrastructure; another non-zero exit is the caller's to class."""
    unit = os.environ.get('XB_UNIT') if unit is None else unit
    if step.get('spy_breach'):
        return hc.InfrastructureFailure('spy_breach', f"step {step['name']}: the fd sampler saw {step['spy_breach']}")
    rc = step.get('returncode')
    events = hc.events_delta(step.get('memory_events_before'), step.get('memory_events_after'))
    lines = step.get('kernel_oom_lines')
    killed = rc == -9
    evidence = hc.oom_evidence(unit=unit, oom_lines=lines, events=events, step_kill=killed)
    machine = [r for r in lines or [] if r.get('constraint') == 'CONSTRAINT_NONE' or r.get('global_oom')]
    if step.get('memory_guard_killed') or evidence or machine:
        status, kind, record = hc.classify_unit_end(unit=unit, systemd_result=None, oom_lines=lines, events=events, step_kill=killed,
                                                    memory_guard=bool(step.get('memory_guard_killed')), need=('dmesg', 'end_probe'))
        message = f"step {step['name']}: exit {rc}: {status} {kind} (rule {record['rule']}; memory.events during the step {events}; OOM evidence {record['oom_evidence']})"
        if status == 'unit_limit':
            return UnitLimit(message)
        if status == 'infra_failure':
            return hc.InfrastructureFailure(kind, message)
        return NeedsRejudgment(message)
    if rc == SPYRUN_BREACH_EXIT:
        return hc.InfrastructureFailure('spy_breach', f"step {step['name']}: the method process's spy stopped it (exit {rc})")
    return None


def checked_step(step, *, failure=MethodFailure, what='the method'):
    """Raise what a step's end calls for: step_outcome, then a non-zero exit as `failure`."""
    outcome = step_outcome(step)
    if outcome is not None:
        raise outcome
    if step['returncode'] != 0:
        raise failure(f"{what}: step {step['name']} exit {step['returncode']}")


# ---- the arms (drive.py arm_*, the inputs from the derived npz) --------------------------------------------------------------------------------
def write_sitk_identity(arr, path):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))      # array (z, y, x) -> ITK index (x, y, z)
    img.SetSpacing((1., 1., 1.)); img.SetOrigin((0., 0., 0.)); img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    sitk.WriteImage(img, str(path))


def write_inputs_sitk(work, z):
    t0 = time.perf_counter()
    write_sitk_identity(z['fixed'], work/'fixed.nii.gz'); write_sitk_identity(z['moving'], work/'moving.nii.gz')
    return time.perf_counter()-t0


@dataclass
class Ctx:
    """What an arm function needs besides drive.py's arguments: the policy of the method process, the log, the step runner and the precheck (tests replace them)."""
    policy: dict
    log: object = print
    step: object = run_step
    precheck: object = precheck
    inputs: Path = None
    maps: Path = None


def arm_ants(arm, cfg, work, logfile, z, ctx):
    import SimpleITK as sitk
    out = dict(steps=[], warnings=[])
    out['seconds_input_write'] = write_inputs_sitk(work, z)
    P = f"ants_{cfg['variant']}_"
    env = base_env(dict(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='8', PATH=f"{ANTS_BIN}:/usr/local/bin:/usr/bin:/bin"))
    out['precheck'] = ctx.precheck(False, log=ctx.log)
    cmd = [ANTS_BIN/'antsRegistrationSyN.sh', '-d', '3', '-f', 'fixed.nii.gz', '-m', 'moving.nii.gz', '-o', P, '-t', cfg['variant'], '-n', '8']
    s = ctx.step('register', cmd, env, work, logfile, False, policy=ctx.policy, log=ctx.log); out['steps'].append(s)
    checked_step(s, what='antsRegistrationSyN.sh')
    warps = sorted(glob.glob(str(work/f'{P}[0-9]Warp.nii.gz')))
    if len(warps) != 1:
        raise MethodFailure(f'expected one forward warp, found {warps}')
    tx = ['-t', Path(warps[0]).name]
    if (work/f'{P}0GenericAffine.mat').exists():
        tx += ['-t', f'{P}0GenericAffine.mat']
    out['transforms'] = tx
    s2 = ctx.step('export_disp', [ANTS_BIN/'antsApplyTransforms', '-d', '3', '-r', 'fixed.nii.gz', *tx, '-o', f'DisplacementField[{P}disp.nii.gz]',
                                  '--float', '0', '-v', '1'], env, work, logfile, False, policy=ctx.policy, log=ctx.log); out['steps'].append(s2)
    s3 = ctx.step('warp_aat', [ANTS_BIN/'antsApplyTransforms', '-d', '3', '-i', 'moving.nii.gz', '-r', 'fixed.nii.gz', *tx, '-n', 'Linear',
                               '-o', f'{P}warped_aat.nii.gz', '--float', '0'], env, work, logfile, False, policy=ctx.policy, log=ctx.log); out['steps'].append(s3)
    checked_step(s2, what='antsApplyTransforms (export)')
    checked_step(s3, what='antsApplyTransforms (warp)')
    out['seconds_registration'] = s['seconds_wall']
    out['seconds_method_export'] = s2['seconds_wall']
    out['seconds_total'] = out['seconds_input_write']+s['seconds_wall']+s2['seconds_wall']
    f = sitk.ReadImage(str(work/'fixed.nii.gz'))
    u = sitk.GetArrayFromImage(sitk.ReadImage(str(work/f'{P}disp.nii.gz'))).astype(np.float64)      # (z, y, x, 3), physical (x, y, z)
    M = np.array(f.GetDirection()).reshape(3, 3) @ np.diag(f.GetSpacing())
    d_xyz = u @ np.linalg.inv(M).T
    d = np.moveaxis(d_xyz[..., ::-1], -1, 0)
    own = sitk.GetArrayFromImage(sitk.ReadImage(str(work/f'{P}Warped.nii.gz'))).astype(np.float64)
    alt = sitk.GetArrayFromImage(sitk.ReadImage(str(work/f'{P}warped_aat.nii.gz'))).astype(np.float64)
    out.update(threads=dict(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8, n=8), device='cpu',
               own_warped='antsRegistration --output ...Warped.nii.gz (linear)', alt_warped='antsApplyTransforms -n Linear',
               config=f"antsRegistrationSyN.sh -t {cfg['variant']} defaults, -n 8; ANTS_RANDOM_SEED unset")
    return d, own, dict(antsApplyTransforms_linear=alt), out


def arm_niftyreg(arm, cfg, work, logfile, z, ctx):
    import nibabel as nib
    out = dict(steps=[], warnings=[])
    out['seconds_input_write'] = write_inputs_sitk(work, z)
    env = base_env(dict(OMP_NUM_THREADS='8'))
    out['precheck'] = ctx.precheck(False, log=ctx.log)
    s = ctx.step('register', [NR_BIN/'reg_f3d', '-ref', 'fixed.nii.gz', '-flo', 'moving.nii.gz', '-cpp', 'nr_cpp.nii.gz', '-res', 'nr_warped.nii.gz'],
                 env, work, logfile, False, policy=ctx.policy, log=ctx.log); out['steps'].append(s)
    checked_step(s, what='reg_f3d')
    s2 = ctx.step('export_disp', [NR_BIN/'reg_transform', '-ref', 'fixed.nii.gz', '-disp', 'nr_cpp.nii.gz', 'nr_disp.nii.gz'], env, work, logfile, False, policy=ctx.policy, log=ctx.log)
    out['steps'].append(s2)
    s3 = ctx.step('warp_linear', [NR_BIN/'reg_resample', '-ref', 'fixed.nii.gz', '-flo', 'moving.nii.gz', '-trans', 'nr_cpp.nii.gz',
                                  '-res', 'nr_warped_lin.nii.gz', '-inter', '1', '-pad', '0'], env, work, logfile, False, policy=ctx.policy, log=ctx.log); out['steps'].append(s3)
    checked_step(s2, what='reg_transform')
    checked_step(s3, what='reg_resample')
    out['seconds_registration'] = s['seconds_wall']
    out['seconds_method_export'] = s2['seconds_wall']
    out['seconds_total'] = out['seconds_input_write']+s['seconds_wall']+s2['seconds_wall']
    fimg = nib.load(str(work/'fixed.nii.gz'))
    A = fimg.get_sform() if int(fimg.header['sform_code']) > 0 else fimg.get_qform()
    dimg = nib.load(str(work/'nr_disp.nii.gz'))
    u = np.asarray(dimg.dataobj, dtype=np.float64)
    out['disp_nifti_shape'] = list(u.shape)
    u = u.reshape(u.shape[:3]+(3,))                                            # (X, Y, Z, 1, 3) -> (X, Y, Z, 3), world mm
    d_ijk = u @ np.linalg.inv(A[:3, :3]).T
    d = np.transpose(d_ijk[..., ::-1], (3, 2, 1, 0))
    own = np.asarray(nib.load(str(work/'nr_warped_lin.nii.gz')).dataobj, dtype=np.float64).transpose(2, 1, 0)
    cub = np.asarray(nib.load(str(work/'nr_warped.nii.gz')).dataobj, dtype=np.float64).transpose(2, 1, 0)
    n_nan = int(np.isnan(cub).sum())
    if n_nan:
        out['warnings'].append(f'reg_f3d -res (cubic) has {n_nan} NaN voxels (default -pad nan); set to 0 for the secondary check')
    out.update(threads=dict(OMP_NUM_THREADS=8), device='cpu', fixed_affine_used=A.tolist(),
               own_warped='reg_resample -inter 1 -pad 0 (linear)', alt_warped='reg_f3d -res (cubic, NaN->0)',
               config='reg_f3d defaults (identity start, NMI 64 bins, -sx -5, -be .001, -le .01, -ln 3, -maxit 150), OMP_NUM_THREADS=8')
    return d, own, dict(reg_f3d_res_cubic=np.nan_to_num(cub)), out


def arm_deeds(arm, cfg, work, logfile, z, ctx):
    import nibabel as nib
    import gzip
    import struct
    out = dict(steps=[], warnings=[])
    out['seconds_input_write'] = write_inputs_sitk(work, z)
    hdr = {}
    for k in ('fixed', 'moving'):
        b = gzip.open(str(work/f'{k}.nii.gz')).read()
        hdr[k] = dict(vox_offset=struct.unpack('<f', b[108:112])[0], sizeof_hdr=struct.unpack('<i', b[0:4])[0],
                      extension_flag=list(b[348:352]), datatype=struct.unpack('<h', b[70:72])[0],
                      dim=list(struct.unpack('<8h', b[40:56])[1:4]), file_bytes=len(b))
        if (hdr[k]['vox_offset'] != 352 or hdr[k]['sizeof_hdr'] != 348 or hdr[k]['extension_flag'] != [0, 0, 0, 0]
                or len(b) != 352+int(np.prod(hdr[k]['dim']))*4):
            raise WrapperError(f'deeds input header unsuitable: {hdr}')
    out['input_headers'] = hdr
    env = base_env(dict(OMP_NUM_THREADS='8'))
    out['precheck'] = ctx.precheck(False, log=ctx.log)
    s = ctx.step('register', [DEEDS_BIN/'deedsBCV', '-F', 'fixed.nii.gz', '-M', 'moving.nii.gz', '-O', 'deeds'], env, work, logfile, False, policy=ctx.policy, log=ctx.log)
    out['steps'].append(s)
    checked_step(s, what='deedsBCV')
    if not (work/'deeds_displacements.dat').exists():
        raise MethodFailure('deedsBCV wrote no deeds_displacements.dat')
    out['seconds_registration'] = s['seconds_wall']; out['seconds_method_export'] = 0.0
    out['seconds_total'] = out['seconds_input_write']+s['seconds_wall']
    m, n, o = nib.load(str(work/'fixed.nii.gz')).shape
    flow = np.fromfile(str(work/'deeds_displacements.dat'), dtype='<f4')
    sz1 = flow.size//3
    g = int(round((m*n*o/sz1)**(1/3)))
    m1, n1, o1 = m//g, n//g, o//g
    if m1*n1*o1 != sz1:
        raise MethodFailure(f'deeds_displacements.dat: {sz1} values per component do not make a grid of {m}x{n}x{o} / {g}')
    U, V, W = (flow[c*sz1:(c+1)*sz1].reshape(o1, n1, m1).transpose(2, 1, 0) for c in range(3))

    def lin1d(F, axis, N, G):
        c = np.arange(N, dtype=np.float32)/(np.float32(N)/np.float32(G))
        lo = np.floor(c).astype(np.int64); w = (c-lo).astype(np.float64)
        i0, i1 = np.clip(lo, 0, G-1), np.clip(lo+1, 0, G-1)
        shp = [1, 1, 1]; shp[axis] = N; w = w.reshape(shp)
        return np.take(F, i0, axis=axis)*(1-w)+np.take(F, i1, axis=axis)*w

    def up(F):
        for ax, (N, G) in enumerate(((m, m1), (n, n1), (o, o1))):
            F = lin1d(F.astype(np.float64), ax, N, G)
        return F

    d_ijk = np.stack([up(V), up(U), up(W)], axis=-1)
    d = np.transpose(d_ijk[..., ::-1], (3, 2, 1, 0))
    own = np.asarray(nib.load(str(work/'deeds_deformed.nii.gz')).dataobj, dtype=np.float64).transpose(2, 1, 0)
    out.update(threads=dict(OMP_NUM_THREADS=8), device='cpu', control_grid=[m1, n1, o1], grid_step=g,
               own_warped='deeds_deformed.nii.gz (trilinear, clamp-to-edge)',
               config='deedsBCV program defaults (-a 1.6 -l 5 -G 8x7x6x5x4 -L 8x7x6x5x4 -Q 5x4x3x2x1), no affine, OMP_NUM_THREADS=8')
    return d, own, {}, out


def method_command(arm, cfg, work, ctx):
    """The method process's argv: drive.py's `[py, -I, -B, script, *args, target, big, *post_args]`, under spyrun, with the derived npz as the target (and maps.npz before the output directory for
    ConvexAdam-v4seg)."""
    py = ENVS/cfg['env']/'bin/python'
    script = HERE/cfg['script']
    targets = [ctx.inputs, ctx.maps] if cfg['kind'] == 'v4seg' else [ctx.inputs]
    return xb_spy.spyrun_argv(py, work/'spy_policy.json', work/'spy_method.jsonl', script, [*cfg['args'], *targets, work, *cfg.get('post_args', [])])


def arm_py(arm, cfg, work, logfile, z, ctx):
    """drive.py arm_py: the method script in its environment; XB: under spyrun, the derived npz instead of the original (prereg §2.3 (a)); ConvexAdam-v4seg's wrapper the same way."""
    out = dict(steps=[], warnings=[])
    env = base_env(cfg.get('extra_env'))
    out['precheck'] = ctx.precheck(cfg['device'] == 'cuda', log=ctx.log)
    (work/'spy_policy.json').write_text(json.dumps(ctx.policy, indent=1)+'\n')
    cmd = method_command(arm, cfg, work, ctx)
    s = ctx.step('method', cmd, env, work, logfile, cfg['device'] == 'cuda', policy=ctx.policy, log=ctx.log)
    out['steps'].append(s)
    s['method_spy'] = xb_spy.method_spy_record(work/'spy_method.jsonl', expect=[p for p in (ctx.inputs, ctx.maps if cfg['kind'] == 'v4seg' else None) if p is not None])
    outcome = step_outcome(s)
    if outcome is not None:
        raise outcome
    if s['method_spy']['breaches']:
        raise hc.InfrastructureFailure('spy_breach', f"the method process's spy: {s['method_spy']['breaches'][:3]}")
    if cfg['kind'] == 'v4seg' and s['returncode'] == V4SEG_WRAPPER_ERROR_EXIT:
        raise WrapperError(f'the ConvexAdam-v4seg wrapper\'s own code raised (exit {s["returncode"]}; {work/"wrapper_error.json"})')
    if s['returncode'] != 0 or not (work/'method_out.json').exists():
        raise MethodFailure(f'method process exit {s["returncode"]}' + ('' if (work/'method_out.json').exists() else ' (no method_out.json)'))
    if not s['method_spy']['positive_condition']:
        raise hc.InfrastructureFailure('spy_breach', f"the method process's spy saw nothing it should have (positive condition false): {s['method_spy']}")
    mo = json.loads((work/'method_out.json').read_text())
    out['method_out'] = mo
    out['seconds_registration'] = mo['seconds_registration_cold']
    out['seconds_registration_warm'] = mo.get('seconds_registration_warm')
    out['seconds_total'] = mo['seconds_total']
    out['seconds_process_wall'] = s['seconds_wall']
    out['device'] = mo.get('device', cfg['device']); out['threads'] = mo.get('threads')
    out['config'] = mo.get('config'); out['warnings'] += mo.get('warnings', [])
    if not (work/'disp.npz').exists() or not (work/'own_warped.npz').exists():
        raise MethodFailure('the method wrote no disp.npz or own_warped.npz')
    with np.load(work/'disp.npz') as q:
        d = np.array(q['d'], dtype=np.float64)
    with np.load(work/'own_warped.npz') as q:
        own = np.array(q['w'], dtype=np.float64)
    alts = {}
    if (work/'own_warped_alt.npz').exists():
        with np.load(work/'own_warped_alt.npz') as q:
            alts = {k: np.array(q[k], dtype=np.float64) for k in q.files}
    return d, own, alts, out


ARM_FUNCTIONS = dict(py=arm_py, v4seg=arm_py, ants=arm_ants, niftyreg=arm_niftyreg, deeds=arm_deeds)


# ---- after the method ---------------------------------------------------------------------------------------------------------------------------
def finish_field(d, shape, work):
    """drive.py main after the arm: the shape and finiteness (an incomplete export is the method's failure), disp.npz as float32 (written here unless the method wrote it), read back."""
    if d.shape != (3,)+tuple(shape):
        raise MethodFailure(f'displacement shape {d.shape}, expected {(3,)+tuple(shape)}')
    if not np.isfinite(d).all():
        raise MethodFailure('non-finite displacement')
    if not (work/'disp.npz').exists():
        np.savez_compressed(work/'disp.npz', d=d.astype(np.float32))
    with np.load(work/'disp.npz') as q:
        back = np.array(q['d'], dtype=np.float64)                    # check exactly the saved float32 field
    if back.shape != (3,)+tuple(shape) or not np.isfinite(back).all():
        raise MethodFailure('disp.npz as saved is misshapen or non-finite')
    return back


MANIFEST_HEADER = 'path\tsize_bytes\tsha256\n'                     # runs/20261107-cleanup/delete_verified.sh's manifest format


def intermediates_manifest(work, manifest_path, *, keep=KEEP_FILES):
    """Prereg §5.3: the method's own large intermediate files of a run whose export was written and checked (the inputs it wrote, its native transforms, its warped NIfTI) and the derived npz are
    deleted only after the run is audited and their sha256 recorded -- by a separate, reviewed step (the lead's ruling, 2026-11-07: experiments/xb/delete_verified.sh on a manifest the lead approves).
    The driver deletes nothing: it lists them here, path, size and sha256, written once (the cleanup script's format).  Only for status ok; failed runs are kept whole and listed nowhere."""
    rows = []
    for path in sorted(p for p in Path(work).rglob('*') if p.is_file() and not (p.parent == Path(work) and p.name in keep)):
        rows.append((str(path.resolve()), path.stat().st_size, sha256(path)))
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', encoding='utf-8') as stream:
        stream.write(MANIFEST_HEADER)
        for path, size, digest in rows:
            stream.write(f'{path}\t{size}\t{digest}\n')
    return [dict(path=p, bytes=b, sha256=d) for p, b, d in rows]


def suspicious_pairs(arm, pair_set):
    """The pairs of an arm whose terminal record says the export check is suspicious, in the pair list's order (P19)."""
    record_root = roots(pair_set)[0]
    out = []
    for path in sorted(hc.arm_dir(arm, record_root).glob('*/result.json'))+sorted(hc.arm_dir(arm, record_root).glob('*/resume/result.json')):
        doc = read_json(path)
        if doc is None or read_json(path.with_name('overridden.json')) is not None:
            continue
        if doc.get('status') == 'ok' and (doc.get('export_check') or {}).get('suspicious'):
            out.append(doc.get('pair'))
    return out


def complete(document):
    """Prereg §5.4 "完整记录" for an external arm: result.json and the export check written; a method failure and a unit limit are final too (not rerun).  Infrastructure failures, wrapper errors
    (under the stop policy) and needs_lead_rejudgment are not complete: they are the cells a resume authorization may rerun (an overriding record counts first: attempt_complete, I2)."""
    status = document.get('status')
    if status == 'ok':
        return bool(document.get('export_check'))
    if status in ('method_failure', 'unit_limit'):
        return True
    if status == 'wrapper_error':
        return document.get('wrapper_error_policy') == 'identity'
    return False                                  # infra_failure, wrapper_error under stop, needs_lead_rejudgment


def classify(exc):
    """(status, kind) of an exception raised after the lock (prereg §4.5, §5.4)."""
    if isinstance(exc, MethodFailure):
        return 'method_failure', 'method'
    if isinstance(exc, UnitLimit):
        return 'unit_limit', 'MemoryMax'
    if isinstance(exc, NeedsRejudgment):
        return 'needs_lead_rejudgment', 'evidence_unclear'
    if isinstance(exc, hc.InfrastructureFailure):
        return 'infra_failure', exc.kind
    if isinstance(exc, xb_spy.Breach):
        return 'infra_failure', 'spy_breach'
    if isinstance(exc, hc.Refused):
        return 'infra_failure', f'input_refused_after_lock:{exc.stage}'
    return 'wrapper_error', type(exc).__name__
