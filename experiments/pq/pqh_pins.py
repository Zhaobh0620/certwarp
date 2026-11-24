"""PQH (runs/20261109-pqh/任务书.md; contract C22.H, revision 7, adopted by BZ 2026-11-09 with every default): the pinned values of PQ on the B-line hold-out.  Nothing here reads a file.

The arithmetic is the development package's (experiments/pq: pq_core, pq_members, pq_stats; their pins in pq_pins.py, unchanged); this file holds what the hold-out adds:
  the frozen settings of C22.H3-6 (posterior_qc = pq1, the sampling, D89 = (a), the baselines, the seg4 table, the segmenter anchor, the four members, plan.md, the preregistration, the contract revision,
  the two units' specifications of C22.H5-2 and C22.H6-3), and their settings_sha256 (C22.7-3's way: sha256 of the canonical JSON, recorded in every pq.json and in the unit records);
  the fixed paths (the members unit's one-shot lock of C22.H5-4, the PQ unit's lock, the dev-mode locks of the smoke under he_smoke_v3, D124 (a)); the storage of C22.H5-7 and C22.H6-1;
  the replay gate of D119 (C22.H8): its plan, its lock and the decision constant PQ_REPLAY_DECISION (None until a reviewed commit fills it; the evaluation entry refuses at its step (0) while it is None).
Values that waited for another package or for BZ (None or marked 〔对齐〕, to be aligned): PREREG_SHA256 (pinned at P0, 11-14, BZ having adopted revision 2), EV_EVAL_DIR (EV's C23.12 entry),
RUNTIME_CACHE_DIRS (as it is at P0 it is the pin: the smoke's segment (i) checks only that every cache directory it used is in it -- a breach otherwise -- and describes the directories used; it
is not changed after the smoke, and tightening it is a P0′: the lead's reading of C22.H4-3 revision 7c, 11-14, SM3 revision 3).  The method's records follow the lead's interface alignment (runs/20261109-align/接口对齐.md item 1, main 73723d7)
and HB's implementation (hb-tools a9529c7: experiments/hb/hb_core.py, hb_review.py finalize, hb_store.py move).
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pq_pins as pins

CONTRACT = ('docs/bayesreg/01_合同.md C22.H (revision 7, adopted by BZ 2026-11-09, every default: D113 (a), D114 (a), D117 (a), D119 (a), D120 (a), D121 (a), D123 (b), D124 (a), D125 (a)); '
            'C22.Q (revision 5); C23 (C23.6-5 with the 6f correction, C23.9-12)')
TASK = 'runs/20261109-pqh/任务书.md'
SCHEMA = 'bayesreg.pqh.{}.v1'
TOOL = 'pqh'                                            # holdout_common's authorization names: <prefix>pqh_<kind>_<k>.json
PQ_ARM = 'pq'                                           # the storage roots' <arm> (C22.H5-7: "PQ 当作一个 <arm> 名"); not an arm of the tests (D82 = -a)
UNIT_KEYS = ('members', 'out')                          # holdout_common's cell "pair" of a PQ unit (an authorization names the unit, there is no pair)
MODES = ('holdout', 'dev')                              # two subcommand families, never an option that changes a path (C23.0-3, C22.H8 (ii))

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
MAIN_REPO = pins.MAIN_REPO
DATA_IMAGES = pins.DATA_IMAGES                          # <subject>/aligned_norm.nii.gz, only through the bound OasisFiles (C23.4-1)
MODELS_ROOT = RESULTS/pins.MODELS_ROOT_REL              # the four member files, by path and sha256 (pins.MEMBERS)
MEMBERS = pins.MEMBERS
ANCHOR = pins.ANCHOR
PLAN_REL, PLAN_SHA256 = pins.PLAN_REL, pins.PLAN_SHA256
SEG4_TABLE_REL, SEG4_TABLE_SHA256 = pins.SEG4_TABLE_REL, pins.SEG4_TABLE_SHA256
TAPER_L = pins.TAPER_L
PREREG_REL = 'docs/bayesreg/notes/留出集预登记_20261107.md'
PREREG_SHA256 = 'df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493'     # C22.H3-6, prereg revision 1a item 1, 1b item 9; P0 (11-14), after BZ adopted revision 2; None or another file refuses a hold-out run (dev mode records it)

# ---- the method's records (HB, C22.H1-3; the alignment of 2026-11-09, item 1) --------------------------------------------------------------------
METHOD_ARM = 'method'                                   # the method arm (HB): its cell holdout_B_v1/work/method/<pair>/ until step 5, then /mnt/d/seq_vp_tag_holdout_B_v1/method/<pair>/ (moved whole)
FINALIZED_DIR, FINALIZED_NAME = 'review', 'finalized_outputs.json'      # HB's step-5 queue: holdout_B_v1/review/finalized_outputs.json (hb_review.finalize)
MOVE_RECORD_NAME = 'move.json'                          # holdout_common.move_files' record of a cell moved to D: (hb_store.move_pair): holdout_B_v1/method/<pair>/move.json
MOVE_RESUME_NAME = 'move_resume.json'                   # HB revision 2 (I3): a half move (move.json passed false) moved on once by hb_store.move_resume, recorded beside it (review NI1)
TERMINAL_NAME, REVIEW_RECORD_REL = 'terminal.json', 'review/review.json'    # in the cell: the terminal record (key line_status) and the step-5 review record (C22.H1-5's keys at its top)
LINE_STATUSES_WITH_OUTPUT = ('hx_certified', 'hx_fallback_configC', 'hx_fallback_identity')
LINE_STATUSES_WITHOUT_OUTPUT = ('hx_provenance_rejected', 'hx_implementation_error', 'infra_failure')     # pq_absent.json (C22.H6-3); infra_failure: a committed no_resume (未批补跑) record
HX_CERTIFIED = 'hx_certified'
FALLBACK_KINDS = MappingProxyType(dict(hx_fallback_configC='configC', hx_fallback_identity='identity'))
HE_FILES = MappingProxyType(dict(final='output/final.npz', disp='output/disp.npz', cert='output/cert.json', line='output/line.json', run_json='t88/run.json', field='t88/field_0600.npz',
                                 repair_json='repair/repair.json', repair_npz='repair/repair.npz', terminal=TERMINAL_NAME, review=REVIEW_RECORD_REL))
OUTPUT_KEYS = ('final', 'disp', 'cert', 'line')
CERTIFIED_KEYS = ('run_json', 'field', 'repair_json', 'repair_npz')     # hx_certified: the inputs of the reconstruction of v (D113 (a))

# ---- the evaluation entry (EV, C23.12) 〔对齐〕 ------------------------------------------------------------------------------------------------
EV_EVAL_DIR_NAME = 'eval'                               # bayesreg/eval/holdout.py (ev-tools) EVAL_DIR = holdout_B_v1/eval; its master lock holdout_eval.lock (D109)
EV_MASTER_LOCK_NAME = 'holdout_eval.lock'

# ---- the units (C22.H5-2, C22.H6-3) ---------------------------------------------------------------------------------------------------------------
UNITS = MappingProxyType(dict(members='tag-hb-pq-members', out='tag-hb-pq-out'))
DEV_UNITS = MappingProxyType(dict(members='tag-hb-pq-members-dev', out='tag-hb-pq-out-dev', p3_replay='tag-hb-pq-p3-replay'))
UNIT_SPECS = MappingProxyType(dict(
    members=MappingProxyType(dict(memory_max='12G', memory_swap_max='4G', oom_policy='continue', runtime_max_sec=4800, nice=5)),
    out=MappingProxyType(dict(memory_max='11G', memory_swap_max='4G', oom_policy='continue', runtime_max_sec=7200, nice=5))))
GATES = MappingProxyType(dict(members=MappingProxyType(dict(mem_available_gib=8., gpu_free_min_gib=12., gpu_cap_gib=20.)),
                              out=MappingProxyType(dict(mem_available_gib=9.))))
# 〔估算〕disk (prereg §5.3 both sides): V 27.5 MB x 100 = 2.75 GB on D:; pq.npz about 55 MB + baselines.npz about 200 MB per pair (the development run, run_pq.ESTIMATES) x 50 = 12.75 GB on D:
DISK_NEED_GIB = MappingProxyType(dict(members=MappingProxyType(dict(d=3., home=.2)), out=MappingProxyType(dict(d=13., home=.5))))
ENVIRONMENT_RULE = "hi_core.check_environment against hi_core.DEVCHECK_DECISION (C22.H5-2: Hm5 compares with HI's probs36 written in that environment)"

# ---- paths (fixed; the tests move them into tmp_path) -------------------------------------------------------------------------------------------
MEMBERS_LOCK = Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock')        # C22.H5-4: one fixed path, not derived from an option; O_EXCL; never edited or deleted
OUT_LOCK = Path('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/locks/pq_out.lock')          # the PQ unit runs once too (C22.H6-1 "只写一次")
SMOKE_ROOT = Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v3')                           # D124 (a): one root with HB's new HE smoke (smoke_he.ROOT_PREFIX 'he_smoke')
DEV_PQ_ROOT = SMOKE_ROOT/'pq'                                                              # the dev mode writes only here (C22.H8 冒烟 "只在新根里写")
DEV_MEMBERS_LOCK = DEV_PQ_ROOT/'pq_members_dev.lock'                                       # C22.H8 (i): the two modes' locks are fixed and different
DEV_OUT_LOCK = DEV_PQ_ROOT/'pq_out_dev.lock'
DEV_P3_REPLAY_LOCK = DEV_PQ_ROOT/'pq_p3_replay.lock'
DEV_CELL_REL = 'ii/a/0438-0439'                                                            # 〔对齐〕 HB's smoke_plan §2: segment (ii)'s copy of A's arm a, <root>/<segment>/a/0438-0439/ (its
                                                                                           # review/review.json and terminal.json from hb_review.py dev-unit; a dev review writes no finalized_outputs.json)
# SM3 (runs/20261114-p0/任务书.md; runs/20261109-pqh/smoke_plan_pq_segments.md §4 item 1, the lead's default "加"): the PQ unit of the dev mode also on the fallback pairs that the smoke's (iv-b)
# and (iv-c) leave (hx_fallback_configC, hx_fallback_identity), each on its own copy, with its own lock, its own record directory pq/<segment>/ and pq/big/<segment>/ (the members run of (i) is
# shared) and its own unit.  (ii) keeps DEV_CELL_REL, DEV_OUT_LOCK and DEV_UNITS['out'] as they were.
DEV_OUT_SEGMENTS = ('ii', 'iv-b', 'iv-c')
DEV_OUT_UNITS = MappingProxyType({'ii': DEV_UNITS['out'], 'iv-b': 'tag-hb-pq-out-dev-ivb', 'iv-c': 'tag-hb-pq-out-dev-ivc'})
DEV_OUT_COMMANDS = MappingProxyType({'ii': 'out-dev', 'iv-b': 'out-dev-ivb', 'iv-c': 'out-dev-ivc'})       # run_pqh.py's subcommands
# SM3: the smoke's v3 queue (experiments/he/smoke_v3.py QUEUE_UNIT) is active while it waits for a dev-mode PQ unit it launched; holdout_common.machine_gate takes it as a parent unit (the way
# xb_drive takes the chain's unit).  Never in the hold-out mode.
DEV_PARENT_UNITS = ('tag-he-smoke-v3-queue',)
DEV_SMOKE_PAIR = 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1'                                # C22.H8 冒烟: the development pair 0438-0439
DEV_SMOKE_SUBJECTS = ('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')
PQ_V1 = RESULTS/pins.PQ_ROOT_REL                                                           # the development reading's root (only read: smoke (i) the V record, (iii) the arrays, the replay gate)
# C22.H4-3 revision 7c, as the lead reads it (11-14, B1 on the plans' review; SM3 revision 3): the runtime caches are outside the watched scope; this list as it is at P0 is the pin; the smoke's
# segment (i) checks that every cache directory it used is in it (a breach otherwise) and describes them; not changed after the smoke (tightening it is a P0′)
RUNTIME_CACHE_DIRS = ('/home/rmcfz/.cache', '/home/rmcfz/.triton', '/home/rmcfz/.nv')
PYTHON_ENV = '/home/rmcfz/miniconda3'
WATCHED = ('/home/rmcfz', '/mnt/d')                                                        # C22.H4-3 revision 7a (m15): /mnt/d whole; /home/rmcfz but code, the environment, the caches
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')                             # mc0_core.FROZEN: the out unit's and the replay's jd_curl (PYTHONPATH=<frozen>:<code>); a code tree (m8)
# I7 (the lead's ruling (8) of 11-09): the real roots PQH writes into, spelled out here and never moved by a test.  With XB_TEST_SANDBOX set, every write of PQH refuses a path under one of them
# (pqh_inputs.guard_write), on top of holdout_common.real_root_problem (holdout_B_v1, xb_*, /mnt/d/seq_vp_tag_holdout_B_v1).
REAL_WRITE_ROOTS = (Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v3'), Path('/home/rmcfz/seq_vp_tag_results/pq_replay_v1'),
                    Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock'), Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock.rerun'),
                    Path('/home/rmcfz/seq_vp_tag_results/seg/pq_holdout_members.lock.resume'),       # review NM6: the members lock's control-rerun and resume locks
                    Path('/home/rmcfz/seq_vp_tag_results/holdout_B_v1'), Path('/mnt/d/seq_vp_tag_holdout_B_v1'), Path('/home/rmcfz/seq_vp_tag_results/pq_v1'))

# ---- the replay gate (D119 (a), C22.H8) -------------------------------------------------------------------------------------------------------------
REPLAY_PLAN_REL = 'runs/20261109-pqh/replay_gate_plan.md'
REPLAY_PLAN_SHA256 = '045e68cc5ae224916de5918e2c63a169a40d79b40bd0fc88c56e42386b9924d1'     # P0 (11-14): the final plan (PT 5ef8243); None or another plan refuses the gate's run (its dry run reports it)
REPLAY_ROOT = RESULTS/'pq_replay_v1'                    # a new root (pq_v1 is never written)
REPLAY_LOCK = REPLAY_ROOT/'pq_replay_gate.lock'
REPLAY_READING_PASS = '逐位相同 19/19'
REPLAY_READING_FAIL = '不过（交 BZ）'
REPLAY_READING_UNDECIDED = '不判（交 BZ）'
# Filled by a reviewed commit after the gate ran (C22.H8 "结论由一个经独立复核的提交钉进代码常量"); the evaluation entry checks at its step (0) that it is not None, that its reading is REPLAY_READING_PASS and
# that the result file has this sha256 (C22.H7-1 last item); otherwise the entry refuses before any truth.
PQ_REPLAY_DECISION = {
    'result_json': '/home/rmcfz/seq_vp_tag_results/pq_replay_v1/result/replay_result.json',
    'result_sha256': '3f37ef8e9c4dc5955cb7e6a8fd29e95aa891824d7925bd83f742bbf809bae91a',
    'reading': '逐位相同 19/19',
    'code_commit': '318e72ad3d097bcbd03db180a56c0144ee4344bc',
    'plan_sha256': '045e68cc5ae224916de5918e2c63a169a40d79b40bd0fc88c56e42386b9924d1',
    'environment': {'python': '3.10.20', 'numpy': '2.0.2', 'scipy': '1.15.2', 'nibabel': '5.4.0'},
    'decided_by': 'lead (total responsible), 2026-11-15, per replay_gate_plan.md §4, §6',
    'review': 'docs/bayesreg/reviews/D58_复核_20261115.md (sha256 9629f2495a4f099347871e4eb3d606a22f2484838745e65e494fe1145f946c4b; 0 blocking, 1 important, 8 minor; reading holds)',
}     # P1 (11-15): filled after the gate ran on P0'' 318e72ad (C6, D8); code_commit and environment copied from replay_result.json (code.commit, clean true)
REPLAY_DECISION_KEYS = ('result_json', 'result_sha256', 'reading', 'code_commit', 'plan_sha256', 'environment', 'decided_by', 'review')     # I4: + environment (C23.12-10's constant)
REPLAY_ENVIRONMENT_KEYS = ('python', 'numpy', 'scipy', 'nibabel')                         # I4: the environment the gate ran in (the result records it; the decision copies it)
REPLAY_CODE_FILES = ('bayesreg/eval/holdout.py', 'experiments/pq/pq_core.py', 'experiments/pq/pq_stats.py', 'experiments/pq/pq_io.py', 'experiments/pq/pqh_replay.py')     # I4: their sha256 in the result
REPLAY_UNIT = 'tag-hb-pq-replay'                        # m6: the gate's own unit (CPU), started by run_pqh.py launch replay-gate
REPLAY_UNIT_SPEC = MappingProxyType(dict(memory_max='8G', memory_swap_max='4G', oom_policy='continue', runtime_max_sec=3600, nice=5))     # 〔估算〕19 pairs' truth and arrays, one at a time

# ---- the frozen settings of C22.H3-6 ------------------------------------------------------------------------------------------------------------
JUDGED_BASELINES = ('h', 'H_F', 'H2', 'V_F', 'V2', 'G_plus', 'X_lex', 'B_d')         # = pq_stats.JUDGED_BASELINES (a test compares)
DESCRIBED_SCORES = ('G_minus',)
BASELINE_KEYS = ('h', 'H_F', 'H2', 'V_F', 'V2', 'G_plus', 'G_minus', 'delta')         # = pq_core.BASELINE_KEYS (a test compares)


def frozen_settings():
    """The settings of C22.H3-6 as one JSON-able dict (no file read)."""
    return dict(
        name='PQ (pq1) on the B-line hold-out: an added output of the method Hx24-RF05-F1200-v4 (C22.H2-1; not an arm, D82 = -a)',
        contract=CONTRACT, contract_revision=7, posterior_qc='pq1',
        sampling='iota = warp_labels_l2r of the voxel numbers + 1 (scripts/evaluate_l2r_style.py), order 0, outside -> background; d = disp_voxels(v, spacing) bit for bit output/disp.npz (C22.Q2-1, C22.H4-2 Hm4)',
        v='D113 (a): hx_certified rebuilt from t88/field_0600.npz, boundary_taper (L = 4), repair.npz nodes and values, then bit for bit output/final.npz; a fallback pair: final.npz bound to the fallback record',
        d89='(a): eps-EM described only, pi not stored', judged_baselines=list(JUDGED_BASELINES), described_scores=list(DESCRIBED_SCORES), baseline_keys=list(BASELINE_KEYS),
        storage=dict(h='uint8', others='float32 after float64 arithmetic', pq_npz=['q', 'q4']),
        seg4_table=dict(path=SEG4_TABLE_REL, sha256=SEG4_TABLE_SHA256), anchor=dict(ANCHOR), taper_L=TAPER_L,
        members=[dict(model_id=m, name=n, sha256=s) for m, n, s in MEMBERS], models_root=str(MODELS_ROOT),
        plan=dict(path=PLAN_REL, sha256=PLAN_SHA256), preregistration=dict(path=PREREG_REL, sha256=PREREG_SHA256),
        units={k: dict(unit=UNITS[k], **dict(UNIT_SPECS[k]), gates=dict(GATES[k]), disk_need_gib=dict(DISK_NEED_GIB[k])) for k in UNIT_KEYS},
        environment=ENVIRONMENT_RULE, d114='(a): line.json is never changed; PQ writes pq/<pair>/{pq.json, pq.npz, baselines.npz}', d117='(a)', d123='(b)')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def settings_sha256(settings=None):
    """sha256 of the canonical JSON of the frozen settings (C22.7-3's way); recorded in every pq.json and in the unit records."""
    return hashlib.sha256(canonical(frozen_settings() if settings is None else settings).encode('utf-8')).hexdigest()


# ---- SM3: the dev mode's segments (the constants above, DEV_OUT_SEGMENTS) ----------------------------------------------------------------------
def dev_cell_rel(segment):
    """The copy of segment `segment` under the smoke root: (ii) DEV_CELL_REL, (iv-b) and (iv-c) <segment>/a/0438-0439 (runs/20261109-hb/smoke_plan.md v3 section 2)."""
    if segment not in DEV_OUT_SEGMENTS:
        raise ValueError(f'segment {segment!r}: one of {DEV_OUT_SEGMENTS}')
    return DEV_CELL_REL if segment == 'ii' else f'{segment}/a/0438-0439'
