"""C23.9-11 and C23.9-12 (SG): the static census of the reading primitives and the subject-path constructions (C23.6-3) against its pinned inventory, the
disjointness of the F rows (development constants written in the code) from the sealed lists, and the call sites of bayesreg.io.unseal's load_unseal, bind_unseal
and bind_files (D99).  Reads source files, the allowlist JSON, the sealed id lists and, where the code tree has its data/ link, the development stage's selection.json
and manifests (ids only, through the loader's whitelisting decoder)."""
import ast
import json
from pathlib import Path

import pytest

from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import _read_manifest
from bayesreg.io.sealed import lists_of, load_sealed_lists

from . import _census
from ._census_inventory import F_STAGES, INVENTORY

ROOT = Path(__file__).resolve().parents[3]
CLASSES = frozenset('GLFSDPERT')       # T (LP-C2, contract C26.5-12, D147): the LPBA40 evaluation entry's label reader, pinned by T_ROWS below
# C23.9-12 (D99): the only places that call bayesreg.io.unseal's three functions; SG adds HI's hold-out entry; EV (the lead, 11-07) adds the hold-out
# evaluation entry of C23.12 (admit_holdout: load_unseal, bind_unseal).  The other hold-out tools add their rows through their own contract revision and review.
CALL_SITES = frozenset({('experiments/hi/hi_core.py', 'load_unseal', 'load_unseal'), ('experiments/hi/hi_core.py', 'bind', 'bind_unseal'),
                        ('experiments/hi/hi_core.py', 'bind', 'bind_files'),
                        # XB (runs/20261107-xb, the lead's instruction of 2026-11-07 after SG's merge): the shared hold-out admission; bind_files is not called (no OasisFiles there)
                        ('experiments/holdout/holdout_common.py', 'admit_holdout_pair', 'load_unseal'), ('experiments/holdout/holdout_common.py', 'admit_holdout_pair', 'bind_unseal'),
                        # HP (runs/20261107-hp, branch hp-tools): the hold-out data preparation and Atropos entry admit through hp_core.admit_holdout; bind_files
                        # only for the preparation's image reads.  Registered as implemented: the contract revision of the hold-out call sites does not exist yet
                        # (prereg revision 1a item 18; decision_log 2026-11-09 ruling (1) on the XB review: tool packages may merge first, that revision is adopted
                        # before the unsealing)
                        ('experiments/holdout/hp_core.py', 'admit_holdout', 'load_unseal'), ('experiments/holdout/hp_core.py', 'admit_holdout', 'bind_unseal'),
                        ('experiments/holdout/hp_core.py', 'admit_holdout', 'bind_files'),
                        # EV: the hold-out evaluation entry of C23.12 (admit_holdout: load_unseal, bind_unseal)
                        ('bayesreg/eval/holdout.py', 'admit_holdout', 'load_unseal'), ('bayesreg/eval/holdout.py', 'admit_holdout', 'bind_unseal'),
                        # HB (runs/20261109-hb/实现说明.md): the hold-out queues' first check, before any list of the hold-out is read (no binding: the units admit each pair in their own processes
                        # through admit_holdout_pair above, he_holdout.admitted_images and hb_configc.admit)
                        ('experiments/hb/hb_core.py', 'unseal_gate', 'load_unseal')})
E_ROWS = frozenset({('bayesreg/eval/holdout.py', '<module>'), ('bayesreg/eval/holdout.py', '_load_label_volume'), ('bayesreg/eval/holdout.py', 'read_truth'),
                    ('bayesreg/eval/holdout.py', 'evaluate_pair')})       # C23.6-3: class E, C23.12's entry only (EV, the lead 11-07)
T_ROWS = frozenset({('bayesreg/eval/lpba_eval.py', '<module>'), ('bayesreg/eval/lpba_eval.py', '_label_volume'), ('bayesreg/eval/lpba_eval.py', 'read_label'),
                    ('bayesreg/eval/lpba_eval.py', 'evaluate_pair')})     # C26.5-12 N51, C26.10 T14 (D147): class T, the LPBA40 evaluation entry only (LP-C2)
UNSEAL_FUNCTIONS = ('load_unseal', 'bind_unseal', 'bind_files')
# SG code review m7: hi_core.load_unseal and hi_core.bind are public wrappers of the three; their callers are pinned too (HI's entry only).
HI_WRAPPER_SITES = frozenset({('experiments/hi/hi_core.py', 'infer_holdout', 'load_unseal'), ('experiments/hi/hi_core.py', 'infer_holdout', 'bind'),
                              ('experiments/hi/run_hi.py', 'dry_run_infer', 'load_unseal'), ('experiments/hi/run_hi.py', 'dry_run_infer', 'bind'),
                              # PQH (runs/20261109-pqh): the members unit's dry run reads BZ's record through hi_core (also in INDIRECT_CALL_SITES); pqh_units.members_admission calls
                              # hi_core through h_i = hi() (a function returning the module), which this scan does not follow -- its rows are INDIRECT_CALL_SITES'
                              ('experiments/pq/run_pqh.py', 'dry_run', 'load_unseal')})
HI_WRAPPERS = ('load_unseal', 'bind')


def test_the_census_matches_its_inventory_row_for_row():
    found = _census.scan(ROOT)
    new = sorted(set(found)-set(INVENTORY))
    gone = sorted(set(INVENTORY)-set(found))
    assert not new and not gone, (f'{len(new)} new rows (classify them, have them reviewed): {new[:20]}', f'{len(gone)} rows no longer found: {gone[:20]}')
    assert {cls for cls, _ in INVENTORY.values()} <= CLASSES
    assert all(note for _, note in INVENTORY.values())
    assert {key for key, (cls, _) in INVENTORY.items() if cls == 'E'} == E_ROWS      # C23.12's entry, the only place of class E
    assert {key for key, (cls, _) in INVENTORY.items() if cls == 'T'} == T_ROWS      # C26.7's entry, the only place of class T (LP-C2, D147)


def test_the_r_rows_are_runs_and_the_two_exempted_generators_only():
    """C23.6-3: runs/ is history (R); C23.2-2 exempts experiments/split/draw_split.py and experiments/d0/make_dev_subjects.py (R); nothing else is R, and every runs/ row is R."""
    exempt = ('experiments/split/draw_split.py', 'experiments/d0/make_dev_subjects.py')
    for (path, name), (cls, _) in INVENTORY.items():
        assert (cls == 'R') == (path.startswith('runs/') or path in exempt), (path, name, cls)


def test_the_rows_named_by_the_contract_have_the_contracts_class():
    """C23.6-1 and C23.6-2; e2e_control.field_process, setup and pair_files are S since revisions 6f, 6g."""
    want = {('bayesreg/seg/unet/data.py', 'OasisFiles.image_path'): 'P', ('bayesreg/seg/unet/data.py', 'OasisFiles.label_path'): 'P', ('bayesreg/seg/unet/data.py', 'read_image'): 'P',
            ('bayesreg/seg/unet/synthetic.py', 'write_pair'): 'S', ('bayesreg/seg/channels.py', 'read_back_probabilities'): 'D', ('bayesreg/seg/unet/infer.py', 'read_probabilities'): 'D',
            ('bayesreg/solve/init_field.py', '_load_frozen'): 'D', ('scripts/evaluate_l2r_style.py', 'evaluate_run'): 'P', ('experiments/seed/e2e_control.py', 'field_process'): 'S',
            ('experiments/d0/check_seg4_map.py', 'run'): 'P', ('experiments/d0/evaluate.py', 'truth_labels'): 'F', ('experiments/d0/atropos_dice.py', 'main'): 'F',
            ('experiments/unv2/eval_seg.py', 'read_truth'): 'P', ('experiments/nn/eval_nn.py', 'read_nn_prediction'): 'P', ('experiments/fl/nx_seg.py', 'read_labels_file'): 'P',
            ('experiments/he/firewall_c.py', 'firewall_check'): 'G', ('experiments/seed/e2e_control.py', 'setup'): 'S', ('experiments/seed/e2e_control.py', 'pair_files'): 'S'}
    assert {key: INVENTORY[key][0] for key in want} == want


def test_the_f_rows_name_development_stages_disjoint_from_the_sealed_lists():
    """C23.6-3: every F row names its stage (F_STAGES); the stage is a development source of the repository's allowlist whose ids are on no sealed list.  Where the code
    tree has its data/ link (the main checkout), the stage's selection.json pairs and their manifests' ids (read through the whitelisting decoder) are checked too."""
    assert sorted(F_STAGES) == sorted(key for key, (cls, _) in INVENTORY.items() if cls == 'F')
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    sources = {s['name']: s for s in allowlist.sources}
    sealed = load_sealed_lists(ROOT)
    for key, stage in F_STAGES.items():
        assert stage in sources, (key, stage)
        on = sorted(s for s in sources[stage]['subjects'] if lists_of(sealed, s))
        assert on == [], (key, stage, on)
    for stage in sorted(set(F_STAGES.values())):
        directory = ROOT/'data'/'three_d'/stage
        if not (directory/'selection.json').is_file():
            continue                                  # a worktree has no data/ link; the allowlist source (pinned by its sha256 in the allowlist) is checked above
        pairs = [row['pair'] for row in json.loads((directory/'selection.json').read_text(encoding='utf-8'))['pairs']]
        for pair in pairs:
            _, _, selected = _read_manifest(directory/pair/'manifest.json')
            for subject in (selected['fixed_subject'], selected['moving_subject']):
                assert not lists_of(sealed, subject) and subject in sources[stage]['subjects'], (stage, pair, subject)


def unseal_calls(path):
    """(function, called name) of every call of bayesreg.io.unseal's three functions in the file: through a module alias of bayesreg.io.unseal or a name imported from it."""
    tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            source = (node.module or '')
            for alias in node.names:
                if source in ('bayesreg.io', 'io', '') and alias.name == 'unseal' or source.endswith('.io') and alias.name == 'unseal':
                    modules.add(alias.asname or alias.name)
                if (source.endswith('io.unseal') or source == 'unseal') and alias.name in UNSEAL_FUNCTIONS:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith('io.unseal'):
                    modules.add(alias.asname or alias.name)
    out = set()

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr in UNSEAL_FUNCTIONS and isinstance(func.value, (ast.Name, ast.Attribute)) \
                        and ast.unparse(func.value) in modules | {f'bayesreg.io.unseal'}:
                    out.add((scope, func.attr))
                elif isinstance(func, ast.Name) and func.id in names:
                    out.add((scope, func.id))
            visit(child, inner)
    visit(tree, '<module>')
    return out


def test_the_call_sites_of_load_unseal_bind_unseal_and_bind_files_are_pinned():
    found = set()
    for path in _census.files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel == 'bayesreg/io/unseal.py':
            continue
        found |= {(rel, scope, name) for scope, name in unseal_calls(path)}
    assert found == CALL_SITES, sorted(found ^ CALL_SITES)


def hi_wrapper_calls(path, *, inside_hi_core=False):
    """(function, called name) of every call of hi_core.load_unseal / hi_core.bind: through a module alias of hi_core, a name imported from it, or (in hi_core
    itself) the bare name."""
    tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    modules, names = set(), set(HI_WRAPPERS) if inside_hi_core else set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[-1] == 'hi_core':
            names |= {alias.asname or alias.name for alias in node.names if alias.name in HI_WRAPPERS}
        elif isinstance(node, ast.ImportFrom):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name == 'hi_core'}
        elif isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name.split('.')[-1] == 'hi_core'}
    out = set()

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr in HI_WRAPPERS and ast.unparse(func.value) in modules:
                    out.add((scope, func.attr))
                elif isinstance(func, ast.Name) and func.id in names:
                    out.add((scope, func.id))
            visit(child, inner)
    visit(tree, '<module>')
    return out


def test_the_call_sites_of_hi_cores_load_unseal_and_bind_are_pinned():
    """Review m7: the callers of HI's public wrappers of load_unseal / bind_unseal / bind_files."""
    found = set()
    for path in _census.files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        found |= {(rel, scope, name) for scope, name in hi_wrapper_calls(path, inside_hi_core=rel == 'experiments/hi/hi_core.py')}
    found -= {('experiments/hi/hi_core.py', scope, name) for scope, name in (('load_unseal', 'load_unseal'), ('bind', 'bind'))}       # the wrappers' own names, not calls
    assert found == HI_WRAPPER_SITES, sorted(found ^ HI_WRAPPER_SITES)


def test_the_hi_wrapper_scan_sees_each_way_of_calling(tmp_path):
    sample = tmp_path/'sample.py'
    sample.write_text('import hi_core as hi\nfrom hi_core import bind as b\nfrom experiments.hi import hi_core\n'
                      'def a():\n    hi.load_unseal(1)\ndef c():\n    b(1, 2)\ndef d():\n    hi_core.bind(1, 2)\n', encoding='utf-8')
    assert hi_wrapper_calls(sample) == {('a', 'load_unseal'), ('c', 'b'), ('d', 'bind')}


def test_the_call_site_scan_sees_each_way_of_calling(tmp_path):
    sample = tmp_path/'sample.py'
    sample.write_text('from bayesreg.io import unseal as U\nfrom bayesreg.io.unseal import bind_files as bf\nimport bayesreg.io.unseal\n'
                      'def a():\n    U.load_unseal(1)\ndef b():\n    bf(1, 2)\ndef c():\n    bayesreg.io.unseal.bind_unseal(1, 2)\n', encoding='utf-8')
    assert unseal_calls(sample) == {('a', 'load_unseal'), ('b', 'bf'), ('c', 'bind_unseal')}


# PQH (runs/20261109-pqh; contract C22.H4-4, C23.8-1, C23.9-12): the hold-out entries that reach SG through hi_core (load_unseal, bind: the same record, the allowlist and the OasisFiles bound)
# or holdout_common (admit_holdout_pair: load_unseal, bind_unseal, check_subject) rather than calling bayesreg.io.unseal themselves; a new entry of this kind adds its row here and to its
# implementation notes' call-site table.  The calls inside the defining modules are CALL_SITES' direct rows.
INDIRECT_CALL_SITES = frozenset({('experiments/hi/run_hi.py', 'dry_run_infer', 'hi.bind'), ('experiments/hi/run_hi.py', 'dry_run_infer', 'hi.load_unseal'),
                                 ('experiments/xb/xb_drive.py', 'admit', 'hc.admit_holdout_pair'),
                                 # HB (merged d008f50; contract revision 9 draft C24.8 item 10): HE's hold-out images and the config C arm admit each pair through holdout_common
                                 ('experiments/he/he_holdout.py', 'admitted_images', 'hc.admit_holdout_pair'), ('experiments/hb/hb_configc.py', 'admit', 'hc.admit_holdout_pair'),
                                 ('experiments/pq/pqh_units.py', 'admit_pairs', 'h.admit_holdout_pair'), ('experiments/pq/pqh_units.py', 'members_admission', 'h_i.bind'),
                                 ('experiments/pq/pqh_units.py', 'members_admission', 'h_i.load_unseal'), ('experiments/pq/run_pqh.py', 'dry_run', 'hc.admit_holdout_pair'),
                                 ('experiments/pq/run_pqh.py', 'dry_run', 'hi.load_unseal'),
                                 # A5b (C24.1 row 29): the repair ablation's preparation admits each pair through holdout_common before it reads anything of the pair
                                 ('experiments/hb/hb_prerepair.py', 'prepare', 'hc.admit_holdout_pair')})
INDIRECT_NAMES = ('load_unseal', 'admit_holdout_pair', 'bind')
HI_ALIASES = ('hi', 'h_i', 'hi_core')
DIRECT_ALIASES = ('bayes_unseal', 'unseal_module', 'U')


def indirect_calls(path):
    """(function, alias.name) of every call alias.load_unseal / alias.admit_holdout_pair (any module alias but bayesreg.io.unseal's) and of hi_core's alias.bind in the file."""
    tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    out = set()

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, f'{owner}.{child.name}' if owner else child.name)
                continue
            func = child.func if isinstance(child, ast.Call) else None
            if isinstance(func, ast.Attribute) and func.attr in INDIRECT_NAMES and isinstance(func.value, ast.Name) and func.value.id not in DIRECT_ALIASES:
                if func.attr != 'bind' or func.value.id in HI_ALIASES:
                    out.add((owner or '<module>', f'{func.value.id}.{func.attr}'))
            visit(child, owner)
    visit(tree, '')
    return out


def test_the_indirect_call_sites_of_the_hold_out_entries_are_pinned():
    found = set()
    for path in _census.files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith('experiments/'):
            found |= {(rel, owner, call) for owner, call in indirect_calls(path)}
    assert found == INDIRECT_CALL_SITES, sorted(found ^ INDIRECT_CALL_SITES)


# ---- A5b (contract revision 9 draft C24.1-3; decision_log 2026-11-10 ~2:33 AM CST (2)): every caller of every hold-out admission wrapper -----------------------------------
# hi_core.load_unseal and hi_core.bind, holdout_common.admit_holdout_pair, hp_core.admit_holdout, bayesreg.eval.holdout.admit_holdout, hb_core.unseal_gate, he_holdout.admitted_images and
# run_hx.admitted_images (which forwards to it): one row more or less fails.  The calls of bayesreg.io.unseal's own functions are CALL_SITES' rows (left out here).  A new hold-out entry
# registers its row here, in its implementation notes' call-site table and in the call-site contract revision.
WRAPPER_NAMES = ('load_unseal', 'bind', 'admit_holdout_pair', 'admit_holdout', 'unseal_gate', 'admitted_images')
WRAPPER_CALL_SITES = frozenset({
    ('bayesreg/eval/holdout.py', '_open_holdout', 'admit_holdout'), ('bayesreg/eval/holdout.py', 'dry_run', 'admit_holdout'),
    ('experiments/hb/hb_configc.py', 'admit', 'hc.admit_holdout_pair'), ('experiments/hb/hb_method.py', 'start_checks', 'hb.unseal_gate'),
    ('experiments/hb/hb_prerepair.py', 'prepare', 'hc.admit_holdout_pair'),
    ('experiments/hb/hb_review.py', 'continue_fallback', 'R.admitted_images'), ('experiments/hb/hb_review.py', 'review_pair', 'R.admitted_images'),
    ('experiments/he/he_holdout.py', 'admitted_images', 'hc.admit_holdout_pair'), ('experiments/he/run_hx.py', 'admitted_images', 'hold.admitted_images'),
    ('experiments/he/run_hx.py', 'dry_run', 'admitted_images'), ('experiments/he/run_hx.py', 'holdout_dry_run', 'admitted_images'),
    ('experiments/he/run_hx.py', 'repair_main', 'admitted_images'), ('experiments/he/run_hx.py', 'run_pair', 'admitted_images'),
    ('experiments/he/run_hx.py', 't88_main', 'admitted_images'), ('experiments/he/run_hx.py', 'third_cert_main', 'admitted_images'),
    ('experiments/hi/hi_core.py', 'infer_holdout', 'bind'), ('experiments/hi/hi_core.py', 'infer_holdout', 'load_unseal'),
    ('experiments/hi/run_hi.py', 'dry_run_infer', 'hi.bind'), ('experiments/hi/run_hi.py', 'dry_run_infer', 'hi.load_unseal'),
    ('experiments/holdout/atropos_holdout.py', 'run_holdout', 'core.admit_holdout'), ('experiments/holdout/prep_stage.py', 'run_holdout', 'core.admit_holdout'),
    ('experiments/pq/pqh_units.py', 'admit_pairs', 'h.admit_holdout_pair'), ('experiments/pq/pqh_units.py', 'members_admission', 'h_i.bind'),
    ('experiments/pq/pqh_units.py', 'members_admission', 'h_i.load_unseal'), ('experiments/pq/run_pqh.py', 'dry_run', 'hc.admit_holdout_pair'),
    ('experiments/pq/run_pqh.py', 'dry_run', 'hi.load_unseal'), ('experiments/xb/xb_drive.py', 'admit', 'hc.admit_holdout_pair'),
    # T24 (contract C24.12-3, revision 9d; BZ 2026-11-19 (乙)): the 24-tetrahedron proxy after the unsealing admits through the evaluation entry's admission (SG)
    ('bayesreg/eval/holdout_tet24.py', '_admit', 'H.admit_holdout')})
DIRECT_UNSEAL_CALLS = ('unseal_module.load_unseal', 'bayes_unseal.load_unseal')       # bayesreg.io.unseal's own (CALL_SITES)
DIRECT_BARE = {('bayesreg/eval/holdout.py', 'load_unseal'), ('experiments/holdout/hp_core.py', 'load_unseal')}   # imported from bayesreg.io.unseal there (CALL_SITES)


def wrapper_calls(path):
    """(owner, call) of every call of a WRAPPER_NAMES name in the file: alias.name (any module alias) or the bare name; owner the qualified function or class.method."""
    tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    out = set()

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, f'{owner}.{child.name}' if owner else child.name)
                continue
            func = child.func if isinstance(child, ast.Call) else None
            if isinstance(func, ast.Attribute) and func.attr in WRAPPER_NAMES and isinstance(func.value, ast.Name):
                out.add((owner or '<module>', f'{func.value.id}.{func.attr}'))
            elif isinstance(func, ast.Name) and func.id in WRAPPER_NAMES:
                out.add((owner or '<module>', func.id))
            visit(child, owner)
    visit(tree, '')
    return out


def test_every_caller_of_every_hold_out_admission_wrapper_is_pinned():
    found = set()
    for path in _census.files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(('experiments/', 'bayesreg/')):
            found |= {(rel, owner, call) for owner, call in wrapper_calls(path) if call not in DIRECT_UNSEAL_CALLS and (rel, call) not in DIRECT_BARE}
    assert found == WRAPPER_CALL_SITES, sorted(found ^ WRAPPER_CALL_SITES)
    assert set(INDIRECT_CALL_SITES) <= found                         # the older, narrower pin is a part of this one


def test_the_wrapper_scan_sees_each_way_of_calling(tmp_path):
    sample = tmp_path/'sample.py'
    sample.write_text('import holdout_common as hc\nclass A:\n    def m(self):\n        hc.admit_holdout_pair(1, 2)\ndef f():\n    admitted_images(1)\n'
                      'def g():\n    core.admit_holdout(1)\n', encoding='utf-8')
    assert wrapper_calls(sample) == {('A.m', 'hc.admit_holdout_pair'), ('f', 'admitted_images'), ('g', 'core.admit_holdout')}


# ---- A5b (C24.1-4; decision_log 2026-11-10 ~2:33 AM CST (1)): the readings the census' patterns cannot see, kept by hand ---------------------------------------------
# Each row: (file, function, what it reads, order) -- the function must exist; `order` (when given) is (scope, [calls]) and the first call of each name in that scope must come in
# that order (the admission or the check before the read).  GONE: functions of the contract's table that no longer exist (a reader that comes back must be classified again).
MANUAL_ROWS = (
    ('experiments/xb/xb_core.py', 'check_pair_input', "N1: the pair's manifest (_read_manifest), then the general guard with the admission's allowlist, then pair.npz's container sha256",
     ('check_pair_input', ['_read_manifest', 'check_pair_subjects', 'sha256'])),
    ('experiments/hb/hb_store.py', 'input_lock', "N11: HI's predictions and the method cells' field_0600.npz by sha256 (nothing decoded), after the start checks",
     ('main', ['start_checks', 'input_lock'])),
    ('experiments/hb/hb_store.py', 'move_pair', 'N12: every file of a finalized method cell (copied, sha256 before and after), after the finalized record is checked',
     ('move_pair', ['finalized', 'move_files'])),
    ('experiments/hb/hb_store.py', 'move_resume', 'N12: the files a half move left, after the committed resume record', ('move_resume', ['load_resume', 'move_files'])),
    ('experiments/hb/hb_store.py', 'where', 'N14: move.json, move_resume.json (records)', None),
    ('experiments/hb/hb_review.py', 'check_inputs', "N15: the cell's files listed in the input manifest (sha256), under the review spy, before this process's admission",
     ('review_pair', ['check_inputs', 'admitted_images', 'take_lock'])),
    ('experiments/hb/hb_review.py', 'tree_rows', 'N16: every file of the cell and attempt directories (sha256)', None),
    ('experiments/hb/hb_review.py', 'evidence_rows', 'N16: the evidence directories (sha256)', None),
    ('experiments/hb/hb_review.py', 'copy_to_home', 'N17: a step-5 record (bytes copied to /home)', None),
    ('experiments/hb/hb_review.py', 'keep_on_home', 'N17: review.json and terminal.json (bytes copied to /home)', None),
    ('experiments/hb/hb_review.py', 'third_child_records', "N18: the third-certification child's review/*.json", None),
    ('experiments/hb/hb_review.py', 'fallback_record', 'N18: fallback.json of a third certification', None),
    ('experiments/xb/xb_core.py', 'method_t88_run', "N19: HB's finalized record and move.json (records), read by the driver before its spy", None),
    ('experiments/xb/xb_core.py', 'method_t88_sha256', "N19: HB's finalized record (a record)", None),
    ('experiments/hb/hb_prerepair.py', 'prepare', 'N27: the admission first, then the move records, the stage manifest (ids, grid) and the field',
     ('prepare', ['admit_holdout_pair', 'moved', '_read_manifest', 'load'])),
    ('bayesreg/eval/holdout.py', 'check_ablation', 'N28: the preparation records (prerepair.json) in (b), after (a); hashed into the manifest check of (c)',
     ('_open_holdout', ['rule_one', 'admit_holdout', 'check_cells', 'check_ablation', 'check_manifest', 'take_lock'])),
    ('bayesreg/eval/holdout.py', 'evaluate_ablation', 'N28: prerepair_disp.npz as a plain displacement, hashed against the manifest before the read',
     ('evaluate_ablation', ['verify_files', 'read_disp', 'score_d'])),
    ('bayesreg/eval/holdout.py', 'finalized_record', "N39 (A5b): HB's finalized record (a record, hashed into the manifest check)", None),
    ('bayesreg/eval/holdout.py', 'method_moves', 'N40 (A5b): move.json / move_resume.json (records, hashed into the manifest check)', None),
    ('bayesreg/eval/holdout.py', 'stop_cell', "N41 (A5b): XB's arm_stopped.json at the committed stop record's sha256 (a record)", None),
    ('bayesreg/eval/holdout.py', 'p20_cell', "N42 (A5b): the cell's refusal copies in its record directory (records)", None),
    ('bayesreg/io/rejudgment.py', 'rejudgment', 'N29: a committed rejudgment record (checked committed and unchanged before it is read) and the sha256 of the record it judges',
     ('rejudgment', ['tracked_unchanged', 'read_bytes'])),
    ('experiments/holdout/holdout_common.py', 'rejudgment', 'N29: the same (holdout_common carries or wraps the reader)', None),
    ('experiments/hb/hb_review.py', 'review_rejudgment', "N30: a review unit's rejudgment record (a trace only)", None),
    ('experiments/hb/hb_review.py', 'last_stop_record', 'N30: the /home stop records (by seq)', None),
    ('experiments/hb/hb_review.py', 'unit_stop_record', 'N30: writes review_unit_stopped (a record)', None),
    ('experiments/hb/hb_configc.py', 'keep_refusals', "N31: the batch refusal records, copied into the cell's record directory", None),
    ('experiments/hb/hb_configc.py', 'declined_refusal', "N31: the newest refusal copy and BZ's no_resume", None),
    ('experiments/hb/hb_configc.py', 'effective_path', "N31: the configuration C arm's effective terminal record", None),
    ('experiments/holdout/noseg_pair.py', 'run', "P0S I14: the development pair's admission (C23.4-2), the guard on the loader's hook, then load_pair, then the copy's members (strip)",
     ('run', ['admit_dev_pair', 'MemberGuard', 'load_pair', 'strip'])),
    # Contract C24.11 (revision 9c, BZ 2026-11-15 ~4:20 AM CST; C24.11-5, the rows suggested by C24.11-6): readings in the code before P0 (af593248) that C24.1 did not list; all class D
    # (records and derived files of HI, HB and XB read by path; no subject's image or label).  The line numbers in the notes are P0's.
    ('experiments/hb/hb_core.py', 'hi_union', "N32 (:413): HI's hold-out lock (through hi_core.run_outs), its runs' records (run_consistency_problems, run_files, complete_subjects), "
     "<root>/subjects/<s>.json, the directory names under pred/ and the sha256 of pred/<s>/'s probs36.npz, structure_groups.npz, provenance.json (nothing decoded); not attempt<k>/", None),
    ('experiments/hb/hb_method.py', 'queue', "N32: the method arms' queue checks HI's union (hb_core.hi_union, :308) after its start checks (:288); the input lock's call (hb_store.input_lock :180) "
     "is N11's row", ('queue', ['start_checks', 'hi_union'])),
    ('experiments/pq/pqh_inputs.py', 'hi_files', "N33 (:487): the paths of the PQ units' spy allowlist: the lock's run roots (hi_run_outs), each root's run records (hi_core.run_files, a listing), "
     "the subject records and the three prediction files; it reads the lock and lists directories, opens no prediction", None),
    ('experiments/xb/xb_drive.py', 'method_run_check', "N34 (:157): the method arm's t88/run.json of this pair (its path from xb_core.method_t88_run; on the hold-out from N19's finalized and "
     "move records): the path, the name rule, the file there, labels_sha256 of both sides, its sha256 equal to the finalized record's; before the lock",
     ('_run', ['admit', 'method_run_check', 'check_pair_input'])),
    ('experiments/hi/hi_core.py', 'lock_lines', "N36 (:502): HI's lock file (its JSON lines)", None),
    ('experiments/hi/hi_core.py', 'run_lines', "N36 (:514): the lock's and its resume locks' lines (<lock>.resume_*, globbed)", None),
    ('experiments/hi/hi_core.py', 'run_outs', "N36 (:523): the output roots of the lock's runs", None),
    ('experiments/hi/hi_core.py', 'run_files', "N36 (:228): a run root's run records (a listing of RUN_FILE names)", None),
    ('experiments/hi/hi_core.py', 'subject_records', "N36 (:536): the file names under <root>/subjects/ (a listing)", None),
    ('experiments/hi/hi_core.py', 'finished_subjects', "N36 (:548): the subject records of the lock's run roots (listings)", None),
    ('experiments/hi/hi_core.py', 'complete_subjects', "N36 (:557): finished_subjects against the given subjects", None),
    ('experiments/hi/hi_core.py', 'run_consistency_problems', "N36 (:564): the run records' hi_stopped and hi_record (RUN_FILE), the listing of subjects/, whether pred/<s>/provenance.json "
     "is there; not the content of subjects/<s>.json", None),
    ('experiments/hi/hi_core.py', 'residue_subjects', "N36 (:262): the directory names under pred/", None),
    ('experiments/hi/hi_core.py', 'written_subject_problems', "N36 (:234): the content of <root>/subjects/<s>.json and the sha256 of the pred/<s>/ files it names and of provenance.json", None),
    ('experiments/hi/hi_core.py', 'keep_residue', "N36 (:268): the sha256 of every file of pred/<s>/ before the directory moves whole to attempt<k>/, checked again after", None),
    ('experiments/xb/xb_core.py', 'check_prediction', "N37 (:508): this side's provenance.json (the anchor unet35_v4_ens and the rest) and the sha256 of probs36.npz (not decoded); before the lock",
     None),
    ('experiments/xb/xb_core.py', 'check_hi_record', "N37 (:541): HI's subject record subjects/<s>.json (output_sha256, provenance_sha256); before the lock", None),
    ('experiments/xb/xb_drive.py', '_run', "N37: xb_drive._run checks the predictions (check_prediction :324) after the admission (:297), then, on the hold-out, HI's subject records "
     "(check_hi_record :327)", ('_run', ['admit', 'check_prediction', 'check_hi_record'])),
    ('experiments/pq/pqh_inputs.py', 'hi_run_outs', "N38 (:447): the lock's run roots (hi_core.run_outs)", None),
    ('experiments/pq/pqh_inputs.py', 'hi_predictions', "N38 (:452): whether the lock is there, hi_core.run_consistency_problems, each subject's subjects/<s>.json (output_sha256)", None),
    ('experiments/xb/xb_core.py', 'method_t88_lookup', "N19 (:247, C24.11-4): HB's finalized record and move.json, move_resume.json (records); method_t88_holdout (:269) only forwards to it",
     None),
    # Contract C24.12 (revision 9d; BZ 2026-11-19 (乙), decision_log 2026-11-19 last line of 36a40941): the 24-tetrahedron proxy after the unsealing (N43); class D (the step-11 records, the
    # evaluation's lock lines, the stage manifests and the arms' fields, by path, each field at its manifest sha256; no truth, image, label, PQ or hold file)
    ('bayesreg/eval/holdout_tet24.py', 'admit', "N43: the spy first (the data root watched from the start), then the code identity of C24.12-2, then the admission",
     ('admit', ['start_spy', 'identity', '_admit'])),
    ('bayesreg/eval/holdout_tet24.py', '_admit', "N43 (C24.12-3): SG's admission, the pins, the machine gate, the committed manifest, the evaluation's lock lines, the step-11 records "
     "(holdout_decide.collect), the plan, the spy's allowance, then the stage manifests (holdout.check_stage, hashed first)",
     ('_admit', ['admit_holdout', 'check_prereg', 'load_arm_table', 'check_code_place', 'check_root', 'machine_gate', 'read_committed', 'run_lines', 'collect',
                 'slim_records', 'plan_cells', 'allow', 'check_stage'])),
    ('bayesreg/eval/holdout_tet24.py', 'compute', "N43 (C24.12-5; revision 9d-1, review minor 3): folding_metrics checked, then the lock (or a committed rerun authorization "
     "and its lock), before the first field", ('compute', ['folding_module', 'take_lock', 'compute_cell'])),
    ('bayesreg/eval/holdout_tet24.py', 'compute_cell', "N43 (C24.12-4): the field hashed against the manifest, describe_run and load_field (holdout.score_cell's call), hashed again, "
     "d_to_u (holdout.score_d's), sha256(u) against the step-11 BL1 state_hash, then folding_metrics.tet_ratio_summary",
     ('compute_cell', ['verify_files', 'describe_run', 'load_field', 'd_to_u', 'array_sha256', 'tet_ratio_summary'])),
    # Contract C26.5-12 N57 (revision 11f; LP-C2 final diff review minor 2, the lead 2026-11-20 ~3:23 AM CST): the LPBA40 step-10 manifest generator.  None of the census' patterns matches
    # it (it builds no subject path and hashes bytes through its own sha256_file), so its readings are pinned here, D108's way of listing a step-10 script.  Class G: every subject file it
    # touches (the admitted subjects' original image.nii.gz, by sha256 only) comes after rule one and lpba_common.admit (check_subject on the pinned LPBA allowlist: L34's admission);
    # the rest are the cells' records and files and the stage's manifest.json / pair.npz (bytes hashed, nothing decoded; the stage manifests read by lpba_eval.check_stage after the list).
    ('experiments/lpba/make_eval_manifest_lpba.py', 'run', "N57 (G): --out checked, then the guard (no label file, nothing under lpba_eval.spy_roots() but the originals) and the "
     "evaluation entry's spy, before anything is derived", ('run', ['check_out', 'guard_on', 'start_spy', 'derive'])),
    ('experiments/lpba/make_eval_manifest_lpba.py', 'derive', "N57 (G): rule one, lpba_common.admit, then the entry's cells, original images and input roles; every input hashed "
     "(bytes only), the entry's spy allowance, then the stage manifests (lpba_eval.check_stage, hashed against the list first)",
     ('derive', ['check_code_identity', '_admit', 'check_cells', 'image_originals', 'input_roles', 'sha256_file', 'allow', 'check_stage'])),
    ('experiments/lpba/make_eval_manifest_lpba.py', 'sha256_file', "N57 (G): the bytes of one input (sha256, nothing decoded), the guard's check before the open",
     ('sha256_file', ['guard_path', 'open'])),
    ('experiments/lpba/make_eval_manifest_lpba.py', 'verify_text', "N57 (G): the manifest's text read as the entry reads it (parse_manifest), the sha256 checks, then (read-back) every "
     "input hashed a second time", ('verify_text', ['parse_manifest', '_cell_sha_checks', 'sha256_file'])))
GONE = (('experiments/hb/hb_store.py', 'pq_done'), ('experiments/hb/hb_store.py', 'committed_file'))     # N13: removed in HB revision 3 (decision_log 2026-11-10 ~5:29 AM CST (4))


def _functions(tree):
    """{qualified name: node} of the module's functions and methods."""
    out = {}

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{owner}.{child.name}' if owner else child.name
                if not isinstance(child, ast.ClassDef):
                    out[name] = child
                visit(child, name)
    visit(tree, '')
    return out


def _first_call_line(node, name):
    lines = [c.lineno for c in ast.walk(node) if isinstance(c, ast.Call) and
             ((isinstance(c.func, ast.Attribute) and c.func.attr == name) or (isinstance(c.func, ast.Name) and c.func.id == name))]
    return min(lines) if lines else None


def test_the_readings_the_census_cannot_see_are_pinned_by_hand():
    """C24.1-4: the rows the census patterns do not match are pinned here: the function is there, and where the order matters the admission (or the check) comes before the read in
    the source (the first call of each name, in order)."""
    for rel, function, note, order in MANUAL_ROWS:
        functions = _functions(ast.parse((ROOT/rel).read_text(encoding='utf-8')))
        assert function in functions, (rel, function, note)
        if order is not None:
            scope, calls = order
            assert scope in functions, (rel, scope)
            lines = [_first_call_line(functions[scope], c) for c in calls]
            assert None not in lines and lines == sorted(lines) and len(set(lines)) == len(lines), (rel, scope, dict(zip(calls, lines)), note)
    for rel, function in GONE:
        assert function not in _functions(ast.parse((ROOT/rel).read_text(encoding='utf-8'))), (rel, function)


def test_the_hand_kept_order_check_fails_on_a_read_before_its_admission(tmp_path):
    sample = tmp_path/'s.py'
    sample.write_text('def f(p):\n    load(p)\n    admit_holdout_pair(1, 2)\n', encoding='utf-8')
    f = _functions(ast.parse(sample.read_text()))['f']
    assert _first_call_line(f, 'load') < _first_call_line(f, 'admit_holdout_pair')
