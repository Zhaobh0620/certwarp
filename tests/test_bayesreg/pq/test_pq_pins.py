"""PQ's pins (plan.md section 1; 任务书 范围 7): the plan's sha256, the 19 pairs' and 20 subjects' tables parsed from the plan again, the anchors against the repository files, the four members against the
registry.  Reads only repository files (the plan, configs/bayesreg/*.json)."""
import hashlib
import json
import re

from . import _world as W

pins = W.pins
ROOT = W.ROOT
HEX = re.compile(r'\b[0-9a-f]{64}\b')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks():
    return re.findall(r'```\n(.*?)```', (ROOT/pins.PLAN_REL).read_text(encoding='utf-8'), re.S)


def test_the_plan_is_the_pinned_one():
    assert sha(ROOT/pins.PLAN_REL) == pins.PLAN_SHA256 == '4c2ac725be266ac1f0312b05096a3913dc62f9be974862c34e8808c4ee43742d'


def test_the_p3_table_is_the_plans():
    lines = [line for line in blocks()[0].splitlines() if line.strip()][2:]
    assert len(lines) == 38
    rows = {}
    for a, b in zip(lines[0::2], lines[1::2]):
        t88, rep = re.search(r'(C_first5|C_rest600) / (RC_c600|RC_rest600)', a).groups()
        h1, h2 = HEX.findall(a), HEX.findall(b)
        rows[a.split()[0]] = pins.P3Row(t88, rep, repair_npz=h1[0], state_hash=h1[1], field_npz=h2[0], repair_json=h2[1])
    assert rows == dict(pins.P3) and tuple(rows) == pins.SHORTS
    assert [r.t88_dir for r in rows.values()].count('C_first5') == 5 and all((r.t88_dir == 'C_first5') == (r.repair_dir == 'RC_c600') for r in rows.values())


def test_the_run_json_table_is_the_plans():
    rows = dict(line.split() for line in blocks()[1].splitlines() if line.strip())
    assert rows == dict(pins.RUN_JSON) and tuple(rows) == pins.SHORTS


def test_the_v4_table_is_the_plans():
    lines = [line for line in blocks()[2].splitlines() if line.strip()][2:]
    rows = {}
    for a, b in zip(lines[0::2], lines[1::2]):
        h1, h2 = HEX.findall(a), HEX.findall(b)
        rows[a.split()[0]] = pins.V4Row(probs36=h1[0], provenance=h1[1], structure_groups=h2[0])
    assert rows == dict(pins.V4) and tuple(rows) == pins.SUBJECTS


def test_the_anchors_are_the_repository_files():
    assert sha(ROOT/pins.ANCHOR['registry']) == pins.ANCHOR['registry_sha256']
    assert sha(ROOT/pins.SEG4_TABLE_REL) == pins.SEG4_TABLE_SHA256
    assert pins.ANCHOR['segmenter_id'] == 'unet35_v4_ens' and pins.ANCHOR['policy'] == 'all_folds_mean'
    text = (ROOT/pins.PLAN_REL).read_text(encoding='utf-8')
    assert pins.ANCHOR['registry_sha256'] in text and pins.SEG4_TABLE_SHA256 in text


def test_the_four_members_are_the_registrys_in_hss_order():
    registry = json.loads((ROOT/pins.ANCHOR['registry']).read_text())['models']
    expected = [(model_id, sha) for model_id in sorted(registry) for sha in registry[model_id]['member_model_sha256']]
    assert [(m, s) for m, _, s in pins.MEMBERS] == expected
    assert [name for _, name, _ in pins.MEMBERS] == ['foldA_s0', 'foldA_s1', 'foldB_s0', 'foldB_s1']


def test_the_chain_of_pairs():
    assert len(pins.SHORTS) == 19 and len(pins.SUBJECTS) == 20
    for i, short in enumerate(pins.SHORTS):
        assert pins.subjects_of(short) == (pins.SUBJECTS[i], pins.SUBJECTS[i+1])
        assert pins.pair_name(short) == f'{pins.SUBJECTS[i]}__{pins.SUBJECTS[i+1]}'
