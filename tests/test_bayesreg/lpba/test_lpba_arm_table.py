"""LP-T0 item 2: configs/frozen/lpba40_test_v1_arm_table.json -- made by runs/20261119-lpba/make_lpba40_arm_table.py byte for byte, of the LPBA40 form
(experiments/lpba/lpba_arm_table.validate_lpba40_arm_table) and the hold-out's per-arm form, each arm the hold-out's with only its roots replaced; reads files of this
code tree only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'/'lpba'))
import lpba_arm_table as lat  # noqa: E402

from bayesreg.eval import holdout  # noqa: E402


def make_module():
    spec = importlib.util.spec_from_file_location('make_lpba40_arm_table', ROOT/'runs'/'20261119-lpba'/'make_lpba40_arm_table.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def committed():
    return (ROOT/lat.ARM_TABLE_REL).read_bytes()


@pytest.fixture(scope='module')
def holdout_table():
    raw = (ROOT/lat.HOLDOUT_TABLE_REL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lat.HOLDOUT_TABLE_SHA256 == holdout.ARM_TABLE_SHA256
    return json.loads(raw)


def test_the_committed_table_is_the_generators_output_byte_for_byte(committed):
    assert make_module().build() == committed
    assert committed.endswith(b'}\n') and committed.count(b'\n}\n') == 1


def test_the_committed_table_passes_both_forms(committed):
    table = lat.load_bytes(committed)
    assert [a['name'] for a in table['arms']] == ['method', 'configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg']
    assert table['reference'] == [{'name': 'identity', 'field': 'zero', 'family': None}] and table['pq'] is None and table['segmentation'] is None
    assert table['main_family'] == ['configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg']


def test_each_arm_is_the_holdouts_with_only_its_roots_replaced(committed, holdout_table):
    table = json.loads(committed)
    by_name = {a['name']: a for a in holdout_table['arms']}
    for arm in table['arms']:
        old = by_name[arm['name']]
        assert arm == lat._map_strings(copy.deepcopy(old))
        assert set(arm) == set(old) and all(arm[k] == old[k] for k in arm if k not in ('record_dir', 'field', 'stop_record', 'authorization'))
        assert arm['authorization']['tool'] == old['authorization']['tool'] and arm['authorization']['arm'] == old['authorization']['arm']
    text = committed.decode('utf-8')
    arms_text = json.dumps(table['arms'])+table['stage_root']
    assert 'holdout_B_v1' not in arms_text and '/mnt/' not in arms_text and 'seq_vp_tag_holdout' not in arms_text
    assert text.count('/home/rmcfz/seq_vp_tag_results/lpba40_test_v1/') >= 1+6*2


def test_the_root_map_is_the_holdouts_roots(holdout_table):
    """The hold-out's /home root and D: root (holdout_common.HOLDOUT_ROOT, BIG_ROOT as the hold-out table writes them) and its stop records' directory."""
    assert lat.ROOT_MAP[0][0] == str(holdout.HOLDOUT_RESULTS)+'/' and holdout_table['stage_root'].startswith(lat.ROOT_MAP[0][0])
    strings = list(lat._strings(holdout_table['arms']))
    absolute = [s for s in strings if s.startswith('/')]
    assert absolute and all(s.startswith((lat.ROOT_MAP[0][0], lat.ROOT_MAP[1][0])) for s in absolute)
    assert all(s.startswith(lat.ROOT_MAP[2][0]) for s in (a['stop_record'] for a in holdout_table['arms']))
    assert lat.remap('/mnt/d/seq_vp_tag_holdout_B_v1/method/{pair}') == '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1/work/method/{pair}'
    assert lat.remap('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/{arm}/{pair}') == '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1/{arm}/{pair}'


def test_the_frozen_pins_hold_on_this_tree():
    make_module().check_pins(ROOT)
    assert lat.FROZEN['method']['settings_sha256'] == '414eccd37fe264d081e87bb63235790dc3a1ec40655c57ac64337bf84992f35d'
    baselines = json.loads((ROOT/lat.BASELINES_REL).read_text(encoding='utf-8'))
    assert set(lat.EXTERNAL_ARMS) <= set(baselines['arms']) and set(lat.MAIN_FAMILY) == {'configc', *lat.EXTERNAL_ARMS}


def _mutations():
    def set_(path, value):
        def f(t):
            target = t
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return f

    def extra_arm(t):
        t['arms'].append(copy.deepcopy(t['arms'][2]) | {'name': 'ants_so', 'family': 'descriptive'})

    def drop_identity(t):
        t['reference'] = []

    def extra_key(t):
        t['note'] = 'x'

    def reorder(t):
        t['arms'][1], t['arms'][2] = t['arms'][2], t['arms'][1]

    return [('schema', set_(['schema'], 'bayesreg.holdout.arm_table.v3')), ('pq', set_(['pq'], {'small_root': '/a', 'big_root': '/b'})), ('extra arm', extra_arm),
            ('identity', drop_identity), ('extra key', extra_key), ('order', reorder), ('main family', set_(['main_family'], ['ants_s', 'configc', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg'])),
            ('stage', set_(['stage_root'], '/home/rmcfz/seq_vp_tag_results/holdout_B_v1/stage')),
            ('holdout root', set_(['arms', 2, 'record_dir'], '/home/rmcfz/seq_vp_tag_results/holdout_B_v1/{arm}/{pair}')),
            ('D:', set_(['arms', 0, 'field', 'dir'], '/mnt/d/seq_vp_tag_holdout_B_v1/method/{pair}/output')),
            ('outside', set_(['arms', 3, 'field', 'root'], '/home/rmcfz/elsewhere/{arm}')),
            ('stop record', set_(['arms', 1, 'stop_record'], 'runs/holdout_B_v1/arm_stop_{arm}.json')),
            ('pin', set_(['frozen', 'method', 'settings_sha256'], '0'*64)),
            ('family', set_(['arms', 4, 'family'], 'descriptive')),
            ('per-arm form', set_(['arms', 2, 'cert_mode'], 'full')),
            ('per-arm classes', set_(['arms', 0, 'terminal', 'classes', 'output'], []))]


@pytest.mark.parametrize('name,mutate', _mutations(), ids=[m[0] for m in _mutations()])
def test_a_changed_table_is_refused(committed, name, mutate):
    table = json.loads(committed)
    mutate(table)
    with pytest.raises((lat.ArmTableError, holdout.HoldoutRefused)):
        lat.validate_lpba40_arm_table(table)


def test_duplicate_keys_are_refused(committed):
    raw = committed.replace(b'"pq": null,', b'"pq": null,\n "pq": null,', 1)
    with pytest.raises(holdout.HoldoutRefused):
        lat.load_bytes(raw)


def test_the_generator_never_overwrites(tmp_path):
    module = make_module()
    target = tmp_path/'t.json'
    assert module.main(['--out', str(target)]) == 0
    with pytest.raises(FileExistsError):
        module.main(['--out', str(target)])
    assert module.main(['--check']) == 0


# ---- LP-T0 revision (review 0/2/5: important 2, minor 4; the lead's ruling (1) on the configc key) ---------------------------------------------------
def _content_mutations():
    base = '/home/rmcfz/seq_vp_tag_results/lpba40_test_v1'

    def at(*path_value):
        *path, value = path_value

        def f(t):
            target = t
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value(target[path[-1]]) if callable(value) else value
        return f

    def move_wrapper_error(t):
        classes = t['arms'][2]['terminal']['classes']
        classes['wrapper_exception'].remove('wrapper_error')
        classes['method_failure'].append('wrapper_error')

    return [('terminal class moved', move_wrapper_error),
            ('terminal files', at('arms', 1, 'terminal', 'files', lambda v: v[1:])),
            ('terminal files order', at('arms', 2, 'terminal', 'files', lambda v: v[::-1])),
            ('status key', at('arms', 3, 'terminal', 'status_key', 'state')),
            ('complete files', at('arms', 0, 'complete_files', lambda v: [x for x in v if x != 'review/review.json'])),
            ('review final', at('arms', 0, 'review_record', 'final', [True, False])),
            ('review final 1 for true', at('arms', 0, 'review_record', 'final', [1])),           # diff review minor 1: True == 1 in Python, not in JSON
            ('decision certificate', at('arms', 0, 'decision_certificate', 'final', 'output/disp.npz')),
            ('export check dropped', at('arms', 2, 'export_check', None)),
            ('authorization tool', at('arms', 4, 'authorization', 'tool', 'hb')),
            ('authorization records', at('arms', 0, 'authorization', 'records', f'{base}/work/method/{{pair}}')),
            ('field key', at('arms', 1, 'field', 'file_key', 'output.disp.file')),
            ('record dir', at('arms', 3, 'record_dir', f'{base}/other/{{arm}}/{{pair}}')),
            ('field root', at('arms', 5, 'field', 'root', f'{base}/{{arm}}')),
            ('method field dir', at('arms', 0, 'field', 'dir', f'{base}/method/{{pair}}/output')),
            ('stop record', at('arms', 2, 'stop_record', 'runs/lpba40_test_v1/stop_{arm}.json')),
            ('comment', at('comment', 'another comment'))]


@pytest.mark.parametrize('name,mutate', _content_mutations(), ids=[m[0] for m in _content_mutations()])
def test_a_table_of_the_right_form_with_other_content_is_refused(committed, name, mutate):
    """Review important 2: each change keeps the form (_validate_form passes, so a form check alone lets it through) and is refused by the content check."""
    table = json.loads(committed)
    mutate(table)
    assert lat._validate_form(table) is table
    with pytest.raises(lat.ArmTableError, match='content'):
        lat.validate_lpba40_arm_table(table)


def test_the_content_check_reads_the_pinned_holdout_table(committed, tmp_path):
    raw = (ROOT/lat.HOLDOUT_TABLE_REL).read_bytes()
    (tmp_path/'configs'/'frozen').mkdir(parents=True)
    (tmp_path/lat.HOLDOUT_TABLE_REL).write_bytes(raw.replace(b'"output_store.disp.file"', b'"output_store.disp.path"', 1))
    with pytest.raises(lat.ArmTableError, match='not the pinned'):
        lat.validate_lpba40_arm_table(json.loads(committed), repo=tmp_path)
    (tmp_path/lat.HOLDOUT_TABLE_REL).write_bytes(raw)
    assert lat.validate_lpba40_arm_table(json.loads(committed), repo=tmp_path)['schema'] == lat.SCHEMA


@pytest.mark.parametrize('key', ['family', 'stop_record', 'name'])
def test_an_arm_without_a_key_is_an_arm_table_error(committed, key):
    """Review minor 4: not a KeyError."""
    table = json.loads(committed)
    del table['arms'][1][key]
    with pytest.raises(lat.ArmTableError, match='exactly the keys'):
        lat.validate_lpba40_arm_table(table)


def test_the_configc_field_root_holds_the_unmoved_outputs_and_their_resume(committed):
    """The lead's ruling (1) of 2026-11-19 ~1:01 PM CST: the key stays output_store.disp.file; an unmoved configuration C cell writes it as work/configc/<pair>/output/disp.npz,
    a resumed one as work/configc/<pair>.resume/output/disp.npz (hb_configc's .resume work directory); both lie under the arm's field root as the evaluation entry judges it."""
    arm = next(a for a in json.loads(committed)['arms'] if a['name'] == 'configc')
    assert arm['field']['file_key'] == 'output_store.disp.file' and arm['field']['sha256_key'] == 'output_store.disp.sha256'
    root = holdout.template_root(arm['field']['root'].replace('{arm}', arm['name']))
    for cell in ('LPBA40_S03__LPBA40_S04', 'LPBA40_S03__LPBA40_S04.resume'):
        path = f'{lat.LPBA_ROOT}/work/configc/{cell}/output/disp.npz'
        assert holdout._inside(path, root) and Path(path).name == 'disp.npz' and not holdout.forbidden_path(path), path
    assert 'xb_drive.py:402' in json.loads(committed)['comment'] and '不换键' in json.loads(committed)['comment']
