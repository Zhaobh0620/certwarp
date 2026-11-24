"""HE (runs/20261106-he/任务书.md): the rules, records and checks of experiments/he/{he_core,firewall_c,run_hx}.py, without the frozen PYTHONPATH and without a solve (synthetic files only).

- the frozen values are the code's constants; configuration C's options are run_queue.sh's;
- the decision, every branch; the fuse; the mechanical rule of the third certificate; certificate documents;
- the output files (C16 round trip), written once and only with a passing certificate of the same array;
- the code identity: PYTHONPATH, the code checkout's top level, the frozen tree (each with counterexamples);
- the derived pair of fallback one under a guard of the original's members; the gate (1) and its counterexamples; the candidates of D75; the launchers' start strings.
HE2 (runs/20261107-he2/任务书.md): the frozen values of Hx24-RF05-F1200-v4-C3 (v4, C, the units); the segmenter's anchor (its four values word for word; a changed setting, registry, byte or model id;
the reader's records item by item); the D86 statuses of the decision; OOMPolicy and the unit probe in the nested unit's string and in a real shell; the probe on a made-up cgroup and systemd; the third
certificate's fallback in a unit of its own (H3).
"""
from dataclasses import asdict, fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from bayesreg.io import pair as pair_module
from bayesreg.io.pair import load_pair
from jd_curl.three_d.feasibility import FeasibilityResult, PrescreenCertificateConfig
from jd_curl.three_d.grid import make_grid_3d

from ..io_seg._helpers import DEV, allowlist_document, source_sha, write_json
from ..io_seg.test_pair import GuardedNpz
from ..solve.test_d0_a4_tools import MI

ROOT = Path(__file__).resolve().parents[3]
for _d in ('he', 'fl', 'ef', 'mc0'):
    sys.path.insert(0, str(ROOT/'experiments'/_d))

import firewall_c as fc  # noqa: E402
import he_core as he  # noqa: E402
import run_hx  # noqa: E402

FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
needs_frozen_tree = pytest.mark.skipif(not (FROZEN/'jd_curl').is_dir(), reason='the frozen worktree is not on this machine')


# ---- the frozen values ---------------------------------------------------------------------------------------------------------------------------
def test_the_frozen_values_are_the_codes_constants():
    import fd2_core
    import fl_core
    import fx_core
    import mc0_core as core
    import mc0p_core as mc
    import repair
    import run_mc0
    import fast_channels as fast
    s = he.frozen_settings()
    pre = s['t88']['pre']
    assert pre['iterations'] == list(fl_core.COARSE_ITERATIONS)+[fx_core.NATIVE] == [50, 50, 100, 600] and pre['seconds'] == fl_core.RUN_CAP_SECONDS == 900.
    assert pre['sigma_field'] == fx_core.SPEC['sigma_field'] == fd2_core.ARMS['Hx24']['sigma_field'] == .5
    assert {k: v for k, v in pre.items() if k not in ('iterations', 'sigma_field', 'seconds')} == {k: list(v) if isinstance(v, tuple) else v for k, v in core.PRE_C.items() if k not in ('iterations', 'sigma_field', 'seconds')}
    spec = fx_core.SPEC
    assert (s['channels']['k'], s['channels']['channels'], s['channels']['weight'], s['channels']['D_factor']) == (spec['k'], spec['channels'], spec['weight'], spec['D_factor']) == (35, 'onehot', 'Q3h', 24.)
    assert fd2_core.ARMS['Hx24']['D_factor'] == 24. and fd2_core.ARMS['Hx24']['weight'] == 'Q3h' and s['channels']['sigma'] == fast.SIGMA == .75 and s['t88']['taper'] == core.TAPER == 4
    assert s['repair'] == dict({f.name: f.default for f in fields(repair.RepairConfig) if f.name != 'eps'}, eps=fl_core.REPAIR_EPS, wall_seconds=fx_core.FUSE_SECONDS)
    assert asdict(he.certificate_config(s)) == asdict(run_mc0.certificate_config())
    assert s['segmenter'] == he.CONTRACT_SEGMENTER and s['segmenter'] is not he.CONTRACT_SEGMENTER and s['channels']['colour_table_sha256'] == core.COLOUR_TABLE_SHA256      # HE2: v4, the anchor (C22.9-4 (xv))
    assert (s['segmenter']['segmenter_id'], s['segmenter']['policy']) != (mc.SEGMENTER_ID, mc.POLICY) and mc.SEGMENTER_ID == 'unet35_v1_ens'                     # the reader's own constant is v1's: not the anchor
    # HB (C22 卷首修订 8, D81 = option II; C22.H0-3): the method Hx24-RF05-F1200-v4, tissue_prior off, review_certificate_in a frozen setting; HE2's settings (-C3, moment3) a record whose sha256 is HE2's
    assert s['name'] == 'Hx24-RF05-F1200-v4' == he.NAME and 'revision 8' in s['contract'] and he.METHOD.startswith('A_H Hx24-RF05-F1200-v4 ') and s['tissue_prior'] == 'off' == he.TISSUE_PRIOR
    assert s['review_certificate_in'] == he.REVIEW_CERTIFICATE_IN and 'review/review.json' in s['review_certificate_in'] and 'C22.H1-5' in s['review_certificate_in']
    assert he.settings_sha256(he.he2_settings()) == he.HE2_SETTINGS_SHA256 == '76a9e563ce86e9cb01e6487759949de66f479a8cd35787c55dc42d9a8d7453e5' and not he.is_frozen(he.he2_settings())
    assert sorted(k for k in set(s) | set(he.he2_settings()) if s.get(k) != he.he2_settings().get(k)) == ['contract', 'name', 'review_certificate_in', 'tissue_prior']
    assert s['device_limit_gib']*2**30 == mc.VRAM_LIMIT_BYTES and s['children'] == dict(t88_timeout_seconds=run_mc0.RUN_RUNTIME_MAX, repair_timeout_seconds=fx_core.REPAIR_RUNTIME_SECONDS)
    assert s['fuse_rule'] == 'c-ii' and s['fallbacks'] == ['configC', 'identity'] and he.is_frozen(s) and len(he.settings_sha256(s)) == 64
    u = s['units']
    assert u['supervisor'] == he.SUPERVISOR_UNIT and u['supervisor']['oom_policy'] == 'continue' and u['supervisor']['memory_max'] == '12G' and u['evaluation'] == he.EVALUATION_UNIT
    assert u['frozen_c'] == dict(resources=list(fc.UNIT_RESOURCES), oom_policy='--property=OOMPolicy=continue', probe=['unit_probe_start.json', 'unit_probe_end.json'])
    assert u['gate_frozen_run_gib'] == 10. and u['gate_other_gib'] == 7. and u['third_certificate_fallback'] == 'unit'
    changed = he.frozen_settings()
    changed['t88']['pre']['iterations'] = [2, 2, 2, 4]
    assert not he.is_frozen(changed) and he.settings_sha256(changed) != he.settings_sha256(s)


# ---- the segmenter's anchor (C22.9-4 (xv)) -------------------------------------------------------------------------------------------------------------
def test_the_anchor_is_the_contracts_four_values_word_for_word():
    """docs/bayesreg/01_合同.md C22.9-4 (xv) "锚点" (C22.1-2 修订 4, D80): the four values written here, not taken from the code."""
    assert he.CONTRACT_SEGMENTER == dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                                         registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1')
    source = (ROOT/'experiments/he/he_core.py').read_text()
    assert source.count("registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'") == 2              # the anchor and the settings, each written out (not one derived from the other)
    assert hashlib.sha256((ROOT/'configs/bayesreg/segmenters_hs_v1.json').read_bytes()).hexdigest() == he.CONTRACT_SEGMENTER['registry_sha256']
    problems, record = he.segmenter_start_problems(he.frozen_settings())
    assert problems == [] and record['registry_rel'] == 'configs/bayesreg/segmenters_hs_v1.json' and set(record['model_segmenter_ids'].values()) == {'unet35_v4_ens'}
    contract = (ROOT/'docs/bayesreg/01_合同.md').read_text()                                                                                  # HE2 review minor 13: the contract's own text
    xv = contract[contract.index('(xv) **换分割器之后入口要改的**'):contract.index('(xvi) **C 开着时入口要加的**')]
    for value in ('`unet35_v4_ens`', '`all_folds_mean`', '`configs/bayesreg/segmenters_hs_v1.json`', '`b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1`'):
        assert value in xv[xv.index('**锚点**'):xv.index('**启动时**')], value
        assert value.strip('`') in he.CONTRACT_SEGMENTER.values()


def fake_root(tmp_path, change=None):
    """A code root with the anchor's registry, changed by `change(bytes) -> bytes`."""
    root = tmp_path/'repo'
    path = root/he.CONTRACT_SEGMENTER['registry']
    path.parent.mkdir(parents=True)
    raw = (ROOT/he.CONTRACT_SEGMENTER['registry']).read_bytes()
    path.write_bytes(raw if change is None else change(raw))
    return root, path


def test_the_anchor_refuses_another_setting_another_registry_or_a_changed_byte(tmp_path, monkeypatch):
    s = he.frozen_settings()
    other_id = dict(s, segmenter=dict(s['segmenter'], segmenter_id='unet35_v1_ens'))
    assert any('anchor' in p for p in he.segmenter_start_problems(other_id)[0])
    derived = dict(s, segmenter={k: v for k, v in s['segmenter'].items() if k != 'registry_sha256'})
    assert he.segmenter_start_problems(derived)[0]
    root, path = fake_root(tmp_path)
    assert he.segmenter_start_problems(s, registry=path, code_root=root)[0] == []                               # the same bytes at the anchor's path of another root
    assert any('is not configs/bayesreg/segmenters_hs_v1.json' in p for p in he.segmenter_start_problems(s, registry=path)[0])     # the same bytes elsewhere: the path differs
    other = tmp_path/'other'
    root2, path2 = fake_root(other, lambda raw: raw.replace(b'"fold": 0', b'"fold": 0 '))                       # one byte more
    problems = he.segmenter_start_problems(s, registry=path2, code_root=root2)[0]
    assert len(problems) == 1 and 'sha256' in problems[0]
    v1 = he.segmenter_start_problems(s, registry=ROOT/'configs/bayesreg/segmenters.json')[0]
    assert any('sha256' in p for p in v1) and any('do not all name' in p for p in v1) and any('is not configs/bayesreg' in p for p in v1)
    third = tmp_path/'third'
    root3, path3 = fake_root(third, lambda raw: raw.replace(b'"segmenter_id": "unet35_v4_ens"', b'"segmenter_id": "unet35_v9_ens"', 1))
    sha3 = hashlib.sha256(path3.read_bytes()).hexdigest()
    monkeypatch.setitem(he.CONTRACT_SEGMENTER, 'registry_sha256', sha3)                                         # only the models' ids differ now
    problems = he.segmenter_start_problems(dict(s, segmenter=dict(s['segmenter'], registry_sha256=sha3)), registry=path3, code_root=root3)[0]
    assert len(problems) == 1 and 'do not all name unet35_v4_ens' in problems[0]
    assert any('No such file' in p or 'FileNotFoundError' in p for p in he.segmenter_start_problems(s, registry=tmp_path/'none.json', code_root=tmp_path)[0])


def sources_of(registry=None):
    registry = str(ROOT/he.CONTRACT_SEGMENTER['registry']) if registry is None else registry
    side = dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry_sha256=he.CONTRACT_SEGMENTER['registry_sha256'], registry_path=registry)
    return dict(sides=dict(fixed=dict(side), moving=dict(side)), segmenter=dict(segmenter_id='unet35_v4_ens', registry=registry))


@pytest.mark.parametrize('change', ['side_id', 'side_policy', 'side_sha', 'side_path', 'one_side', 'source_id', 'source_registry', 'relative_elsewhere'])
def test_the_readers_records_against_the_anchor(change, tmp_path, monkeypatch):
    assert he.channel_source_problems(sources_of()) == []
    s = sources_of()
    if change == 'side_id':
        s['sides']['moving']['segmenter_id'] = 'unet35_v1_ens'
    elif change == 'side_policy':
        s['sides']['fixed']['policy'] = 'cross_fit_other_folds'
    elif change == 'side_sha':
        s['sides']['fixed']['registry_sha256'] = '0'*64
    elif change == 'side_path':
        s['sides']['moving']['registry_path'] = str(ROOT/'configs/bayesreg/segmenters.json')
    elif change == 'one_side':
        del s['sides']['moving']
    elif change == 'source_id':
        s['segmenter']['segmenter_id'] = None
    elif change == 'source_registry':
        s['segmenter']['registry'] = '/elsewhere/configs/bayesreg/segmenters_hs_v1.json'
    else:
        s = sources_of('configs/bayesreg/segmenters_hs_v1.json')                                                # a relative path resolves against the working directory, not the code root
        monkeypatch.chdir(tmp_path)
    assert len(he.channel_source_problems(s)) >= 1


def test_the_reader_is_pointed_at_the_anchors_segmenter(tmp_path):
    ctx = run_hx.make_context(tmp_path, 'p', tmp_path/'out', paths=dict(run_hx.default_paths(tmp_path/'out')), check_identity=False)
    u = run_hx.unet_source(ctx)
    assert u.segmenter_id == 'unet35_v4_ens' and Path(u.registry) == ROOT/'configs/bayesreg/segmenters_hs_v1.json' and u.gate == he.SEGMENTER_GATE and 'D80' in u.gate['rule']
    assert Path(u.pred_root) == he.V4_SOURCES['pred_root'] == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred') and Path(u.models_root) == Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full')
    reference = json.loads((ROOT/'experiments/p2/reference_unet35_v4_ens.json').read_text())['channels']
    assert (str(u.pred_root), str(u.models_root), str(u.image_root), reference['registry_rel']) == (reference['pred_root'], reference['models_root'], reference['image_root'], he.CONTRACT_SEGMENTER['registry'])
    record = u.record()
    assert record['segmenter_id'] == 'unet35_v4_ens' and he.repo_relative(record['registry']) == he.CONTRACT_SEGMENTER['registry'] and record['a5pv2_gate'] == he.SEGMENTER_GATE
    paths = dict(ctx['paths'])
    paths.pop('registry')
    assert run_hx.registry_of(dict(ctx, paths=paths)) == ROOT/'configs/bayesreg/segmenters_hs_v1.json'


def test_configuration_c_is_run_queue_shs():
    text = (ROOT/'experiments/d0/run_queue.sh').read_text()
    pre = re.search(r'^PRE="([^"]+)"', text, re.M).group(1)
    block = re.search(r'^OPTS=\((.*?)\)\n', text, re.M | re.S).group(1)
    import shlex
    opts = tuple(a if a != '$PRE' else pre for a in shlex.split(block.replace('"$PRE"', '$PRE')))
    assert pre == fc.PRE and opts == fc.OPTS
    assert re.search(r'WATCHDOG=(\d+)', text).group(1) == str(fc.WATCHDOG_SECONDS) and '--kill-after=15s' in text and fc.KILL_GRACE == 15
    for item in fc.UNIT_RESOURCES:
        assert item.replace('1200%', '${QUOTA}%') in text or item in text
    assert '--setenv=NUMBA_NUM_THREADS=8' in text and fc.THREADS == 8
    assert fc.DERIVED_KEYS == MI.KEY_ORDER


def test_the_pinned_top_level_is_the_commits():
    done = subprocess.run(['git', 'ls-tree', '--name-only', he.TOP_LEVEL_COMMIT], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and set(done.stdout.split()) >= set(he.REPO_TOP_LEVEL)


# ---- the decision ---------------------------------------------------------------------------------------------------------------------------------
def test_the_t88_decision():
    good = dict(status='done', oracle=False, field=dict(file='field_0600.npz'))
    assert he.t88_decision(good) == ('go', None)
    kind, f = he.t88_decision(None, 'timeout')
    assert kind == 'fallback' and f['class'] == 'entry' and 'timeout' in f['reason']
    assert he.t88_decision(dict(good, status='provenance_rejected', reason='r')) == ('rejected', dict(stage='channels', reason='r'))
    assert he.t88_decision(dict(good, status='segmentation_qc', reason='q'))[1]['class'] == 'segmentation_qc'
    for status in ('time_cap', 'vram_stop', 'cuda_oom', 'nonfinite', 'scale_mismatch', 'not_predicted_channels', 'estimate_failed', 'tissue_data'):
        kind, f = he.t88_decision(dict(good, status=status))
        assert kind == 'fallback' and f['class'] == 'entry' and 'needs_lead_rejudgment' not in f
    assert he.t88_decision(dict(good, oracle=True))[1]['class'] == 'entry' and he.t88_decision(dict(good, field=None))[1]['class'] == 'entry'
    kind, f = he.t88_decision(dict(good, status='tissue_error', reason='OSError'))                                   # HE2, D86 (4)
    assert kind == 'fallback' and f['class'] == 'entry' and f['needs_lead_rejudgment'] is True
    assert he.t88_decision(dict(good, status='implementation_error', reason='g')) == ('rejected', dict(stage='tissue', kind='implementation_error', reason='g', stop_queue=True))     # D86 (2)
    assert he.t88_decision(dict(good, status='provenance_rejected', reason='a', rejection_stage='segmenter_anchor', rejection_kind='segmenter_anchor')) == ('rejected', dict(stage='segmenter_anchor', reason='a', kind='segmenter_anchor'))
    assert he.STOP_QUEUE_EXIT == 9


def test_the_repair_decision_and_the_fuse():
    passed = dict(passed=True, reason='certified', rounds=[dict(inner=dict(stop='inner_cap'))])
    assert he.repair_decision(passed, True) == ('hx', None)
    assert he.repair_decision(passed, 'hashes differ')[1]['class'] == 'repair_error'
    assert he.repair_decision(None, None, -9)[1]['class'] == 'repair_error'
    assert he.repair_decision(dict(error='ValueError: x', passed=False), None)[1]['class'] == 'repair_error'
    assert he.repair_decision(dict(passed=False, reason='wall'), None)[1]['class'] == 'repair_fuse'
    for reason in ('round_cap', 'empty_mask'):
        assert he.repair_decision(dict(passed=False, reason=reason), None)[1] == he.failure('repair_not_certified', reason)
    assert he.repair_decision(dict(status='provenance_rejected', passed=False, reason='admission: x'), None) == ('rejected', dict(stage='repair', reason='admission: x'))     # never a fallback
    walled = dict(passed, rounds=[dict(inner=dict(stop='wall'))])
    fuse = he.fuse_record(walled, he.HX)
    assert {k: v for k, v in fuse.items() if k != 'note'} == dict(wall_hit=True, certified_after_wall=True, fuse_rule='c-ii', line_status_if_c_ii='hx_certified', line_status_if_c_i='hx_fallback_not_run') and 'c-ii' in fuse['note']
    assert he.fuse_record(passed, he.HX)['line_status_if_c_i'] == 'hx_certified' and he.fuse_record(None, he.FALLBACK_ID)['wall_hit'] is False and he.fuse_record(passed, he.HX)['note'] is None
    assert he.fuse_record(dict(passed=False, reason='wall', rounds=[]), he.FALLBACK_C) == dict(wall_hit=True, certified_after_wall=False, fuse_rule='c-ii', line_status_if_c_ii=he.FALLBACK_C, line_status_if_c_i=he.FALLBACK_C, note=None)
    with pytest.raises(ValueError):
        he.failure('provenance', 'x')


def test_hit_wall_is_decide_fxs():
    import decide_fx
    for record in (dict(reason='wall'), dict(reason='certified', rounds=[dict(inner=dict(stop='wall'))]), dict(reason='certified', rounds=[dict(inner=dict(stop='inner_cap'))]), dict(reason='round_cap', rounds=[]), {}):
        assert he.hit_wall(record) == decide_fx.hit_wall(record)


def test_the_third_certificate_rule_and_its_order():
    assert he.next_fallback(he.HX) == 'configC' and he.next_fallback(he.FALLBACK_C) == 'identity' and he.next_fallback(he.FALLBACK_ID) is None and he.next_fallback(he.REJECTED) is None


# ---- certificates ----------------------------------------------------------------------------------------------------------------------------------
CONFIG = PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)


def result(**changes):
    base = dict(status='certified_feasible', min_jd_lower_bound=.3, min_margin_lower_bound=.1, sampled_min_jd=.4, n_cells_checked=210, n_uncertain=0, n_infeasible=0, n_boxes=210, max_depth=0, witness_cell=None,
                witness_reference_point=None, reason='all cells certified', state_hash='a'*64, grid_id='g', delta_hex=(0.).hex(), config=CONFIG)
    return FeasibilityResult(**dict(base, **changes))


def test_passes_and_the_certificate_document():
    assert he.passes(result()) and he.passes(dict(status='certified_feasible', min_jd_lower_bound=1e-9, n_infeasible=0, n_uncertain=0))
    for bad in (dict(min_jd_lower_bound=0.), dict(min_jd_lower_bound=None), dict(n_uncertain=1), dict(n_infeasible=1), dict(status='uncertain')):
        assert not he.passes(result(**bad))
    doc = he.certificate_document(result(witness_reference_point=(.5, .25, 0.)), CONFIG, process='p', array_sha256_of_u='a'*64)
    assert doc['passed'] is True and doc['state_hash_matches_array'] is True and doc['n_cells'] == 210 and doc['witness'] == dict(cell=None, reference_point=[.5, .25, 0.]) and doc['process'] == 'p'
    assert doc['config']['delta'] == (0.).hex() and doc['config']['tile_size'] == 65536 and doc['config']['config_class'] == 'PrescreenCertificateConfig' and 'config' not in json.dumps(asdict(CONFIG))
    json.dumps(doc)
    same, diffs = he.same_certificate(doc, dict(doc))
    assert same and not diffs
    other = he.certificate_document(result(min_jd_lower_bound=.15), CONFIG, process='q')
    same, diffs = he.same_certificate(doc, other)
    assert not same and set(diffs) == {'min_jd_lower_bound'}
    tiled = he.certificate_document(result(), PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=8192), process='q')
    assert he.same_certificate(doc, tiled)[1] == {'config.tile_size': (65536, 8192)}                  # certificate_version follows the configuration's class, not its tile (revision 2, review new minor 2)
    assert set(he.same_certificate(doc, he.certificate_document(result(grid_id='another'), CONFIG, process='q'))[1]) == {'grid_id'}          # revision 1, review minor 3
    versioned = dict(doc, config=dict(doc['config'], certificate_version='another-version'))
    assert set(he.same_certificate(doc, versioned)[1]) == {'config.certificate_version'} and doc['config']['certificate_version']


def test_the_recording_certify_keeps_every_result():
    calls = []

    def certify(u, grid, *, config):
        calls.append(config)
        return result(state_hash=he.array_sha256(u))
    rec = he.RecordingCertify(certify)
    a, b = np.zeros((2, 2, 2, 3)), np.ones((2, 2, 2, 3))
    rec(a, None, config=CONFIG)
    rec(b, None, config=CONFIG)
    assert rec.of(he.array_sha256(a)).state_hash == he.array_sha256(a) and rec.of('x') is None and len(rec.results) == 2 and calls == [CONFIG, CONFIG]


# ---- the output ----------------------------------------------------------------------------------------------------------------------------------
def field(grid, seed=1):
    u = np.random.default_rng(seed).normal(scale=.2, size=grid.shape+(3,))
    u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1] = 0, 0, 0, 0, 0, 0
    return u


@pytest.mark.parametrize('spacing', [(1., 1., 1.), (1., 1.25, .75)])
def test_the_output_files_and_their_round_trip(tmp_path, spacing):
    from bayesreg.eval.fields import read_disp, roundtrip_check
    from bayesreg.solve.export import displacement_to_voxels
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=spacing[0], hy=spacing[1], hz=spacing[2], unit='mm')
    u = field(grid)
    cert = he.certificate_document(result(state_hash=he.array_sha256(u)), CONFIG, process='p')
    files = he.write_output(tmp_path/'output', u, grid, cert, dict(line_status=he.HX, pair='p'))
    back, line, c = he.read_output(tmp_path/'output', grid)
    assert back.tobytes() == u.tobytes() and line['line_status'] == he.HX and line['output'] == files and c == cert
    d, record = read_disp(tmp_path/'output/disp.npz', grid.shape)
    assert d.tobytes() == displacement_to_voxels(u, grid).tobytes() and roundtrip_check(u, d, spacing)['passed'] is True
    with np.load(tmp_path/'output/disp.npz') as z:
        meta = json.loads(str(z['meta']))
    assert meta['u_state_hash'] == files['final']['state_hash'] and meta['layout'] == '(3, nz, ny, nx)' and meta['exact_inverse'] is (spacing == (1., 1., 1.))
    with pytest.raises(FileExistsError):
        he.write_output(tmp_path/'output', u, grid, cert, dict(line_status=he.HX))


def test_no_output_without_a_passing_certificate_of_the_same_array(tmp_path):
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    u = field(grid)
    for k, cert in enumerate((he.certificate_document(result(state_hash='b'*64), CONFIG, process='p'), he.certificate_document(result(state_hash=he.array_sha256(u), n_uncertain=1), CONFIG, process='p'))):
        with pytest.raises(ValueError):
            he.write_output(tmp_path/f'o{k}', u, grid, cert, dict(line_status=he.HX))
        assert not (tmp_path/f'o{k}').exists()
    v = u.copy()
    v[0, 1, 1, 0] = .1
    with pytest.raises(ValueError, match='faces'):
        he.write_output(tmp_path/'faces', v, grid, he.certificate_document(result(state_hash=he.array_sha256(v)), CONFIG, process='p'), dict(line_status=he.HX))
    he.write_output(tmp_path/'ok', u, grid, he.certificate_document(result(state_hash=he.array_sha256(u)), CONFIG, process='p'), dict(line_status=he.HX))
    with np.load(tmp_path/'ok/final.npz') as z:
        np.savez(tmp_path/'ok/final.npz', u=np.array(z['u'])*1.5, state_hash=z['state_hash'], grid=z['grid'], status=z['status'])
    with pytest.raises(ValueError, match='state_hash'):
        he.read_output(tmp_path/'ok')


def test_records_are_written_once(tmp_path):
    he.write_json(tmp_path/'a.json', dict(x=1))
    with pytest.raises(FileExistsError):
        he.write_json(tmp_path/'a.json', dict(x=2))
    assert json.loads((tmp_path/'a.json').read_text()) == dict(x=1)
    assert run_hx.numbered(tmp_path, 'identity_eval') == 'identity_eval_0.json'
    (tmp_path/'identity_eval_0.json').write_text('{}')
    (tmp_path/'identity_eval_end_0.json').write_text('{}')
    assert run_hx.numbered(tmp_path, 'identity_eval') == 'identity_eval_1.json' and run_hx.numbered(tmp_path, 'identity_eval_end') == 'identity_eval_end_1.json'


# ---- the code identity --------------------------------------------------------------------------------------------------------------------------
def git_tree(path, files):
    path.mkdir(parents=True)
    for rel, text in files.items():
        (path/rel).parent.mkdir(parents=True, exist_ok=True)
        (path/rel).write_text(text)
    run = lambda *a: subprocess.run(['git', '-C', str(path), '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *a], check=True, capture_output=True)     # noqa: E731
    run('init', '-q')
    run('add', '-A')
    run('commit', '-q', '-m', 'x')
    return run


def test_the_code_checkouts_top_level(tmp_path):
    base = {'bayesreg/__init__.py': '', 'experiments/he/run_hx.py': '', 'README.md': ''}
    root = tmp_path/'repo'
    run = git_tree(root, base)
    record, problems = he.repo_identity(root)
    assert problems == [] and record['tracked_clean'] is True and record['added_top_level'] == []
    (root/'refs').mkdir()                                                     # a directory without __init__: a namespace portion, which cannot shadow
    (root/'notes.txt').write_text('x')
    assert he.repo_identity(root)[1] == [] and he.repo_identity(root)[0]['untracked_top_level'] == ['notes.txt', 'refs']
    for name, make in (('numba.py', lambda p: p.write_text('')), ('scipy', lambda p: (p.mkdir(), (p/'__init__.py').write_text(''))), ('torch.cpython-310-x86_64-linux-gnu.so', lambda p: p.write_bytes(b''))):
        make(root/name)
        problems = he.repo_identity(root)[1]
        assert len(problems) == 1 and name in problems[0], (name, problems)
        (shutil.rmtree if (root/name).is_dir() else os.remove)(root/name)
    os.remove(root/'notes.txt')
    (root/'surface_distance').mkdir()
    (root/'surface_distance/__init__.py').write_text('')
    run('add', '-A')
    run('commit', '-q', '-m', 'a new top-level package')
    problems = he.repo_identity(root)[1]
    assert len(problems) == 1 and "['surface_distance']" in problems[0] and 'existing ones' in problems[0]
    (root/'README.md').write_text('changed')
    assert he.repo_identity(root)[0]['tracked_clean'] is False


def test_pythonpath_is_exactly_the_frozen_tree_then_the_code(tmp_path):
    frozen, code = tmp_path/'f', tmp_path/'c'
    assert he.pythonpath_problems(f'{frozen}:{code}', code, frozen) == []
    for bad in (f'{code}:{frozen}', f'{frozen}', f'{frozen}:{code}:{tmp_path/"x"}', '', None):
        assert he.pythonpath_problems(bad, code, frozen)


@needs_frozen_tree
def test_the_frozen_tree_passes_and_its_counterexamples_fail(tmp_path):
    record, problems = he.frozen_identity(FROZEN, cwd=ROOT)
    assert problems == [] and record['same'] is True and record['n_files'] >= 125 and record['untracked_under_jd_curl'] == [] and record['top_level_beyond_commit'] == []
    copy = tmp_path/'frozen'
    shutil.copytree(FROZEN/'jd_curl', copy/'jd_curl', ignore=shutil.ignore_patterns('__pycache__'))
    for rel in ('data/three_d', '.git'):
        (copy/rel).mkdir(parents=True) if rel == 'data/three_d' else (copy/rel).write_text('gitdir: x')
    (copy/'jd_curl/__pycache__').mkdir()
    (copy/'jd_curl/__pycache__/x.cpython-310.pyc').write_bytes(b'')
    assert he.frozen_identity(copy, cwd=ROOT)[1] == []
    for rel, kind in (('numba', 'dir'), ('jd_curl/three_d/feasibility.cpython-310-x86_64-linux-gnu.so', 'file'), ('jd_curl/three_d/notes.txt', 'file')):
        target = copy/rel
        target.mkdir() if kind == 'dir' else target.write_bytes(b'x')
        problems = he.frozen_identity(copy, cwd=ROOT)[1]
        assert problems and any(Path(rel).name in p for p in problems), (rel, problems)
        (shutil.rmtree if kind == 'dir' else os.remove)(target)
    (copy/'jd_curl/three_d/grid.py').write_text('# changed\n')
    assert he.frozen_identity(copy, cwd=ROOT)[1]


def test_the_code_paths_must_be_committed_and_clean(tmp_path):
    """Revision 1, review important 2: in the paths of the code and the configuration (bayesreg, experiments, configs, splits, jd_curl, scripts) no changed, untracked or ignored file (__pycache__ aside) when the
    process starts from the command line (require_clean); under experiments/ (first on sys.path) no untracked or ignored module, ever.  docs/ is not checked."""
    root = tmp_path/'repo'
    git_tree(root, {'.gitignore': '__pycache__/\n*.pyc\n*.log\n', 'bayesreg/__init__.py': '', 'experiments/he/run_hx.py': '', 'experiments/fl/fl_core.py': '', 'configs/bayesreg/dev_subjects.json': '{}', 'docs/a.md': 'x'})
    assert he.repo_identity(root)[1] == []
    (root/'docs/a.md').write_text('changed')
    (root/'docs/new.md').write_text('new')
    (root/'experiments/he/__pycache__').mkdir()
    (root/'experiments/he/__pycache__/run_hx.cpython-310.pyc').write_bytes(b'')
    record, problems = he.repo_identity(root)
    assert problems == [] and record['code_paths_clean'] is True and record['tracked_clean'] is False                 # docs and __pycache__ do not count
    cases = (('experiments/he/numpy.py', lambda p: p.write_text(''), True, True),                                       # an untracked module first on sys.path
             ('experiments/fl/stale.pyc', lambda p: p.write_bytes(b''), True, True),                                    # an ignored, importable sourceless module
             ('configs/bayesreg/dev_subjects.json', lambda p: p.write_text('{"changed": true}'), True, False),            # the allowlist changed and not committed
             ('experiments/he/notes.log', lambda p: p.write_text('x'), True, False),                                     # an ignored file in a code path
             ('splits/new_split.txt', lambda p: (p.parent.mkdir(), p.write_text('x')), True, False))                    # an untracked file in a configuration path
    for rel, make, refused_clean, refused_always in cases:
        target = root/rel
        original = target.read_bytes() if target.exists() else None
        make(target)
        clean_problems = he.repo_identity(root)[1]
        loose_problems = he.repo_identity(root, require_clean=False)[1]
        assert bool(clean_problems) is refused_clean and bool(loose_problems) is refused_always, (rel, clean_problems, loose_problems)
        assert any(Path(rel).name in p for p in clean_problems), (rel, clean_problems)
        if original is None:
            os.remove(target)
        else:
            target.write_bytes(original)
    assert he.repo_identity(root)[1] == []


@needs_frozen_tree
def test_identity_of_this_checkout_with_the_units_pythonpath():
    """On a checkout whose code paths are committed and clean, the identity of the units' layout passes; on another the check is about that checkout's state, so it is skipped and says why (not loosened)."""
    record, problems = he.identity(ROOT, frozen=FROZEN, pythonpath=f'{FROZEN}:{ROOT}')
    assert record['repo']['code_root'] == str(ROOT) and he.identity(ROOT, frozen=FROZEN, pythonpath=str(ROOT))[1]
    if not record['repo']['code_paths_clean']:
        pytest.skip(f"this checkout has uncommitted code or configuration ({record['repo']['dirty_paths'][:5]}): the identity refuses it, as it should; run on a clean checkout")
    assert problems == []


# ---- fallback one: the derived pair ------------------------------------------------------------------------------------------------------------
SHAPE = (6, 7, 8)


def synthetic_pair(tmp_path, *, initial_u=True):
    grid = (SHAPE[2], SHAPE[1], SHAPE[0])
    rng = np.random.default_rng(1)
    d = tmp_path/'stage'/f'{DEV[0]}__{DEV[1]}'
    d.mkdir(parents=True)
    arrays = dict(fixed=rng.uniform(size=SHAPE), fixed_seg4=rng.integers(0, 5, SHAPE).astype(float), fixed_seg35=rng.integers(0, 36, SHAPE).astype(float), moving=rng.uniform(size=SHAPE).astype(np.float32),
                  moving_seg4=rng.integers(0, 5, SHAPE).astype(float), moving_seg35=rng.integers(0, 36, SHAPE).astype(float))
    if initial_u:
        arrays['initial_u'] = np.zeros(SHAPE+(3,))
    np.savez(d/'pair.npz', **arrays)
    from jd_curl.three_d.nifti import file_sha256
    manifest = dict(pair=d.name, fixed_subject=DEV[0], moving_subject=DEV[1], shape_zyx=list(SHAPE), spacing_xyz=[1., 1.25, .75], origin_xyz=[-2., 3., .5], unit='mm',
                    sources=dict(fixed=dict(sha256=source_sha(DEV[0])), moving=dict(sha256=source_sha(DEV[1]))), pair_sha256=file_sha256(d/'pair.npz'),
                    baseline_label_metrics=dict(seg4=dict(macro_dice=.77)), weights=dict(lam=.123), labels_policy='x')
    write_json(d/'manifest.json', manifest)
    allowlist = write_json(tmp_path/'dev_subjects.json', allowlist_document())
    return d, arrays, allowlist, grid


@pytest.fixture
def guarded(monkeypatch):
    opened = []

    def open_guarded(path):
        opened.append(GuardedNpz(np.load(path, allow_pickle=False)))
        return opened[-1]
    monkeypatch.setattr(pair_module, '_open_npz', open_guarded)
    return opened


@pytest.mark.parametrize('initial_u', [True, False])
def test_the_derived_pair_reads_only_the_images_and_keeps_them_byte_for_byte(tmp_path, guarded, initial_u):
    d, arrays, allowlist, _ = synthetic_pair(tmp_path, initial_u=initial_u)
    pin = load_pair(d/'pair.npz', allowlist=allowlist)
    record = fc.derive_pair(pin, tmp_path/'derived', pair=d.name)
    assert all(g.items_read in (['fixed', 'moving', 'initial_u'], ['fixed', 'moving']) for g in guarded) and len(guarded) == 2      # load_pair, then derive_pair: one key at a time, the whitelist only
    assert record['source_keys_read'] == (['fixed', 'moving', 'initial_u'] if initial_u else ['fixed', 'moving']) and record['initial_u_present'] is initial_u
    with np.load(tmp_path/'derived/pair.npz') as z:
        assert list(z.files) == list(fc.DERIVED_KEYS)
        assert z['moving'].dtype == np.float32 and z['moving'].tobytes() == arrays['moving'].tobytes() and z['fixed'].tobytes() == arrays['fixed'].tobytes()
        assert all(not np.any(z[k]) and z[k].shape == SHAPE for k in fc.SEG_KEYS) and not np.any(z['initial_u']) and z['initial_u'].shape == SHAPE+(3,)
    manifest = json.loads((tmp_path/'derived/manifest.json').read_text())
    assert manifest['baseline_label_metrics']['seg4']['macro_dice'] is None and manifest['baseline_label_metrics']['seg35']['macro_dice'] is None      # not the original's label-derived record
    assert manifest['weights'] != dict(lam=.123) and manifest['weights'] == fc.frozen_weights(pin.grid) and manifest['pair'] == d.name and 'labels_policy' in manifest
    assert {k: manifest[k] for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz', 'unit')} == pin.grid_metadata and manifest['pair_sha256'] == record['pair_npz_sha256']
    gate, problems = fc.firewall_check(pin, tmp_path/'derived', pair=d.name)
    assert problems == [] and gate['passed'] is True and all(gate['same'].values()) and all(gate['label_members_zero'].values())


def test_the_gate_fails_on_any_change_of_the_derived_pair(tmp_path):
    d, arrays, allowlist, _ = synthetic_pair(tmp_path)
    pin = load_pair(d/'pair.npz', allowlist=allowlist)
    fc.derive_pair(pin, tmp_path/'derived', pair=d.name)
    good = (tmp_path/'derived/pair.npz').read_bytes()
    with np.load(tmp_path/'derived/pair.npz') as z:
        base = {k: np.array(z[k]) for k in z.files}

    def write(**changes):
        a = dict(base, **changes)
        np.savez(tmp_path/'derived/pair.npz', **{k: a[k] for k in a if a[k] is not None})
        return fc.firewall_check(pin, tmp_path/'derived', pair=d.name)[1]
    moved = base['fixed'].copy()
    moved[1, 1, 1] = np.nextafter(moved[1, 1, 1], 2.)
    assert any('fixed' in p for p in write(fixed=moved))
    assert any('moving' in p for p in write(moving=base['moving'].astype(np.float64)))                 # the same values in another dtype are not the member
    assert any('fixed_seg35' in p for p in write(fixed_seg35=np.ones(SHAPE)))
    assert any('keys' in p for p in write(fixed_seg4=None))
    changed = write(extra=np.zeros(1))
    assert any('keys' in p for p in changed) and 'the derived manifest does not name the derived pair.npz' in changed      # any rewrite also leaves the manifest's pair_sha256 behind
    (tmp_path/'derived/pair.npz').write_bytes(good)
    assert fc.firewall_check(pin, tmp_path/'derived', pair=d.name)[1] == []
    manifest = json.loads((tmp_path/'derived/manifest.json').read_text())
    write_json(tmp_path/'derived/manifest.json', dict(manifest, spacing_xyz=[1., 1., 1.]))
    assert any('grid metadata' in p for p in fc.firewall_check(pin, tmp_path/'derived', pair=d.name)[1])
    write_json(tmp_path/'derived/manifest.json', manifest)
    assert fc.firewall_check(pin, tmp_path/'derived', pair=d.name)[1] == []
    run = tmp_path/'run/inputs'/d.name
    run.mkdir(parents=True)
    shutil.copy(d/'pair.npz', run/'pair.npz')                                                         # a prepared run with the ORIGINAL pair: refused
    shutil.copy(tmp_path/'derived/manifest.json', run/'manifest.json')
    assert any('prepared run inputs' in p for p in fc.firewall_check(pin, tmp_path/'derived', tmp_path/'run', pair=d.name)[1])


def test_the_frozen_weights_and_the_slot():
    grid = make_grid_3d(nx=160, ny=224, nz=192, hx=1., hy=1., hz=1., unit='mm')
    L = (159.*223.*191.)**(1/3)
    assert fc.frozen_weights(grid) == dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001)
    text = (FROZEN/'scripts/prepare_stage_3d_holdout.py').read_text() if FROZEN.is_dir() else None
    if text is not None:
        assert "L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)" in text and 'return dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001)' in text
        assert f"PAIR = '{fc.FROZEN_DEV_PAIR}'" in (FROZEN/'scripts/run_stage_3d_e17_pilot.py').read_text()
        assert f"('{fc.SLOT_STAGE}', 'data/three_d/{fc.SLOT_STAGE}')" in (FROZEN/'scripts/run_stage_3d_e48_pg.py').read_text()
    assert fc.slot_name(fc.FROZEN_DEV_PAIR) == fc.FROZEN_DEV_PAIR+'__he_slot' and fc.slot_name('A__B') == 'A__B'


# ---- fallback one: the candidates and the launchers ---------------------------------------------------------------------------------------------
PROOF = dict(status='certified_feasible', min_jd_lower_bound=.4, n_infeasible=0, n_uncertain=0, state_hash='s')


def fake_run(tmp_path, *, final=True, seed=True, grid=(5, 6, 7), proof=PROOF):
    run = tmp_path/'run'
    task = run/'pg_weak__four__90s'
    task.mkdir(parents=True)
    write_json(run/'protocol.json', dict(task=dict(name=task.name)))
    events = []
    if seed:
        np.savez(task/'level_3_level_prepared.npz', u=np.zeros(grid+(3,)))
        events.append(dict(event='level_prepared', level=3, path='level_3_level_prepared.npz', sha256=fc.sha_file(task/'level_3_level_prepared.npz'), transfer_certificate=proof))
    events.append(dict(event='level_prepared', level=2, path='level_2_level_prepared.npz', sha256='x', transfer_certificate=PROOF))
    write_json(task/'levels.json', events)
    if final:
        np.savez(task/'final_native.npz', u=np.zeros(grid+(3,)), warped=np.zeros(grid))
        write_json(task/'final_native.json', dict(path='final_native.npz', sha256=fc.sha_file(task/'final_native.npz'), role='formal_native_outer_commit', certificate=dict(status='certified_feasible')))
    return run


def test_the_candidates_of_d75(tmp_path):
    run = fake_run(tmp_path)
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    for cls in ('ok', 'handover_only'):
        found = fc.candidates(run, {'class': cls})
        assert [name for name, _, _ in found] == ['final_native.npz', 'level_3_level_prepared.npz']
        assert all(fc.load_candidate(path, record, grid).shape == (5, 6, 7, 3) for _, path, record in found)
    for cls in ('method_failure', 'infrastructure', 'unsure'):
        assert fc.candidates(run, {'class': cls}) == []
    (run/'pg_weak__four__90s/final_native.json').write_text(json.dumps(dict(sha256='0'*64)))
    name, path, record = fc.candidates(run, {'class': 'ok'})[0]
    with pytest.raises(ValueError, match='sha256'):
        fc.load_candidate(path, record, grid)
    with pytest.raises(ValueError, match='grid'):
        fc.load_candidate(*fc.candidates(run, {'class': 'ok'})[1][1:], make_grid_3d(nx=8, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm'))
    only_seed = fake_run(tmp_path/'b', final=False)
    assert [n for n, _, _ in fc.candidates(only_seed, {'class': 'handover_only'})] == ['level_3_level_prepared.npz']
    assert fc.candidates(fake_run(tmp_path/'c', final=False, seed=False), {'class': 'ok'}) == []
    for k, bad in enumerate((dict(PROOF, n_uncertain=1), dict(PROOF, min_jd_lower_bound=0.), dict(PROOF, status='certified_infeasible'), dict(state_hash='s'))):     # revision 1, review minor 14: the transfer certificate must pass
        assert fc.candidates(fake_run(tmp_path/f'd{k}', final=False, proof=bad), {'class': 'handover_only'}) == []


def test_a_field_that_cannot_be_an_output():
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    u = field(grid)
    assert he.field_problem(u, grid) is None
    face = u.copy()
    face[:, :, -1, 1] = .1
    nan = u.copy()
    nan[2, 2, 2, 0] = np.nan
    for bad, word in ((face, 'faces'), (nan, 'non-finite'), (u.astype(np.float32), 'float64'), (u[:-1], 'grid')):
        assert word in he.field_problem(bad, grid)


def test_the_launchers_start_strings(tmp_path):
    named = tmp_path/'pytest-x'                                                                       # a directory that names the test runner, built explicitly: tmp_path itself need not (--basetemp)
    assert 'pytest' in str(named)
    with pytest.raises(ValueError, match='forbidden'):                                                # a start string that names the test runner is refused
        fc.UnitLauncher(named).argv(named/'run', 'x')
    clean = Path('/home/rmcfz/seq_vp_tag_results')                                                    # only the strings are built; nothing is read or written there
    for launcher, run in ((fc.UnitLauncher(named), clean/'run'), (fc.UnitLauncher(clean), named/'run'), (fc.DirectLauncher(clean), named/'run')):
        with pytest.raises(ValueError, match='forbidden'):                                            # the word in the cache path alone, or in the run path alone, is enough
            launcher.argv(run, 'x')
    tmp_path = clean
    run = tmp_path/'he_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1/fallback_configc/run'
    unit = fc.UnitLauncher(tmp_path/'cache').argv(run, '0438-0439')
    assert unit[:2] == ['systemd-run', '--user'] and '--unit=tag-he-c-0438-0439' in unit and '--wait' in unit and '--collect' in unit and f'--working-directory={run}/snapshot' in unit
    assert all(r in unit for r in fc.UNIT_RESOURCES) and '--setenv=NUMBA_NUM_THREADS=8' in unit and f'--setenv=JD_CURL_JIT_CACHE={tmp_path}/cache/numba_jit' in unit and unit[-3:-1] == ['/bin/bash', '-c']
    assert unit[unit.index(fc.UNIT_RESOURCES[-1])+1] == '--property=OOMPolicy=continue' == fc.UNIT_OOM_POLICY and len([a for a in unit if 'OOMPolicy' in a]) == 1      # HE2 (H2), explicitly
    inner = unit[-1]
    assert 'timeout --signal=TERM --kill-after=15s 300s' in inner and f'{run}/snapshot/scripts/run_stage_3d_e48_pg.py run {run}' in inner and 'exit_code' in inner and 'supervisor_started_utc.txt' in inner
    assert not any('seq-vp-' in a for a in unit) and fc.DirectLauncher(tmp_path).argv(run, 'x') == ['/bin/bash', '-c', inner]
    plain = fc.inner_command(run, fc.frozen_run_argv(run))                                            # D0's inner command, unchanged between the two probes (HE2, C22.B4-6 (d))
    assert inner == f"{fc.probe_command(run, 'start', 'unit_probe_start.json')}; {plain[:-len('; exit $ec')]}; {fc.probe_command(run, 'end', 'unit_probe_end.json')}; exit $ec"
    assert inner.index('unit_probe_start.json') < inner.index('supervisor_started_utc.txt') < inner.index('supervisor_finished_utc.txt') < inner.index('unit_probe_end.json') < inner.rindex('exit $ec')
    assert f'{fc.PY} -I -B {fc.PROBE} --when start' in inner and '|| true' in inner
    with pytest.raises(ValueError, match='forbidden'):
        fc.check_launch(['python', '-m', 'pytest'])


def test_the_probed_inner_command_runs_and_keeps_the_frozen_runs_exit_code(tmp_path):
    """DirectLauncher's real shell on a stand-in frozen run (a script in the run's snapshot that writes a file and exits 3): the exit code and the time stamps as inner_command writes them, the two probe records
    (this process's cgroup: a session, not a tag- unit, so systemd is not asked), the shell's exit the frozen run's."""
    run = Path(tempfile.mkdtemp(prefix='he_probe_run_'))
    try:
        script = run/'snapshot/scripts/run_stage_3d_e48_pg.py'
        script.parent.mkdir(parents=True)
        script.write_text("import sys, pathlib\npathlib.Path(sys.argv[2], 'ran.txt').write_text(sys.argv[1])\nsys.exit(3)\n")
        record = fc.DirectLauncher(run/'cache')(run, 'x')
        assert record['returncode'] == 3 and (run/'exit_code').read_text().strip() == '3' and (run/'ran.txt').read_text() == 'run'
        assert (run/'supervisor_started_utc.txt').is_file() and (run/'supervisor_finished_utc.txt').is_file()
        probes = record['probes']
        assert set(probes) == {'unit_probe_start.json', 'unit_probe_end.json'} and all(p['schema'] == 'bayesreg.he.unit_probe.v1' for p in probes.values())
        assert probes['unit_probe_start.json']['when'] == 'start' and probes['unit_probe_end.json']['when'] == 'end' and probes['unit_probe_end.json']['in_tag_unit'] is False
        assert probes['unit_probe_end.json']['systemd'] is None and '读不到' in probes['unit_probe_end.json']['systemd_unread']
    finally:
        shutil.rmtree(run, ignore_errors=True)


# ---- the unit probe (H2; C22.B4-6 (d)) ----------------------------------------------------------------------------------------------------------------
import unit_probe  # noqa: E402


def cgroup_world(tmp_path, leaf, files=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path/'cgroup'
    path = f'/user.slice/user-1000.slice/user@1000.service/app.slice/{leaf}'
    proc.write_text(f'0::{path}\n')
    d = tmp_path/'sys'/path.lstrip('/')
    d.mkdir(parents=True)
    for name, text in (files or {'memory.max': '12884901888\n', 'memory.swap.max': '4294967296\n', 'memory.peak': '7340032000\n', 'memory.events': 'low 0\nhigh 0\nmax 2\noom 0\noom_kill 0\noom_group_kill 0\n'}).items():
        (d/name).write_text(text)
    return proc, tmp_path/'sys'


def test_the_probe_reads_the_cgroup_and_asks_systemd_only_for_a_tag_unit(tmp_path):
    proc, root = cgroup_world(tmp_path, 'tag-he-smoke-a.service')
    asked = []

    def show(unit):
        asked.append(unit)
        return dict(Id=unit, MemoryMax='12884901888', MemorySwapMax='4294967296', OOMPolicy='continue'), None
    r = unit_probe.probe('start', proc=proc, root=str(root), show=show)
    assert r['in_tag_unit'] is True and r['unit'] == 'tag-he-smoke-a.service' and asked == ['tag-he-smoke-a.service']
    assert r['peak_bytes'] == 7340032000 and r['max_bytes'] == 12*2**30 and r['swap_max_bytes'] == 4*2**30 and r['events'] == dict(low=0, high=0, max=2, oom=0, oom_kill=0, oom_group_kill=0)
    assert set(r['unread']) == {'memory.swap.peak', 'memory.current', 'memory.swap.events'} and all(v.startswith('读不到') for v in r['unread'].values())
    expected = he.SUPERVISOR_UNIT
    assert he.unit_matches(r, expected) == dict(MemoryMax=True, MemorySwapMax=True, OOMPolicy=True)
    stop = dict(r, systemd=dict(r['systemd'], OOMPolicy='stop'))
    assert he.unit_matches(stop, expected)['OOMPolicy'] is False and he.unit_matches(dict(r, systemd=None), expected) == dict(MemoryMax=None, MemorySwapMax=None, OOMPolicy=None)
    rec = he.unit_record(r, dict(r, when='end', events=dict(r['events'], oom_kill=1)), expected)
    assert rec['oom_kills_at_end'] == 1 and rec['peak_bytes_at_end'] == 7340032000 and rec['in_tag_unit'] is True and rec['matches_at_start']['OOMPolicy'] is True
    proc2, root2 = cgroup_world(tmp_path/'s', 'session-461.scope')
    r2 = unit_probe.probe('end', proc=proc2, root=str(root2), show=lambda unit: (_ for _ in ()).throw(AssertionError('systemd asked for a session')))
    assert r2['in_tag_unit'] is False and r2['systemd'] is None and 'not a tag-*.service' in r2['systemd_unread']
    r3 = unit_probe.probe('start', proc=tmp_path/'no_such_file')
    assert r3['cgroup'] is None and r3['cgroup_unread'].startswith('读不到') and r3['files'] == {}
    assert he.safe_probe('x', probe=lambda when: 1/0)['error'].startswith('读不到: ZeroDivisionError')
    assert unit_probe.parse_flat('oom 1\noom_kill x\n') is None and unit_probe.parse_bytes('max') == 'max' and unit_probe.parse_bytes('?') is None
    # HE2 clean: memory.stat's anon and file (recorded only); not there: 读不到
    assert r['memory_stat'] is None and r['memory_stat_unread'].startswith('读不到')
    proc4, root4 = cgroup_world(tmp_path/'m', 'tag-he-smoke-b.service', files={'memory.max': 'max\n', 'memory.stat': 'anon 2147483648\nfile 1073741824\nkernel 4096\nfile_dirty 0\n'})
    r4 = unit_probe.probe('end', proc=proc4, root=str(root4), show=lambda unit: (None, '读不到: test'))
    assert r4['memory_stat'] == dict(anon=2*2**30, file=2**30) and r4['memory_stat_unread'] is None and 'memory.stat' not in r4['files']
    assert he.unit_record(r4, r4, expected)['memory_stat_at_end'] == dict(anon=2*2**30, file=2**30)
    proc5, root5 = cgroup_world(tmp_path/'n', 'tag-he-smoke-c.service', files={'memory.stat': 'anon 5\nshmem 0\n'})
    r5 = unit_probe.probe('end', proc=proc5, root=str(root5), show=lambda unit: (None, '读不到: test'))
    assert r5['memory_stat'] == dict(anon=5, file=None) and "no ['file']" in r5['memory_stat_unread']
    # HE2 diff review new minor 1: systemd answered, an item missing or empty: not read (None), so no refusal
    partial = dict(r, systemd=dict(Id='tag-he-smoke-a.service', MemoryMax=str(12*2**30), MemorySwapMax=None, OOMPolicy=''))
    assert he.unit_matches(partial, expected) == dict(MemoryMax=True, MemorySwapMax=None, OOMPolicy=None) and he.unit_start_problems(partial, expected) == []
    assert he.unit_matches(dict(r, systemd=dict(r['systemd'], OOMPolicy=' continue\n')), expected)['OOMPolicy'] is True
    assert he.gib_bytes('12G') == 12*2**30 and he.gib_bytes('512M') == 2**29 and he.gib_bytes('7') == 7 and he.gib_bytes('x') is None and he.gib_bytes('xG') is None


def test_systemctl_show_is_parsed_and_its_failures_recorded():
    class Done:
        def __init__(self, rc, out='', err=''):
            self.returncode, self.stdout, self.stderr = rc, out, err
    calls = []

    def ok(argv, **kw):
        calls.append(argv)
        return Done(0, 'Id=tag-x.service\nMemoryMax=12884901888\nOOMPolicy=continue\n')
    props, why = unit_probe.systemd_show('tag-x.service', run=ok)
    assert why is None and props['OOMPolicy'] == 'continue' and props['MemorySwapMax'] is None and calls[0][:4] == ['systemctl', '--user', 'show', 'tag-x.service']
    assert unit_probe.systemd_show('tag-x.service', run=lambda argv, **kw: Done(1, err='Failed to connect to bus'))[1].startswith('读不到: systemctl exit 1')
    assert 'not' in unit_probe.systemd_show('tag-x.service', run=lambda argv, **kw: Done(0, 'Id=other.service\n'))[1]

    def missing(argv, **kw):
        raise FileNotFoundError('systemctl')
    assert unit_probe.systemd_show('tag-x.service', run=missing)[1].startswith('读不到: FileNotFoundError')


def test_the_probe_script_writes_once_and_never_fails(tmp_path):
    out = tmp_path/'probe.json'
    env = dict(os.environ, PATH='/usr/bin:/bin')
    done = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'experiments/he/unit_probe.py'), '--when', 'start', '--out', str(out)], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and json.loads(out.read_text())['when'] == 'start'
    again = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'experiments/he/unit_probe.py'), '--when', 'end', '--out', str(out)], capture_output=True, text=True, env=env)
    assert again.returncode == 0 and 'FileExistsError' in again.stderr and json.loads(out.read_text())['when'] == 'start'      # written once; the second call fails quietly


def test_the_third_certificates_fallback_runs_in_a_unit_of_its_own(tmp_path):
    """H3: after a failed third certificate (in the evaluation unit's child), configuration C runs through UnitLauncher (tag-he-c3-<pair>, D0's limits, OOMPolicy=continue), whatever the pair's launcher."""
    for kind in ('unit', 'direct'):
        ctx = run_hx.make_context(tmp_path, 'p', tmp_path/'out', paths=dict(cache='/home/rmcfz/seq_vp_tag_results/cache', allowlist='a'), launcher=kind, check_identity=False)     # only strings are built
        launcher = run_hx.third_launcher(ctx)
        assert isinstance(launcher, fc.UnitLauncher) and launcher.unit_prefix == 'tag-he-c3' == run_hx.THIRD_UNIT_PREFIX
        argv = launcher.argv(Path('/home/rmcfz/seq_vp_tag_results/x/run'), '0438-0439')
        assert '--unit=tag-he-c3-0438-0439' in argv and '--property=OOMPolicy=continue' in argv and all(r in argv for r in fc.UNIT_RESOURCES)
    settings = he.frozen_settings()
    settings['units']['third_certificate_fallback'] = 'direct'
    ctx = run_hx.make_context(tmp_path, 'p', tmp_path/'out', settings=settings, paths=dict(cache=str(tmp_path/'cache'), allowlist='a'), check_identity=False)
    with pytest.raises(SystemExit, match='only unit'):
        run_hx.third_launcher(ctx)
    source = inspect.getsource(run_hx.third_cert_main)
    assert 'launcher = third_launcher(ctx) if launcher is None else launcher' in source and 'make_launcher' not in source
    assert source.index('launcher = third_launcher(ctx)') < source.index('o.rename(superseded)')               # HE2 review minor 4 (the pipeline test runs it)


def test_the_kernels_limits_and_the_supervisors_refusal_rule():
    """HE2 review minors 8, 9: the kernel's memory.max and memory.swap.max against the settings' unit (in a tag- unit only); a reading that differs stops the supervisor, 读不到 does not."""
    unit = he.SUPERVISOR_UNIT
    good = dict(unit='tag-x.service', in_tag_unit=True, max_bytes=12*2**30, swap_max_bytes=4*2**30, systemd=dict(MemoryMax=str(12*2**30), MemorySwapMax=str(4*2**30), OOMPolicy='continue'))
    assert he.kernel_matches(good, unit) == {'memory.max': True, 'memory.swap.max': True} and he.unit_start_problems(good, unit) == []
    assert he.kernel_matches(dict(good, max_bytes='max'), unit)['memory.max'] is False                                            # no limit at all is not 12G
    assert he.kernel_matches(dict(good, in_tag_unit=False, max_bytes='max'), unit) == {'memory.max': None, 'memory.swap.max': None}   # a session's cgroup: not judged
    assert he.kernel_matches(dict(good, swap_max_bytes=None), unit)['memory.swap.max'] is None
    assert he.unit_start_problems(dict(good, swap_max_bytes=2**30), unit) == ["kernel memory.swap.max: not the settings' unit"]
    assert he.unit_start_problems(dict(good, systemd=dict(good['systemd'], OOMPolicy='stop')), unit) == ["systemd OOMPolicy: not the settings' unit"]
    assert he.unit_start_problems(dict(good, systemd=None, max_bytes=None, swap_max_bytes=None), unit) == []
    record = he.unit_record(good, good, unit)
    assert record['kernel_matches_at_start'] == {'memory.max': True, 'memory.swap.max': True} and record['matches_at_start'] == dict(MemoryMax=True, MemorySwapMax=True, OOMPolicy=True)
    evaluation = dict(good, max_bytes=8*2**30, systemd=dict(good['systemd'], MemoryMax=str(8*2**30)))
    assert he.EVALUATION_UNIT['oom_policy'] == 'continue' and he.unit_start_problems(evaluation, he.EVALUATION_UNIT) == [] and he.unit_start_problems(good, he.EVALUATION_UNIT) == [     # review minor 14
        "systemd MemoryMax: not the settings' unit", "kernel memory.max: not the settings' unit"]


@pytest.mark.parametrize('rc, passed, want', [(9, True, 9), (9, False, 8), (0, True, 0), (0, False, 8), (2, True, 8), (8, True, 8)])
def test_spy_pair_passes_9_through_only_when_the_spies_saw_nothing(tmp_path, monkeypatch, rc, passed, want):
    """HE2 diff review new minor 4: spy_pair.main with run_pair replaced by a stand-in returning rc and spies that pass or not: 9 comes back as 9 when the spies passed, 8 when they did not (a breach
    comes first); 0 only with passing spies; any other code 8."""
    import spies as spying
    import spy_pair
    out = tmp_path/'out'
    out.mkdir()

    class Spies:
        def __init__(self, path):
            self.path = path

        def install(self):
            return self

        def remove(self):
            pass

        def record(self, *, expect_reads=True):
            return dict(passed=passed, positive_condition=passed, loader_calls=[], reads_seen={}, forbidden_access=[] if passed else ['pair.npz:fixed_seg35'], evaluation_records_opened=[])
    monkeypatch.setattr(spying, 'Spies', Spies)
    monkeypatch.setattr(spy_pair.run_hx, 'out_problem', lambda out: None)                      # the out is a test's directory (the real rule asks for the results root)
    monkeypatch.setattr(spy_pair.run_hx, 'run_pair', lambda ctx, child=None: rc)
    assert spy_pair.main(['--stage-dir', str(tmp_path), '--pair', 'p', '--out', str(out)]) == want
    record = json.loads((out/'spy.json').read_text())
    assert record['run_rc'] == rc and record['passed'] is passed


def test_a_start_string_carries_its_units_values():
    """HE2 review minors 8, 11: unit_argv_problems on run_mc0.unit_argv's string with the OOMPolicy inserted (the smoke's way): every value once; run_mc0's own MemorySwapMax=4G is the unit's."""
    import run_mc0
    unit = he.SUPERVISOR_UNIT
    argv = run_mc0.unit_argv('tag-he-x', Path('/home/rmcfz/seq_vp_tag_results/x/code'), ['pair'], log='/home/rmcfz/seq_vp_tag_results/x/log', numba=unit['numba'], quota=unit['quota'],
                             runtime=unit['runtime'], script='experiments/he/run_hx.py', memory_max=unit['memory_max'], nice=unit['nice'])
    assert he.unit_argv_problems(argv, unit) == ["--property=OOMPolicy=continue expected once, the string has []"]
    argv.insert(argv.index('--property=MemorySwapMax=4G')+1, '--property=OOMPolicy=continue')
    assert he.unit_argv_problems(argv, unit) == [] and unit['memory_swap_max'] == '4G'
    for old, new in (('--property=MemoryMax=12G', '--property=MemoryMax=10G'), ('--property=RuntimeMaxSec=4800', '--property=RuntimeMaxSec=600'), ('--setenv=NUMBA_NUM_THREADS=8', '--setenv=NUMBA_NUM_THREADS=1')):
        assert len(he.unit_argv_problems([new if a == old else a for a in argv], unit)) == 1
    assert len(he.unit_argv_problems(argv+['--property=OOMPolicy=stop'], unit)) == 1
    assert he.unit_argv_problems(argv, dict(unit, memory_swap_max='2G')) == ["--property=MemorySwapMax=2G expected once, the string has ['4G']"]


# ---- the CLI and the context ---------------------------------------------------------------------------------------------------------------------
def test_the_cli_refuses_an_out_outside_the_results_or_one_that_exists(tmp_path, capsys):
    assert run_hx.main(['pair', '--stage-dir', str(tmp_path), '--pair', 'p', '--out', str(tmp_path/'out')]) == 2 and 'the outputs go under' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()
    assert run_hx.out_problem(tmp_path, results=tmp_path.parent) == f'{tmp_path} exists: a pair is never rerun'
    assert run_hx.out_problem(tmp_path/'new', results=tmp_path) is None


def test_the_context_records_whether_the_values_are_the_frozen_ones(tmp_path):
    ctx = run_hx.make_context(tmp_path/'stage', 'p', tmp_path/'out/p', device='cpu', paths=dict(allowlist='a', cache=str(tmp_path)), launcher='direct', check_identity=False, workers=1, unseal_token='t')
    assert ctx['settings_frozen'] is True and ctx['unseal_token_sha256'] == hashlib.sha256(b't').hexdigest() and 't' not in json.dumps(ctx).split('unseal_token_sha256')[0][-5:]
    (tmp_path/'out/p').mkdir(parents=True)
    he.write_json(run_hx.context_path(ctx), ctx)
    assert run_hx.read_context(run_hx.context_path(ctx))['settings_sha256'] == ctx['settings_sha256']
    tampered = dict(ctx, settings=dict(ctx['settings'], fuse_rule='c-i'))
    (tmp_path/'out/p/context.json').write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match='sha256'):
        run_hx.read_context(run_hx.context_path(ctx))
    small = he.frozen_settings()
    small['t88']['pre']['iterations'] = [2, 2, 2, 4]
    assert run_hx.make_context(tmp_path, 'p', tmp_path/'q', settings=small, paths={}, check_identity=False)['settings_frozen'] is False


def test_the_command_line_refuses_a_context_that_relaxes_a_check_unless_it_is_a_marked_test_context(tmp_path):
    """Revision 1, review important 2: check_identity false or values other than the frozen ones only in a context made by make_context (marked test_context) that writes outside the results root."""
    results = Path('/home/rmcfz/seq_vp_tag_results')
    production = run_hx.make_context(tmp_path, 'p', results/'he_v1/p', paths=dict(allowlist='a', cache='c'))
    assert production['check_identity'] is True and production['test_context'] is None and run_hx.cli_context_problem(production) is None
    test = run_hx.make_context(tmp_path, 'p', tmp_path/'out/p', paths=dict(allowlist='a', cache='c'), check_identity=False)
    assert test['test_context'] == run_hx.TEST_CONTEXT and run_hx.cli_context_problem(test) is None
    assert 'not marked' in run_hx.cli_context_problem(dict(test, test_context=None))
    assert 'not marked' in run_hx.cli_context_problem(dict(production, check_identity=False))
    assert 'results root' in run_hx.cli_context_problem(dict(test, out=str(results/'he_v1/p')))
    small = he.frozen_settings()
    small['t88']['pre']['iterations'] = [2, 2, 2, 4]
    unfrozen = run_hx.make_context(tmp_path, 'p', tmp_path/'out/q', settings=small, paths=dict(allowlist='a', cache='c'))
    assert unfrozen['check_identity'] is True and unfrozen['test_context'] == run_hx.TEST_CONTEXT and run_hx.cli_context_problem(unfrozen) is None
    assert 'not marked' in run_hx.cli_context_problem(dict(unfrozen, test_context=None))
    record, problems = run_hx.check_identity(dict(test, test_context=None), tmp_path, 'identity_x.json')
    assert problems and record['checked'] is False and json.loads((tmp_path/'identity_x.json').read_text())['problems'] == problems
    (tmp_path/'out/p').mkdir(parents=True)
    he.write_json(tmp_path/'out/p/context.json', dict(test, test_context=None))
    for stage in ('t88', 'repair', 'third-cert', 'evaluate'):
        assert run_hx.main([stage, '--context', str(tmp_path/'out/p/context.json')]) == 8


def write_manifest(stage, pair, fixed, moving):
    d = stage/pair
    d.mkdir(parents=True)
    write_json(d/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=[5, 6, 7], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                                       sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256='0'*64))
    return d


def test_the_second_lock_of_the_admission(tmp_path):
    """Revision 1, review important 3: bayesreg.eval.truth's sealed lists before load_pair, on the names (no file opened) and on the manifest's subjects (the manifest only, through the whitelisting reader)."""
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed
    sealed = load_sealed_lists()
    held_b, held_1 = sorted(sealed.subjects['holdout_B_v1'])[0], sorted(sealed.subjects['H1'])[0]
    stage = tmp_path/'stage_any'
    write_manifest(stage, 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1', 'OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')
    ctx = dict(stage_dir=str(stage), pair='OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1')
    record = run_hx.sealed_admission(ctx)
    assert record['fixed_subject'] == 'OASIS_OAS1_0438_MR1' and record['pair_name_subjects'] == ['OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1'] and len(record['sealed_lists']) >= 4
    write_manifest(stage, f'{held_b}__OASIS_OAS1_0439_MR1', held_b, 'OASIS_OAS1_0439_MR1')
    with pytest.raises(SubjectSealed, match='holdout_B_v1'):
        run_hx.sealed_admission(dict(stage_dir=str(stage), pair=f'{held_b}__OASIS_OAS1_0439_MR1'))
    write_manifest(stage, 'pair_neutral', 'OASIS_OAS1_0438_MR1', held_1)
    with pytest.raises(SubjectSealed, match='sealed list H1'):
        run_hx.sealed_admission(dict(stage_dir=str(stage), pair='pair_neutral'))
    write_manifest(tmp_path/'stage_H4_v1', 'pair_x', 'OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')
    with pytest.raises(SubjectSealed, match='hold-out stage'):
        run_hx.sealed_admission(dict(stage_dir=str(tmp_path/'stage_H4_v1'), pair='pair_x'))
    with pytest.raises(ValueError):
        run_hx.sealed_admission(dict(stage_dir=str(stage), pair='../escape'))
    from bayesreg.eval.truth import SealedListError
    assert issubclass(SubjectSealed, run_hx.admission_refusals()) and issubclass(SealedListError, run_hx.admission_refusals())


def test_the_side_outcomes_tell_the_qc_from_a_refusal_whatever_the_order():
    """Revision 1, review minor 6: each side of the channel reader on its own record; a side whose QC failed does not stop the other; any other refusal of either side -> provenance_rejected."""
    import fast_channels as fast
    mi = run_hx.core.make_inputs()
    real_qc, real_side = mi.alignment_qc, fast.read_side

    def fake_side(qc_dice=None, refuse=None):
        def side(layer, pin, name, seg_root, workers=1):
            if refuse == name:
                raise SystemExit(f'A4p {name}: structure layer refused (ProvenanceRejected: x)')
            row = mi.alignment_qc(np.ones((3, 3, 3)), np.ones((3, 3, 3)) if qc_dice != name else np.zeros((3, 3, 3)))
            if not row['passed']:
                raise SystemExit(f'EF {name}: alignment QC {row["dice"]} < .8')
            return np.zeros((3, 3, 3), dtype=np.int16), dict(segmenter_id='x')
        return side
    for refuse, qc_side, want in ((None, 'fixed', 'qc'), ('moving', 'fixed', 'refused'), ('fixed', 'moving', 'refused'), (None, None, None)):
        fast.read_side = fake_side(qc_dice=qc_side, refuse=refuse)
        try:
            with run_hx.SideOutcomes() as outcomes:
                results, raised = [], None
                try:
                    for name in ('fixed', 'moving'):
                        results.append(fast.read_side(None, None, name, None))
                except SystemExit as exc:
                    raised = exc
            assert mi.alignment_qc is real_qc
        finally:
            fast.read_side = real_side
        if want == 'qc':
            assert raised is None and outcomes.qc_refused() and set(outcomes.sides) == {'fixed'} and results[0] == (None, dict(qc_refused=True)) and len(results) == 2
        elif want == 'refused':
            assert raised is not None and not outcomes.qc_refused() and any(s['outcome'] == 'refused' for s in outcomes.sides.values())
        else:
            assert raised is None and outcomes.sides == {} and not outcomes.qc_refused()
    assert fast.read_side is real_side


@pytest.mark.parametrize('qc_side, refused_side', [('fixed', 'moving'), ('moving', 'fixed'), ('fixed', None), ('both', None)])
def test_the_side_outcomes_in_two_threads(qc_side, refused_side):
    """Revision 2, review new minor 5: the production path reads the two sides in two threads (fast_channels.read_predicted with workers > 1 -> mc0_core._pool_map(..., 2)); each side's outcome stays its own
    (threading.local), the SystemExit comes back through the pool: a QC refusal on one side and a provenance refusal on the other is provenance_rejected in either order; QC only is a QC refusal."""
    import threading
    import time
    import fast_channels as fast
    mi = run_hx.core.make_inputs()
    real_side = fast.read_side
    barrier = threading.Barrier(2, timeout=10)                                                   # both sides really run at the same time

    def side(layer, pin, name, seg_root, workers=1):
        barrier.wait()
        if refused_side == name:
            time.sleep(.05 if name == 'fixed' else 0.)
            raise SystemExit(f'A4p {name}: structure layer refused (ProvenanceRejected: x)')
        fail = qc_side in (name, 'both')
        row = mi.alignment_qc(np.ones((3, 3, 3)), np.zeros((3, 3, 3)) if fail else np.ones((3, 3, 3)))
        time.sleep(.05 if name == 'moving' else 0.)
        if not row['passed']:
            raise SystemExit(f'EF {name}: alignment QC {row["dice"]} < .8')
        return np.zeros((3, 3, 3), dtype=np.int16), dict(segmenter_id='x')
    fast.read_side = side
    threads = set()
    try:
        with run_hx.SideOutcomes() as outcomes:
            raised = None
            try:
                rows = run_hx.core._pool_map(lambda name: (threads.add(threading.get_ident()), fast.read_side(None, None, name, None))[1], ('fixed', 'moving'), 2)
            except SystemExit as exc:
                raised, rows = exc, None
    finally:
        fast.read_side = real_side
    assert len(threads) == 2 and threading.get_ident() not in threads                         # two worker threads, not this one
    if refused_side is not None:
        assert raised is not None and 'structure layer refused' in str(raised) and not outcomes.qc_refused()
        assert outcomes.sides[refused_side]['outcome'] == 'refused' and outcomes.sides[qc_side]['outcome'] == 'qc' and outcomes.sides[qc_side]['alignment_qc']['passed'] is False
    else:
        assert raised is None and outcomes.qc_refused() and set(outcomes.sides) == ({'fixed', 'moving'} if qc_side == 'both' else {'fixed'})
        assert all(r == (None, dict(qc_refused=True)) for name, r in zip(('fixed', 'moving'), rows) if name in outcomes.sides)


@needs_frozen_tree
def test_every_process_must_find_the_head_pinned_in_the_context(tmp_path):
    """Revision 2, review new minor 1: make_context pins the commit of the checkout; every identity check of the pair's processes compares HEAD with it."""
    paths = dict(allowlist='a', cache=str(tmp_path), frozen=str(FROZEN))
    ctx = run_hx.make_context(tmp_path, 'p', tmp_path/'out/p', paths=paths)
    done = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    assert ctx['code_head'] == done.stdout.strip() and len(ctx['code_head']) == 40
    _, problems = run_hx.check_identity(ctx, tmp_path, 'identity_same.json')
    assert not [p for p in problems if 'pinned' in p]
    _, problems = run_hx.check_identity(dict(ctx, code_head='0'*40), tmp_path, 'identity_other.json')
    assert any('is not the commit pinned' in p for p in problems)
    _, problems = run_hx.check_identity(dict(ctx, code_head=None), tmp_path, 'identity_none.json')
    assert any('is not the commit pinned' in p for p in problems)
    assert json.loads((tmp_path/'identity_other.json').read_text())['pinned_head'] == '0'*40


def test_a_context_that_skips_the_identity_never_writes_under_the_results_root(tmp_path):
    """Revision 2, review new minor 3: also through the function interface; values other than the frozen ones with the identity checked may (the smoke's forced fallback)."""
    results = Path('/home/rmcfz/seq_vp_tag_results')
    out = results/'he_test_never_written'/'p'
    with pytest.raises(ValueError, match='results root'):
        run_hx.make_context(tmp_path, 'p', out, paths=dict(allowlist='a', cache='c'), check_identity=False)
    ctx = run_hx.make_context(tmp_path, 'p', tmp_path/'q', paths=dict(allowlist='a', cache='c'), check_identity=False)
    bad = dict(ctx, out=str(out))
    assert run_hx.run_pair(bad) == 2 and run_hx.evaluate_main(bad) == 2 and run_hx.third_cert_main(bad) == 2 and not out.parent.exists()
    small = he.frozen_settings()
    small['t88']['pre']['iterations'] = [2, 2, 2, 4]
    smoke = run_hx.make_context(tmp_path, 'p', out, settings=small, paths=dict(allowlist='a', cache='c'))
    assert smoke['check_identity'] is True and smoke['settings_frozen'] is False and run_hx.function_context_problem(smoke) is None
    line = run_hx.line_base(ctx)
    assert line['check_identity'] is False and line['test_context'] == run_hx.TEST_CONTEXT and line['code_head'] == ctx['code_head']


# ---- the spies (revision 1, review important 1) --------------------------------------------------------------------------------------------------
def spied_pair(tmp_path):
    d, arrays, allowlist, _ = synthetic_pair(tmp_path)
    return d/'pair.npz', allowlist


@pytest.mark.parametrize('breach', ['loader_member', 'member_list', 'dict', 'iteration', 'npzfile_direct', 'zipfile_direct', 'zipfile_read'])
def test_the_spies_raise_and_record_a_forbidden_read(tmp_path, breach):
    import zipfile
    import spies as spying
    original, _ = spied_pair(tmp_path)
    actions = dict(loader_member=lambda: pair_module._open_npz(original)['fixed_seg4'], member_list=lambda: np.load(original).files, dict=lambda: dict(np.load(original)),
                   iteration=lambda: list(np.load(original)), npzfile_direct=lambda: np.lib.npyio.NpzFile(open(original, 'rb'))['moving_seg35'],
                   zipfile_direct=lambda: zipfile.ZipFile(original).open('fixed_seg35.npy').read(), zipfile_read=lambda: zipfile.ZipFile(original).read('moving_seg4.npy'))
    with spying.Spies(original) as spy:
        with pytest.raises(spying.Breach):
            actions[breach]()
    record = spy.record()
    assert record['passed'] is False and (record['forbidden_access'] or breach in ('member_list', 'iteration', 'dict'))
    assert any(a.startswith(('member', 'attribute', 'iteration')) for a in spy.original_access+spy.npz_access+spy.zip_access)
    assert np.load is spy.real_load and np.lib.npyio.NpzFile.__getitem__ is spy.real_getitem and zipfile.ZipFile.open is spy.real_zip_open      # removed


def test_the_spies_pass_only_when_they_saw_every_read(tmp_path):
    import spies as spying
    original, allowlist = spied_pair(tmp_path)
    with spying.Spies(original) as spy:
        pair_module.load_pair(original, allowlist=allowlist)                                    # through the module, as run_hx.load_images calls it (the spy counts the module's function)
        fc.read_image_members(original)
    record = spy.record()
    assert record['passed'] is True and record['loader_calls'] == ['bayesreg.io.pair.load_pair', 'firewall_c.read_image_members'] and record['reads_seen'] == dict(fixed=2, moving=2) == record['reads_at_the_class']
    with spying.Spies(original) as spy:                                                          # (f) nothing read: not a passing spy when reads were expected
        pass
    assert spy.record()['passed'] is False and spy.record(expect_reads=False)['passed'] is True
    with spying.Spies(original) as spy:                                                          # a read the loaders did not account for (another way in): not passed
        pair_module.load_pair(original, allowlist=allowlist)
        with np.load(original) as z:
            z['fixed'], z['moving']
    assert spy.record()['passed'] is False and spy.record()['positive_condition'] is False
    with spying.Spies(original) as spy:                                                          # (e) an evaluation record read: not passed; written: not counted
        pair_module.load_pair(original, allowlist=allowlist)
        for name in ('metrics_0600.json', 'repair_metrics.json', 'evaluation.json', 'labels.json', 'smoothness.json'):
            (tmp_path/name).write_text('{}')
        assert spy.record()['passed'] is True
        for name in ('metrics_0600.json', 'repair_metrics.json', 'evaluation.json', 'labels.json', 'smoothness.json'):
            json.loads((tmp_path/name).read_text())
    record = spy.record()
    assert record['passed'] is False and len(record['evaluation_records_opened']) == 5
