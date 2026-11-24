"""R2's 19-pair registration reading (experiments/r2/run_r2f.py, r2f_core.py): HS's modules loaded once more on R2's core.  CPU; tmp_path, the committed HS reference and HS's records only read.

- the copies are bound to R2 (root r2_v1, units tag-r2-*, 10 s between units, the C: floor 40 GiB, the upstream decision_r2.json, the registry segmenters_r2_v1.json) and HS's own modules are not
  touched (hs_core, decide_hs, hs_seg, run_hs imported in the same process keep HS's names and functions);
- the reference is HS: decision_hs.json pinned, every pair hx_certified, the records those of the committed reference and of batch.json's pins; each departure a problem of its pair;
- the reading: Delta = R2 - HS, mean >= +.003 and the block lower bound > -.003, 19/19, SDlogJ <= .10; "更好" only with the lower bound > 0; the undecidable cases;
- the expectation (slope x the pair-level Delta of the candidate against UF4 in eval_r2.json) and its file before the first field; the decision and its files;
- the upstream: no file, not pass, a candidate that is not one; the seg phase's inputs; the drift rule; another root refused; the dry run in its own process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT/'experiments'/'r2') not in sys.path:
    sys.path.insert(0, str(ROOT/'experiments'/'r2'))
import r2_common as rc  # noqa: E402
import run_r2f as F  # noqa: E402

hc, D, R = F.hc, F.D, F.R
ALL = list(hc.ALL_PAIRS)


def test_the_copies_are_bound_to_r2():
    assert R.HS is hc.HS and D.HS is hc.HS and R.hc is hc and R.decision is D and R.hs_seg is F.r2_seg.seg and F.r2_seg.seg.hc is hc
    assert hc.HS.DEFAULT_ROOT == rc.RESULTS/'r2_v1' and hc.HS.PREFIX == 'r2' and hc.HS.registry_path() == hc.core.ROOT/rc.REGISTRY_REL
    cell = hc.HS.field_cell(ALL[0])
    assert cell.run_id == 'r2_SHx24_first5_0438-0439' and cell.unit == 'tag-r2-shx24-first5-0438-0439' and hc.HS.repair_cell(ALL[7]).run_id == 'r2_RSHx24_rest600_0445-0446'
    base = R.BASE
    assert base.fl is hc.HS and base.decision is D and base.SCRIPT == F.SCRIPT and base.EVAL_UNIT == 'tag-r2-eval' and base.EVAL_R1_UNIT == 'tag-r2-eval-r'
    assert base.GAP_SECONDS == 10 and base.DEFAULT_MIN_HOST_GIB == 40. and base.GATE.host_floor_gib == 40. and base.GATE.own_units == ('tag-r2-queue.service',)
    assert R.QUEUE_UNIT == 'tag-r2-queue' and R.INFER_UNIT == 'tag-r2-infer' and R.FALLBACK_UNIT == 'tag-r2-fallback' and R.SCRIPT == 'experiments/r2/run_r2f.py'
    argv = base.run_argv(cell, Path('/x/code'), hc.HS.DEFAULT_ROOT, 'abc')
    assert '--unit=tag-r2-shx24-first5-0438-0439' in argv and '/x/code/experiments/r2/run_r2f.py' in argv and 'pytest' not in ' '.join(argv)
    launch = R.queue_launch('fields')
    assert '--unit=tag-r2-queue' in launch and launch[launch.index('--gap-seconds')+1] == '10' and launch[launch.index('--min-host-gib')+1] == '40'
    assert base.fast.read_predicted.__func__ is R.HSFast.read_predicted and base.oracle_mark_checked is R.hs_oracle_mark_checked
    assert hc.UF_OUT == rc.OUT and hc.REGISTRY_REL == 'configs/bayesreg/segmenters_r2_v1.json' and set(hc.CANDIDATES) == {'R2S'}
    assert hc.CANDIDATES['R2S']['uf_arm'] == 'R2S' and hc.CANDIDATES['R2S']['nn_model'] is None and hc.CANDIDATES['R2S']['recipe'] == 'r2s'
    assert D.rw_reference is F.hs_reference and D.read_hs is F.read_r2 and D.write_hs is F.write_hs and R.rw_lines is F.hs_lines and R.queue is F.queue


def test_hss_own_modules_are_not_touched():
    import decide_hs
    import hs_core
    import hs_seg
    import run_hs
    assert hs_core.HS.ROOT_NAME == 'hs_v1' and hs_core.HS is not hc.HS and decide_hs.HS is hs_core.HS and decide_hs.hc is hs_core and hs_seg.hc is hs_core
    assert decide_hs.rw_reference is not F.hs_reference and decide_hs.read_hs.__module__ == 'decide_hs' and hs_core.REGISTRY_REL == 'configs/bayesreg/segmenters_hs_v1.json'
    assert run_hs.HS is hs_core.HS and run_hs.BASE.EVAL_UNIT == 'tag-hs-eval' and run_hs.QUEUE_UNIT == 'tag-hs-queue' and run_hs.BASE.fl is hs_core.HS and run_hs.decision is decide_hs
    assert hs_core.ADOPT_MEAN_MIN == 0. and hc.ADOPT_MEAN_MIN == .003 and hc.ADOPT_LOWER == -.003


def test_the_drift_rule():
    assert R.is_drift_forbidden('experiments/r2/run_r2f.py') and R.is_drift_forbidden('experiments/hs/hs_seg.py') and R.is_drift_forbidden('bayesreg/seg/unet/small.py')
    assert not R.is_drift_forbidden('configs/bayesreg/segmenters_r2_v1.json') and not R.is_drift_forbidden('runs/20261110-r2/plan.md')
    assert R.DRIFT_APPROVABLE == ('experiments/r2/run_r2f.py',) and R.DRIFT_APPROVAL_REL == 'runs/20261110-r2/drift_approval.json'
    assert {rel for _, rel in R.READING_FILES} >= {'experiments/r2/run_r2f.py', 'experiments/r2/r2f_core.py', 'experiments/hs/decide_hs.py'}


def test_another_root_is_refused(tmp_path):
    o = SimpleNamespace(root=str(tmp_path/'r2_v1'), phase='fields', segcheck_reread=None)
    assert F.queue(o) == 2 and not (tmp_path/'r2_v1').exists()


def test_the_seg_phase_floor(monkeypatch):
    """Review m6: the seg phase starts at --min-host-gib 51 or more (40 + the predictions and fields that follow, about 10.5 GB); the fields phase at 40."""
    seg, fields = R.queue_launch('seg'), R.queue_launch('fields')
    assert seg[seg.index('--min-host-gib')+1] == '51' and fields[fields.index('--min-host-gib')+1] == '40' and F.SEG_MIN_HOST_GIB == 51.
    called = []
    monkeypatch.setattr(R.run, 'queue', lambda *a, **k: called.append(1) or 0)
    o = SimpleNamespace(root=str(hc.HS.DEFAULT_ROOT), phase='seg', segcheck_reread=None, min_host_gib=40.)
    assert F.queue(o) == 2 and not called
    assert F.queue(SimpleNamespace(root=str(hc.HS.DEFAULT_ROOT), phase='seg', segcheck_reread=None, min_host_gib=51.)) == 0 and called == [1]


# ---- review m1: the bytes R2 is bound to, and every replaced name checked -------------------------------------------------------------------------------
HS_PINS = {'experiments/hs/run_hs.py': '596706f681926f994d1e21cb4499f2935e5fa666660285bdd9233983a92d03f8', 'experiments/hs/hs_seg.py': '5aea1cb5e7b39df16d6c099ea112cad045db35606597b9010cb95d23afbf3c7f',
           'experiments/hs/decide_hs.py': 'c73ec0f990ca6389b646d5aa25b9da013e22b8a55be5e48c314fac09f3c7af08', 'experiments/hs/hs_core.py': '85ba0e50b575c83adc421fb5e977a5d56083baede4dae025c6fd570b0a5f5fe8',
           'experiments/fl/run_fl.py': '8949b754c86575444963c56793feb5e5f44878749ff3209940333288b551339a'}


def test_the_hs_files_r2_reloads_are_the_ones_it_was_written_on():
    """A change of any of them must come back here (and to R2's review): R2 replaces names in them and uses the rest as they are."""
    assert {rel: rc.sha256_file(ROOT/rel) for rel in HS_PINS} == HS_PINS


def test_a_replaced_name_must_exist():
    module = SimpleNamespace(a=1)
    with pytest.raises(AttributeError, match='changed under R2'):
        hc.replace_names(module, a=2, b=3)
    assert module.a == 1 and not hasattr(module, 'b')
    assert hc.replace_names(module, a=2).a == 2


# ---- the upstream -------------------------------------------------------------------------------------------------------------------------------
def write_decision(out, **document):
    (out/'analysis').mkdir(parents=True, exist_ok=True)
    (out/'analysis'/'decision_r2.json').write_text(json.dumps(document))
    (out/'eval_r2').mkdir(exist_ok=True)
    (out/'eval_r2'/'eval_r2.json').write_text('{}')


def test_the_upstream(tmp_path):
    waiting, stop, candidate, _ = hc.uf_reading(tmp_path)
    assert waiting and stop is None and candidate is None
    write_decision(tmp_path, reading='not_pass', reading_text='不过门')
    assert hc.uf_reading(tmp_path)[1].startswith('R2 的分割级读数是 not_pass')
    write_decision(tmp_path, reading='pass', candidate='R2X')
    assert 'R2X' in hc.uf_reading(tmp_path)[1]
    write_decision(tmp_path, reading='pass', candidate='R2SL')
    assert hc.uf_reading(tmp_path)[2] is None and 'R2SL' in hc.uf_reading(tmp_path)[1]          # review B1 (c): one arm, R2SL refused
    write_decision(tmp_path, reading='pass', candidate='R2S')
    assert hc.uf_reading(tmp_path)[2] == 'R2S' and hc.HS.candidate(tmp_path) == 'R2S' and hc.HS.segmenter_id(tmp_path) == 'unet35_r2s_ens'
    with pytest.raises(SystemExit):
        hc.HS.candidate(tmp_path/'none')


def test_the_seg_phase_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'UF_OUT', tmp_path/'uf')
    monkeypatch.setattr(hc, 'UF_OUT', tmp_path/'out')
    problems = F.seg_input_problems(candidate='R2S')
    assert len(problems) == 6
    for d in [tmp_path/'uf'/n for n in rc.UF_ORDER]+[tmp_path/'out'/'r2s'/n for n in rc.MEMBER_NAMES]:
        d.mkdir(parents=True)
        (d/'member.json').write_text('{}')
    assert F.seg_input_problems(candidate='R2S') == []


# ---- the reference --------------------------------------------------------------------------------------------------------------------------------
def metrics_of(seg35, sdlogj=.07):
    return dict(seg35=seg35, seg4=.92, hd95=1.3, sdlogj_official=sdlogj, sdlogj_brain=.06, small_six=.69, large_six=.91, other_23=.92, cortex=.87, white_matter=.93)


def hs_world(tmp_path, monkeypatch, *, status=None):
    results = tmp_path/'results'
    root = results/'hs_v1'
    pins = {}
    for i, pair in enumerate(ALL):
        short = hc.core.short_name(pair)
        d = root/'runs'/('RSHx24_c600' if i < 5 else 'RSHx24_rest600')/short
        d.mkdir(parents=True)
        (d/'repair.json').write_text(json.dumps(dict(pair=pair, n=i)))
        (d/'repair_metrics.json').write_text(json.dumps(dict(pair=pair, m=i)))
        pins[short] = dict(repair_json_sha256=rc.sha256_file(d/'repair.json'), repair_metrics_sha256=rc.sha256_file(d/'repair_metrics.json'))
    per_pair = [dict(pair=hc.core.short_name(p), line_status=(status or {}).get(i, 'hx_certified'), metrics=metrics_of(.87+.001*i)) for i, p in enumerate(ALL)]
    (root/'analysis').mkdir(parents=True)
    (root/'analysis'/'decision_hs.json').write_text(json.dumps(dict(reading='采用新分割器', per_pair=per_pair)))
    monkeypatch.setattr(rc, 'HS_DECISION_SHA256', rc.sha256_file(root/'analysis'/'decision_hs.json'))
    monkeypatch.setattr(F, 'committed_reference', lambda repo=None: pins)
    return results, pins


def test_the_hs_reference(tmp_path, monkeypatch):
    results, _ = hs_world(tmp_path, monkeypatch)
    reference = F.hs_reference(results)
    assert all(reference[p]['problem'] is None for p in ALL) and reference[ALL[3]]['metrics']['seg35'] == pytest.approx(.873)
    pins = F.hs_pins(results)
    assert all(F.hs_reference(results, pins)[p]['problem'] is None for p in ALL)
    changed = dict(pins, **{hc.core.short_name(ALL[2]): dict(pins[hc.core.short_name(ALL[2])], **{'repair.json': '0'*64})})
    assert [p for p in ALL if F.hs_reference(results, changed)[p]['problem']] == [ALL[2]]


def test_the_hs_reference_problems(tmp_path, monkeypatch):
    results, _ = hs_world(tmp_path, monkeypatch, status={4: 'hx_fallback_configC'})
    assert [p for p in ALL if F.hs_reference(results)[p]['problem']] == [ALL[4]]
    monkeypatch.setattr(rc, 'HS_DECISION_SHA256', '0'*64)
    assert all('钉住' in F.hs_reference(results)[p]['problem'] for p in ALL)


def test_the_hs_reference_against_the_committed_one(tmp_path, monkeypatch):
    results, pins = hs_world(tmp_path, monkeypatch)
    short = hc.core.short_name(ALL[9])
    path = results/'hs_v1'/'runs'/'RSHx24_rest600'/short/'repair_metrics.json'
    path.write_text(path.read_text()+' ')
    assert [p for p in ALL if F.hs_reference(results)[p]['problem']] == [ALL[9]]


@pytest.mark.skipif(not (rc.HS_ROOT/'analysis'/'decision_hs.json').is_file(), reason='HS records not on this machine')
def test_the_real_hs_reference_is_19_of_19():
    reference = F.hs_reference()
    assert all(reference[p]['problem'] is None for p in ALL)
    assert D.mean(reference[p]['metrics']['seg35'] for p in ALL) == pytest.approx(.8778, abs=1e-4)


# ---- the reading -----------------------------------------------------------------------------------------------------------------------------
def rows_of(deltas, *, sdlogj=.07, override=None):
    rows = {p: dict(pair=hc.core.short_name(p), kind='certified', why=None, line_status=hc.HX_CERTIFIED, metrics=metrics_of(.87+deltas[i], sdlogj)) for i, p in enumerate(ALL)}
    for pair, change in (override or {}).items():
        rows[pair] = dict(rows[pair], **change)
    return rows


def reference_of(problem=None):
    return {p: dict(metrics=metrics_of(.87), problem=problem if p == ALL[0] else None) for p in ALL}


def test_adopt_and_better():
    reading, reason, compare = F.read_r2(rows_of([.00390625]*19), reference_of())
    assert reading == hc.READ_ADOPT and reason is None and compare['seg35']['lower'] > 0 and F.better_of(reading, compare) is True


@pytest.mark.parametrize('mean, lower, sdlogj, reading, better', [
    (.003, -.0029, .10, '采用新分割器', False), (.0029999, .001, .07, '不采用', False), (.004, -.003, .07, '不采用', False), (.004, 0., .07, '采用新分割器', False),
    (.004, 1e-9, .07, '采用新分割器', True), (.004, .001, .1000001, '不采用', False)])
def test_the_lines_on_their_bars(monkeypatch, mean, lower, sdlogj, reading, better):
    monkeypatch.setattr(D, 'compare_rows', lambda rows, reference: dict(seg35=dict(mean=mean, lower=lower, n=19), hs_sdlogj_official=sdlogj))
    got, _, compare = F.read_r2(rows_of([.004]*19), reference_of())
    assert got == reading and F.better_of(got, compare) is better


@pytest.mark.parametrize('deltas, sdlogj, word', [([.002]*19, .07, '+0.003'), ([.00390625]*19, .11, 'SDlogJ'), ([.05]+[-.02]*18, .07, '< +0.003')])
def test_not_adopted(deltas, sdlogj, word):
    reading, reason, compare = F.read_r2(rows_of(deltas, sdlogj=sdlogj), reference_of())
    assert reading == hc.READ_NOT and word in reason and F.better_of(reading, compare) is False


def test_a_wide_interval_is_not_adopted():
    deltas = [.05, -.04]*9+[.05]
    reading, reason, compare = F.read_r2(rows_of(deltas), reference_of())
    assert compare['seg35']['mean'] >= .003 and compare['seg35']['lower'] <= -.003 and reading == hc.READ_NOT and '下界' in reason


@pytest.mark.parametrize('override', [{ALL[3]: dict(kind=D.NOT_RUN, why='0441-0442 场没有运行', metrics=None, line_status=None)}, {ALL[5]: dict(line_status=hc.HX_REJECTED, kind='rejected', why='被拒', metrics=None)}])
def test_undecidable(override):
    assert F.read_r2(rows_of([.004]*19, override=override), reference_of())[0] == hc.UNDECIDABLE


def test_a_reference_problem_is_undecidable():
    assert F.read_r2(rows_of([.004]*19), reference_of('HS 这一对不是 hx_certified'))[0] == hc.UNDECIDABLE


def test_the_expectation():
    subjects = list(rc.SUBJECTS)
    rows = dict(UF4={s: dict(seg35=.93) for s in subjects}, R2S={s: dict(seg35=.935) for s in subjects})
    e = F.expectation(dict(rows=rows), 'R2S', rw_mean=.8778)
    assert e['pairs19']['seg_level_delta'] == pytest.approx(.005) and e['pairs19']['expected_delta'] == pytest.approx(.0089*.5) and e['expected_r2_seg35'] == pytest.approx(.8778+.00445)
    assert e['first5']['expected_range'] == pytest.approx([.84*.005, .92*.005]) and e['arm'] == 'R2S'


def test_the_decision_and_its_files(tmp_path, monkeypatch):
    root = tmp_path/'r2_v1'
    up = tmp_path/'up'
    write_decision(up, reading='pass', candidate='R2S', primary_arm='R2S', delta_seg35=.006)
    monkeypatch.setattr(hc.HS, 'UF_OUT', up)
    rows = rows_of([.00390625]*19)
    monkeypatch.setattr(D, 'method_row', lambda root, pair, results=None: rows[pair])
    monkeypatch.setattr(D, 'field_row', lambda root, pair: None)
    monkeypatch.setattr(F, 'hs_reference', lambda results=None, pins=None: reference_of())
    subjects = list(rc.SUBJECTS)
    F.write_prediction(root, dict(rows=dict(UF4={s: dict(seg35=.93) for s in subjects}, R2S={s: dict(seg35=.935) for s in subjects})), 'R2S', .8778, 'e'*64)
    assert json.loads((root/'analysis'/F.PREDICTION).read_text())['eval_r2_sha256'] == 'e'*64
    document = F.write_hs(root)
    assert document['schema'] == 'bayesreg.r2f.decision.v1' and document['reading'] == hc.READ_ADOPT and document['better'] is True and document['candidate'] == 'R2S'
    assert document['prediction']['pairs19']['seg_level_delta'] == pytest.approx(.005) and document['gap_to_89'] == pytest.approx(.89-(.87+.00390625))
    assert (root/'analysis'/F.SUMMARY).read_text().startswith('# R2 的 19 对配准读数') and (root/'analysis'/F.PER_PAIR).is_file() and not (root/'analysis'/'decision_hs.json').exists()


@pytest.mark.skipif(not (rc.HS_ROOT/'analysis'/'decision_hs.json').is_file(), reason='HS records not on this machine')
def test_the_dry_run_in_its_own_process(tmp_path):
    shim = tmp_path/'bin'
    shim.mkdir()
    (shim/'systemd-run').write_text('#!/bin/sh\nexit 99\n')
    (shim/'systemd-run').chmod(0o755)
    env = dict(os.environ, PATH=f"{shim}:{os.environ.get('PATH', '')}", CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    done = subprocess.run([sys.executable, '-B', str(ROOT/'experiments'/'r2'/'run_r2f.py'), '--dry-run'], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    assert 'torch imported by this dry run: False; problems: none' in done.stdout and 'tag-r2-queue' in done.stdout and "HS (hs_v1" in done.stdout
    assert not (rc.RESULTS/'r2_v1').exists()
