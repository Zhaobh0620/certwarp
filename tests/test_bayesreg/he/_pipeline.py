"""The HE pipeline on a synthetic world, in one process under the spies (run by the tests with PYTHONPATH=<frozen>:<repo>; CPU only).

    python -B _pipeline.py SPEC_JSON RESULT_JSON

SPEC: world (from _world.make_world), out, settings changes, the launcher of configuration C ('cpu_control' | 'absent' | 'handover' | 'handover_bad_seed'), the stage overrides ('folded_t88': the T88 child
replaced by a writer of a folded field, so that the real repair child has work; 'fuse': 'after_round' | 'mid_round', a fake clock in the repair process only), evaluate (bool), review ('mismatch_pass' |
'mismatch_fail' | None), expect_reads (the spy's positive condition), changes of the T88 settings.
The spies (experiments/he/spies.py), for the method's processes (the supervisor and its in-process children): the original pair.npz watched through np.load, NpzFile and zipfile; every file opened recorded;
the image loaders counted.  The evaluation runs after the method with the spies off; its order (the review certificate before the truth) is recorded by wrappers.
"""
import json
import os
from pathlib import Path
import sys

os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))     # before the libraries start their pools (the certificate's fallback workers, the frozen worker's threads): a test, the machine may be training

import numpy as np  # noqa: E402

spec = json.loads(Path(sys.argv[1]).read_text())
result_path = Path(sys.argv[2])
REPO = Path(spec['repo'])
sys.path.insert(0, str(REPO/'experiments/he'))
import run_hx  # noqa: E402
import spies as spying  # noqa: E402

he, fc = run_hx.he, run_hx.fc
core = he.core
core.frozen_api()
world = spec['world']
ORIGINAL = (Path(world['stage_dir'])/world['pair']/'pair.npz').resolve()
Breach = spying.Breach


def Spies():                                    # noqa: N802
    return spying.Spies(ORIGINAL)


def settings_of(spec):
    s = he.frozen_settings()
    for key, value in (spec.get('t88_pre') or {}).items():
        s['t88']['pre'][key] = value
    s['repair'].update(spec.get('repair') or {})
    if spec.get('tissue_prior'):                                        # HE2: 'off' is the ablation's definition (C22.B0-2)
        s['tissue_prior'] = spec['tissue_prior']
    if spec.get('segmenter'):                                           # HE2: settings whose segmenter is not the anchor's (C22.9-4 (xv))
        s['segmenter'] = dict(s['segmenter'], **spec['segmenter'])
    return s


def install_tissue_fault(fault):
    """HE2, D86 in the T88 child (in this process): 'implementation' -> p2_core.tissue_levels raises its coarse-grid message; 'other' -> the tissue layer's reader raises an OSError."""
    if fault == 'implementation':
        import he_c

        def coarse(*args, **kwargs):
            raise ValueError('tissue pyramid: the coarse grid differs from four_level_grids')
        he_c.p2.tissue_levels = coarse
    elif fault == 'other':
        from bayesreg.cli import run_pair as reader

        def unreadable(*args, **kwargs):
            raise OSError('an Atropos file could not be read (test)')
        reader.load_tissue_layer = unreadable


def install_reader_fault(fault):
    """HE2 review minor 6: the channel reader's record shows another registry only after StructureLayer passed (fast_channels.read_predicted wrapped; the predictions, the checks and the labels are the
    real ones): 'registry_sha256' -> sides.moving.registry_sha256 changed; 'segmenter_registry' -> channel_sources.segmenter.registry another path."""
    if not fault:
        return
    import fast_channels
    real = fast_channels.read_predicted

    def shown_otherwise(*args, **kwargs):
        predicted = real(*args, **kwargs)
        if fault == 'registry_sha256':
            predicted.record['sides']['moving']['registry_sha256'] = '0'*64
        else:
            predicted.record['segmenter'] = dict(predicted.record['segmenter'], registry='/elsewhere/configs/bayesreg/segmenters_hs_v1.json')
        return predicted
    fast_channels.read_predicted = shown_otherwise


def install_wall_after_pass0(on):
    """HE2 review minor 7, diff review new minor 2: the 900 s wall is the three passes' sum (started = t0 for every pass).  Once pass 0 has returned, the clock of the copied loop (p2_core.perf_counter,
    the passes > 0) is moved so that it reads kwargs['started'] + 900.5 s at that moment: measured from t0 the wall is just passed, so pass 1 stops at its first step (level 0, iteration 0); measured from
    pass 1's own start (he_c's own, unmoved clock) it would be 900.5 s less the time from t0 to the end of pass 0 (recorded: the test asks it to be > 1 s), so pass 1 would run through.  Either way the two
    are told apart; so is started None (p2_core would take its start from the moved clock)."""
    if not on:
        return
    import time
    import he_c
    shift = [0.]
    he_c.p2.perf_counter = lambda: time.perf_counter()+shift[0]
    real = he_c.p2.optimize_p2

    def timed(*args, **kwargs):
        result = real(*args, **kwargs)
        if not wall_info:
            now = time.perf_counter()
            shift[0] = kwargs['started']+900.5-now
            wall_info.update(started=kwargs['started'], pass0_end=now, pass0_end_minus_started=now-kwargs['started'], shift=shift[0])
        else:
            wall_info.setdefault('later_started', []).append(kwargs['started'])
        return result
    he_c.p2.optimize_p2 = timed


wall_info = {}


def probe_of(kind):
    """run_pair's unit probe (HE2 review minor 8): None, the real one; 'stop' a tag unit whose systemd reads OOMPolicy=stop; 'small' one whose kernel limit is 8 GiB; 'unread' a tag unit that could not be read."""
    if not kind:
        return None
    import unit_probe

    def fake(when):
        r = dict(schema=unit_probe.SCHEMA, when=when, unit='tag-he-test.service', in_tag_unit=True, peak_bytes=2**30, max_bytes=12*2**30, swap_max_bytes=4*2**30, events=dict(oom_kill=0), unread={},
                 systemd=dict(Id='tag-he-test.service', MemoryMax=str(12*2**30), MemorySwapMax=str(4*2**30), OOMPolicy='stop' if kind == 'stop' else 'continue'), systemd_unread=None)
        if kind == 'small':
            r['max_bytes'] = 8*2**30
        if kind == 'unread':
            r.update(systemd=None, systemd_unread='读不到: test', max_bytes=None, swap_max_bytes=None)
        return r
    return fake


def bump(grid, amplitude):
    """u_x = amplitude exp(-r^2 / (2 1.5^2)) at the centre, zero faces: at 3 a small ring of folded cells (d u_x / d x down to about -1.2), at .3 none."""
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in grid.shape], indexing='ij')
    centre = [n//2 for n in grid.shape]
    u = np.zeros(grid.shape+(3,))
    u[..., 0] = amplitude*np.exp(-((z-centre[0])**2+(y-centre[1])**2+(x-centre[2])**2)/(2*1.5**2))
    u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1] = 0, 0, 0, 0, 0, 0
    return u


class AbsentLauncher:
    """Configuration C cannot run (no unit, no device): the run directory gets an exit code and no records; classify_run says so."""
    kind = 'absent'

    def __call__(self, run, short):
        (Path(run)/'exit_code').write_text('1\n')
        (Path(run)/'run.log').write_text('Traceback (most recent call last):\nRuntimeError: no device for configuration C (test)\n')
        return dict(kind=self.kind, returncode=1)


class HandoverLauncher:
    """A frozen run that ends as D0's handover_only class (contract C22.12 implementation gate, revision 1 review important 4): exit 0, completion.json, worker_completion.json, summary.json with the role of a
    certified native initializer and no native commit, levels.json with a level-3 level_prepared event whose transfer certificate passes, the seed file; no final_native.npz.  bad_seed: the seed's field is
    folded (its recorded certificate says otherwise): the supervisor's own certificate refuses it and the identity follows."""
    kind = 'handover'

    def __init__(self, bad_seed=False):
        self.bad_seed = bad_seed

    def __call__(self, run, short):
        from jd_curl.three_d.grid import make_grid_3d
        run = Path(run)
        task = fc.task_dir(run)
        task.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(next((run/'inputs').iterdir()).joinpath('manifest.json').read_text())
        nz, ny, nx = manifest['shape_zyx']
        grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., unit='mm')
        u = bump(grid, 3. if self.bad_seed else .3)
        np.savez(task/'level_3_level_prepared.npz', u=u)
        proof = dict(status='certified_feasible', min_jd_lower_bound=.5, n_infeasible=0, n_uncertain=0, state_hash=he.array_sha256(u))
        (task/'levels.json').write_text(json.dumps([dict(event='level_prepared', level=3, path='level_3_level_prepared.npz', sha256=fc.sha_file(task/'level_3_level_prepared.npz'), transfer_certificate=proof)]))
        (task/'summary.json').write_text(json.dumps(dict(status='native_budget_exhausted', reason='handover only (test)', metrics_role='certified_native_initializer_NOT_native_outer_commit', native_commits=0, success=False)))
        (task/'worker_completion.json').write_text(json.dumps(dict(complete=True)))
        (run/'completion.json').write_text(json.dumps(dict(status='native_budget_exhausted')))
        (run/'started.json').write_text('{}')
        (run/'run.log').write_text('handover only (test)\n')
        (run/'supervisor_started_utc.txt').write_text('2026-11-06T00:00:00Z\n')
        (run/'supervisor_finished_utc.txt').write_text('2026-11-06T00:01:00Z\n')
        (run/'exit_code').write_text('0\n')
        return dict(kind=self.kind, returncode=0)


def launcher_of(spec):
    kind = spec.get('launcher')
    if kind == 'cpu_control':
        import configc_control
        return configc_control.CpuControlLauncher(Path(world['paths']['cache'])/'frozen', outer_steps=3)
    if kind in ('handover', 'handover_bad_seed'):
        return HandoverLauncher(bad_seed=kind == 'handover_bad_seed')
    return AbsentLauncher()


def fused_repair_field(mode):
    """repair.repair_field with a fake clock (the repair process only; repair.py unchanged): 'after_round' -> the wall right after round 0's certificate; 'mid_round' -> the wall after the 10th reading of the
    clock (t0, the round's start, then one per inner step: the inner loop stops at 'wall' after 8 steps, a field that passes, measured on the bump fold)."""
    import repair as rp
    real = rp.repair_field

    def fused(u, grid, cells, config, *, certify, **kwargs):
        state = dict(certificates=0, clock=0)

        def counted(*a, **k):
            state['certificates'] += 1
            return certify(*a, **k)

        def clock():
            state['clock'] += 1
            if mode == 'after_round':
                return 1e9 if state['certificates'] >= 2 else 0.
            return 1e9 if state['clock'] > 10 else 0.
        return real(u, grid, cells, config, certify=counted, **dict(kwargs, clock=clock))
    return rp, real, fused


class Scripted(run_hx.InProcessChild):
    """The in-process children, with the T88 child replaced when the spec says so (a folded field written as the T88 child writes its own: status done, oracle false, the field's hashes), and the repair
    child's clock replaced for the fuse."""

    def __init__(self, folded, fuse=None, t88_segmenter=None):
        self.folded, self.fuse, self.t88_segmenter = folded, fuse, t88_segmenter

    def __call__(self, stage, ctx):
        if stage == 't88' and self.t88_segmenter:                      # HE2 review minor 6: the T88 child's own start check, the supervisor's having passed (its settings changed in the child only)
            child = run_hx.read_context(run_hx.context_path(ctx))
            child['settings'] = dict(child['settings'], segmenter=dict(child['settings']['segmenter'], **self.t88_segmenter))
            return run_hx.t88_main(child)
        if stage == 't88' and self.folded:
            import repair
            pin = run_hx.load_images(ctx)
            out = run_hx.started(Path(ctx['out'])/'t88', ctx, 't88')
            u = bump(pin.grid, 3.)
            np.savez(out/he.FIELD_FILE, u=u)
            he.write_json(out/'run.json', dict(schema=he.SCHEMA.format('t88'), pair=ctx['pair'], status='done', reason=None, oracle=False, field=dict(file=he.FIELD_FILE, sha256=core.sha_file(out/he.FIELD_FILE), u_sha256=he.array_sha256(u)),
                                              data=dict(pair_sha256=pin.provenance['pair_sha256']), seconds_task=0., role='a folded field written by the test in place of the T88 child', repair_module=repair.ROLE))
            return 0
        if stage == 'repair' and self.fuse:
            rp, real, fused = fused_repair_field(self.fuse)
            rp.repair_field = fused
            try:
                return super().__call__(stage, ctx)
            finally:
                rp.repair_field = real
        return super().__call__(stage, ctx)


def review_of(spec, real):
    """The evaluation's certify: real, or with the review changed so that it differs from cert.json (C22.3-9 (d))."""
    mode = spec.get('review')
    calls = []

    def certify(u, grid):
        calls.append(he.array_sha256(u))
        r = real(u, grid)
        if mode and len(calls) == 1:
            from dataclasses import replace
            return replace(r, min_jd_lower_bound=(r.min_jd_lower_bound or 0.)*.5)
        return r
    return certify, calls


def third_of(spec, ctx, base_certification, launcher, order):
    """The third certification: in this process with the spies on again (it reads no label); 'mismatch_fail' makes its certificate fail."""
    def third(_ctx):
        order.append('third')
        certify, config = base_certification
        if spec.get('review') == 'mismatch_fail':
            from dataclasses import replace
            fail = lambda u, grid: replace(certify(u, grid), status='certified_infeasible', n_infeasible=1) if he.array_sha256(u) == first_hash[0] else certify(u, grid)     # noqa: E731
        else:
            fail = certify
        chosen = launcher
        if spec.get('third_launcher') == 'refused':                     # HE2 review minor 4: a setting third_launcher refuses, chosen before the output is renamed
            if refusals:                                                # once: the evaluation's later rounds find the same refusal (third_certificate.json is written once)
                return 'refused'
            refused_ctx = dict(ctx, settings=dict(ctx['settings'], units=dict(ctx['settings']['units'], third_certificate_fallback='direct')))
            try:
                with Spies():
                    return run_hx.third_cert_main(refused_ctx, launcher=None, certification=(fail, config))
            except SystemExit as exc:
                refusals.append(str(exc))
                return 'refused'
        if spec.get('third_launcher') == 'default':                     # HE2, H3: third_cert_main's own choice (run_hx.third_launcher), UnitLauncher replaced by a recorder that starts nothing
            class RecordingUnit(AbsentLauncher):
                def __init__(self, cache, unit_prefix='tag-he-c'):
                    self.unit_prefix = unit_prefix
                    made.append(unit_prefix)
            run_hx.fc.UnitLauncher = RecordingUnit
            chosen = None
        with Spies() as spies:
            rc = run_hx.third_cert_main(ctx, launcher=chosen, certification=(fail, config))
        third_spy.update(spies.record(), opened=spies.opened)
        return rc
    return third


made, refusals = [], []


first_hash, third_spy = [], {}
ctx = run_hx.make_context(world['stage_dir'], world['pair'], spec['out'], device='cpu', settings=settings_of(spec), paths=world['paths'], launcher='direct', check_identity=False, workers=1)
launcher = launcher_of(spec)
certification = run_hx.certifier(ctx)
if spec.get('identity_fails'):                  # HB, C22.5-6 (a): the identity's certificate made to fail (only the all-zero array): run_pair's implementation error
    _real_certify, _config = certification

    def _identity_fails(u, grid):
        from dataclasses import replace
        r = _real_certify(u, grid)
        return replace(r, status='certified_infeasible', n_infeasible=1) if not np.any(u) else r
    certification = (_identity_fails, _config)
install_tissue_fault(spec.get('tissue_fault'))
install_reader_fault(spec.get('reader_fault'))
install_wall_after_pass0(spec.get('wall_after_pass0'))
spies = Spies()
spies.install()
error = None
try:
    child = run_hx.SubprocessChild() if spec.get('child') == 'subprocess' else Scripted(spec.get('folded_t88'), spec.get('fuse'), spec.get('t88_segmenter'))
    rc = run_hx.run_pair(ctx, child=child, launcher=launcher, certification=certification, unseal_token=spec.get('unseal_token'), probe=probe_of(spec.get('unit_probe')))
except Breach as exc:
    rc, error = 'breach', str(exc)
finally:
    spies.remove()
result = dict(rc=rc, error=error, original_access=spies.original_access, opened=spies.opened, spy=spies.record(expect_reads=spec.get('expect_reads', True)), wall=dict(wall_info))
out = Path(spec['out'])
line = he.read_json(out/he.OUTPUT_DIR/'line.json') or he.read_json(out/'line.json')
result['line'] = line
if spec.get('evaluate') and rc == 0:
    order = []
    from bayesreg.eval import truth as bench_truth
    real_load_truth = bench_truth.load_truth

    def load_truth(*args, **kwargs):
        order.append('truth')
        return real_load_truth(*args, **kwargs)
    bench_truth.load_truth = load_truth
    review, calls = review_of(spec, certification[0])
    first_hash.append(he.array_sha256(he.read_output(out/he.OUTPUT_DIR)[0]))

    def recording(u, grid):
        order.append('review')
        return review(u, grid)
    result['evaluate_rc'] = run_hx.evaluate_main(ctx, certification=(recording, certification[1]), third=third_of(spec, ctx, certification, launcher, order))
    result['order'] = list(order)                                                              # the first evaluation's; the later calls below append to `order`
    result['third_spy'] = third_spy
    result['third_units_made'] = list(made)
    result['third_refusals'] = list(refusals)
    result['output_after_third'] = sorted(p.name for p in out.iterdir() if p.name.startswith('output'))
    result['evaluation'] = he.read_json(out/'evaluation.json')
    result['line_after'] = he.read_json(out/he.OUTPUT_DIR/'line.json')
    result['evaluate_again_rc'] = run_hx.evaluate_main(ctx, certification=certification)         # once per pair
    import shutil
    shutil.move(str(out/'evaluation.json'), str(out/'evaluation_moved_by_the_test.json'))       # as if it had crashed after the truth: the marker stays
    result['evaluate_after_marker_rc'] = run_hx.evaluate_main(ctx, certification=certification)
    result['evaluate_released_rc'] = run_hx.evaluate_main(ctx, certification=certification, lead_release='test: the lead releases a second read (2026-11-06)')
    result['release'] = he.read_json(out/'evaluation_release_0.json')
    result['evaluation_released'] = he.read_json(out/'evaluation.json')
    result['order_all'] = list(order)
result_path.write_text(json.dumps(result, default=str))
