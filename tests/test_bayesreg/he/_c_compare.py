"""HE2: C in the HE entry (experiments/he/he_c.py) against run_p2's own code on a synthetic world, bit for bit (run by test_he_c.py with PYTHONPATH=<frozen>:<repo>; CPU only).

    python -B _c_compare.py SPEC_JSON RESULT_JSON

The same PairInput and the same predicted channels (the anchor's mark computed on them, as run_hx.t88_main and p2_core.ChannelSet.read_predicted do) go to:
  HE      he_c.tissue_of, he_c.three_passes (the T88 child's calls), and the C-off path of run_hx (one mc0_core.optimize_k)
  run_p2  run_p2.read_tissue, p2_core.tissue_levels, run_p2.run_field for the arm C with a stand-in batch (no reference, no queue); its run.json
The result: what is equal and what is not; the test asserts.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))

import numpy as np  # noqa: E402

spec = json.loads(Path(sys.argv[1]).read_text())
result_path = Path(sys.argv[2])
REPO = Path(spec['repo'])
sys.path.insert(0, str(REPO/'experiments/he'))
import run_hx  # noqa: E402

he = run_hx.he
core = he.core
fz = core.frozen_api()
import torch  # noqa: E402
torch.set_num_threads(1)
import fast_channels as fast  # noqa: E402
import fl_channels as channels  # noqa: E402
import he_c  # noqa: E402
import run_mc0 as run  # noqa: E402


def load_run_p2():
    s = importlib.util.spec_from_file_location('_p2_run_for_he2', REPO/'experiments/p2/run_p2.py')
    module = importlib.util.module_from_spec(s)
    sys.modules[s.name] = module
    s.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(json.loads(json.dumps(value, default=he._default)), sort_keys=True)


def same_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


world = spec['world']
settings = he.frozen_settings()
settings['t88']['pre'].update(spec['t88_pre'])
ctx = run_hx.make_context(world['stage_dir'], world['pair'], spec['out'], device='cpu', settings=settings, paths=world['paths'], check_identity=False)
pin = run_hx.load_images(ctx)
root = Path(world['paths']['atropos_root'])
predicted = fast.read_predicted(pin, unet=run_hx.unet_source(ctx), seg_root=root, allowlist=world['paths']['allowlist'], workers=ctx['workers'])
predicted.record.update(he_c.mark(he.CONTRACT_SEGMENTER['segmenter_id'], predicted.record['pair_keys_read'], predicted.record['sides']))
pre = settings['t88']['pre']
config = fz.PreoptConfig(**run_hx.tuples(pre))
ch = settings['channels']
spec_c = dict(k=ch['k'], channels=ch['channels'], weight=ch['weight'], D_factor=ch['D_factor'])
grids = fz.four_level_grids(pin.grid)

# HE: the T88 child's calls
t0 = perf_counter()
tissue = he_c.tissue_of(pin, root, fz.four_level_grids, fz.downsample_array, workers=ctx['workers'])
passes = []
u, record, status, reason, scale, scale_check, _ = he_c.three_passes(spec_c, pin, predicted, tissue, config, device='cpu', started=t0, prepare=channels.prepare_stacks, table=core.load_table,
                                                                     observer=lambda: run.vram_guard(torch, 'cpu', 9*2**30), downsample=fast.FastDownsample, scale_check_of=channels.scale_check_of,
                                                                     workers=ctx['workers'], passes=passes)
# HE with C off: run_hx.t88_main's other branch (one optimize_k)
stacks, _ = channels.prepare_stacks(spec_c, pin, predicted, core.load_table(), 'cpu', workers=ctx['workers'])
u_off, _ = core.optimize_k(stacks, pin.grid, config, device='cpu', started=perf_counter(), observer=run.vram_guard(torch, 'cpu', 9*2**30), workers=ctx['workers'], record_level_norms=True,
                           downsample=fast.FastDownsample())
del stacks

# run_p2's own code
RUN = load_run_p2()
RUN.mc.SEG_ROOT = root                                               # run_p2.read_tissue's default root (mc0p_core.SEG_ROOT) pointed at the world's
brain, ptilde, names, wm_index, index, record_p2 = RUN.read_tissue(pin)
levels_p2 = RUN.p2.tissue_levels(ptilde, brain, pin.grid, grids, fz.downsample_array, index, workers=ctx['workers'])
tissue_same = dict(brain=same_array(brain, tissue.brain), ptilde=same_array(ptilde, tissue.ptilde), names=names == tissue.names, wm_index=wm_index == tissue.wm_index, index=index == tissue.index,
                   record=canonical(record_p2) == canonical(tissue.record),
                   levels=len(levels_p2) == len(tissue.levels) and all(a.level == b.level and a.shape_cells == b.shape_cells and a.n_brain_cells == b.n_brain_cells and same_array(a.base, b.base)
                                                                      and same_array(a.csf, b.csf) and same_array(a.gm, b.gm) and same_array(a.brain_cells, b.brain_cells) for a, b in zip(levels_p2, tissue.levels)))
batch = SimpleNamespace(ARMS=RUN.p2.ARMS, arm_config=lambda cell, sigma_field: run_hx.tuples(pre), read_images=lambda cell: pin, RUN_CAP_SECONDS=pre['seconds'], steps_of=lambda cell: 600,
                        ROLE='HE2 comparison (tests/test_bayesreg/he/_c_compare.py)', PREFIX='p2',
                        channels=SimpleNamespace(read_predicted=lambda pin_, workers=1: predicted, segmenter_id=he.CONTRACT_SEGMENTER['segmenter_id']))
cell = SimpleNamespace(arm='C', run_id='he2_compare_C', tier='first5', pair=world['pair'], short=he.short_name(world['pair']), is_repair=False)
rc = RUN.run_field(SimpleNamespace(out=spec['p2_out'], device='cpu', code_sha='he2-compare'), cell, batch, RUN.BASES['p2'])
p2_run = json.loads((Path(spec['p2_out'])/'run.json').read_text())
rows = []
for k, (a, b) in enumerate(zip(passes, p2_run['passes'])):
    rows.append(dict(pass_index=k, u_sha256=a['u_sha256'] == b['u_sha256'], tissue=canonical(a.get('tissue')) == canonical(b.get('tissue')), estimate=canonical(a.get('estimate')) == canonical(b.get('estimate')),
                     scale=canonical(a['scale']) == canonical(b['scale']), levels=[lv['iterations'] for lv in a['levels']] == [lv['iterations'] for lv in b['levels']],
                     hooks=a['hooks'] == b['hooks'], cap_hits=a['cap_hits'] == b['cap_hits'], he_u=a['u_sha256'], p2_u=b['u_sha256']))
result_path.write_text(json.dumps(dict(status=status, reason=reason, p2_status=p2_run['status'], p2_rc=rc, n_he=len(passes), n_p2=len(p2_run['passes']), rows=rows, tissue_same=tissue_same,
                                       final_is_pass2=he.array_sha256(u) == passes[-1]['u_sha256'], off_is_pass0=he.array_sha256(u_off) == passes[0]['u_sha256'],
                                       ratios_used=[(r.get('tissue') or {}).get('ratios_used') for r in passes], estimates=[(r.get('estimate') or {}).get('ratios') for r in passes],
                                       field_moved=float(np.abs(u).max()), summary=he_c.summary(passes)), default=str))
