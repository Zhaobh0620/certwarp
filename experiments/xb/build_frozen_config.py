"""XB (runs/20261107-xb/任务书.md item 2): writes configs/frozen/holdout_B_v1_baselines.json once (it refuses if the file is there: configs/frozen/ only grows).

    python -I -B experiments/xb/build_frozen_config.py [--out PATH]

What goes in (prereg §2.3 table, §2.4, §2.6): every arm's executable entry (xb_core.ARMS, which the runner checks against this file), its version, configuration, source (file:line) and label
supervision as the preregistration writes them; the archive's drive.py, the four method scripts and the BL2 protocol by sha256; the 15 environment records by sha256 (the archive's README table);
per environment the records to compare and the installed files and binaries to hash; per baseline group the files under /home/rmcfz/opt/baselines and their sha256; the clones' commits.  The sha256 of
the baseline and installed files are the archived versions.json values, written here from those records (not from the machine); the runner hashes the machine's files against them.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xb_core as xc  # noqa: E402

ENV_RECORD_SHA256 = {                           # archive/seq_vp_3d_bl2_env_20261107/README.md (copied 2026-11-07, sha256 of source and copy equal)
    'ants/versions.json': '359b7c45ffafc2101fc29dddd20a7b4c75b8e93bcdbcd7945a0c4188cfc9b0a3',
    'bl_fireants/versions.json': 'a2772dbbb793cb860be8424aa442e10f0070523cfc22ff3c3fbc52bceb7c9557',
    'bl_fireants/conda_list.txt': '4f159d1e7309ec98375ad675730e77b322e0cf7f7cb22645a314be00a9e268b3',
    'bl_fireants/pip_freeze.txt': '7878baed80afbc04e82ee0f55d0532a23abcaaa18217decdf0d436af20fb4d4f',
    'convexadam/versions.json': '6ba43601c4bfb0727212ef0c3352b8fa238a1ec7a38c8808b10d227321f69a0e',
    'convexadam/conda_list.txt': '23ca40d6b774c0274c56391d8242ca7c550452bc0376b5e581661f82cb60e30a',
    'convexadam/pip_freeze.txt': 'bb151b16836311f22f9b316a8ae1967f5bb6cff8fbee642a38fb37f34da85744',
    'bl_unigradicon/versions.json': '77c8245ecee4ad564a2d889769aea19411de66079b6c9e089dec1e9d17b8933d',
    'bl_unigradicon/conda_list.txt': '3bb4b0715dbe6c83662875b155af7bc2e71af6845c152048d3ec4c7e21058c3a',
    'bl_unigradicon/pip_freeze.txt': 'b139e64ccb83f6f221ad3889014653b75c53e4efaa768eafd3bbb88e8ffa2ceb',
    'niftyreg/versions.json': 'c617607dc3eadb0abe8d777994d9fe202022eed4a992a3fcc35cc4ca8601820b',
    'deedsBCV/versions.json': '523bedd8ab5e55f90894aee123ce8ee832d0bb26ee619a2766f1566d92221217',
    'voxelmorph/versions.json': 'a8365986c9ee5304778dff479db9b48a35a28cca017316afe3b87159b6f41d02',
    'voxelmorph/conda_list.txt': '7ad3d68058aa0433aa2966581090599b642f34334ccd40c25aa27042bd9c70da',
    'voxelmorph/pip_freeze.txt': '7cbd2b612ab5b42c8d858fc3e0491bed73af4ad977e0e0039bb54e2b251b1188',
}
HARNESS_COPIES = {
    'm_fireants.py': 'b92e90e3b92ebdfbccba37b85a6116d01d621e0e6a197b1914dbbcbbf156fa6e',
    'm_convexadam.py': '04e62289cd633c3f27287c91c6cd8676bffb057194dd64f1590cb80e716fe5fc',
    'm_unigradicon.py': '5aded495abb1edbf6d3cecab2f68b37239c14aa59f14fde0877930a7560ce380',
    'm_voxelmorph.py': '41bd8fe0a08243db342b9dac8fefd7710b0b109d8bd16978ca93488be81da951',
}


def records(repo):
    d = repo/xc.ENV_RECORDS_REL
    return {name: json.loads((d/name/'versions.json').read_text(encoding='utf-8')) for name in ('ants', 'bl_fireants', 'convexadam', 'bl_unigradicon', 'niftyreg', 'deedsBCV', 'voxelmorph')}


def build(repo=xc.REPO):
    v = records(repo)
    fa, ca, ug, nr, de, vx, an = v['bl_fireants'], v['convexadam'], v['bl_unigradicon'], v['niftyreg'], v['deedsBCV'], v['voxelmorph'], v['ants']
    site = 'lib/python3.10/site-packages'
    environments = {
        'bl_fireants': dict(record='bl_fireants', python=fa['python'], conda_list='bl_fireants/conda_list.txt', pip_freeze='bl_fireants/pip_freeze.txt', versions='bl_fireants/versions.json',
                            files={f'{site}/fireants_fused_ops.cpython-310-x86_64-linux-gnu.so': fa['cuda_toolchain']['extension_so_sha256']}),
        'bl_convexadam': dict(record='convexadam', python=ca['python'], conda_list='convexadam/conda_list.txt', pip_freeze='convexadam/pip_freeze.txt', versions='convexadam/versions.json',
                              files={**{f'{site}/convexAdam/{Path(k).name}': h for k, h in ca['file_sha256'].items() if k.startswith('src/convexAdam/')},
                                     f'{site}/convexAdam/convex_adam_nnUNet.py': xc.V4SEG_NNUNET_SHA256}),
        'bl_unigradicon': dict(record='bl_unigradicon', python=ug['python'], conda_list='bl_unigradicon/conda_list.txt', pip_freeze='bl_unigradicon/pip_freeze.txt', versions='bl_unigradicon/versions.json',
                               files={f'{site}/unigradicon/__init__.py': ug['file_sha256']['src/unigradicon/__init__.py']}),
        'bl_synthmorph': dict(record='voxelmorph', python=None, conda_list='voxelmorph/conda_list.txt', pip_freeze='voxelmorph/pip_freeze.txt', versions='voxelmorph/versions.json', files={},
                              note='VoxelMorph reuses the env bl_synthmorph (archive/seq_vp_3d/results/verification/3d_stage_BL2_v1/README.md:16)'),
        'bl_niftyreg': dict(record='niftyreg', python=None, conda_list=None, pip_freeze=None, versions='niftyreg/versions.json', files={f'bin/{k}': h for k, h in nr['binary_sha256'].items()},
                            note='the archive has only versions.json: the binaries by sha256'),
    }
    baseline_files = {
        'fireants': {'fireants/src/fireants/scripts/pairwise/oasis.py': fa['authors_oasis_script']['sha256'], **{f'fireants/src/{k}': h for k, h in fa['file_sha256'].items()}},
        'ants': {f'ants/ants-2.6.5/bin/{k}': h for k, h in an['binary_sha256'].items()},
        'deeds': {f'deedsBCV/bin/{k}': h for k, h in de['binary_sha256'].items()},
        'convexadam': {**{f'convexadam/src/{k}': h for k, h in ca['file_sha256'].items()}, 'convexadam/src/src/convexAdam/convex_adam_nnUNet.py': xc.V4SEG_NNUNET_SHA256},
        'unigradicon': {'unigradicon/weights/unigradicon1.0/Step_2_final.trch': ug['weights']['sha256'], **{f'unigradicon/src/{k}': h for k, h in ug['file_sha256'].items()}},
        'voxelmorph': {'voxelmorph/weights_osf_mirror/vxm_dense_brain_T1_3D_mse.h5': vx['released_weights']['vxm_dense_brain_T1_3D_mse.h5']['sha256']},
    }
    clones = dict(fireants=dict(dir='fireants/src', commit=fa['repo']['commit']), convexadam=dict(dir='convexadam/src', commit=ca['repo']['commit']),
                  unigradicon=dict(dir='unigradicon/src', commit=ug['repo']['commit']), deeds=dict(dir='deedsBCV/src', commit=de['commit']))
    described = {
        'fireants_greedy': dict(version='fireants 1.5.1 @ f8a9cc25…, fused ops 1.2.0, env bl_fireants', configuration="the authors' OASIS script's test configuration: scales 4,2,1; iterations 250,200,100; Adam lr .25 (beta1 .1, scaledown False); CC kernel 7; grad sigma 1, warp sigma .25; compositive; the package's convergence test (1e-6, 10 iterations) by default",
                                source='BL2 protocol.md:27; env/bl_fireants/smoke_cmd.md:5-7, 12-24; harness/m_fireants.py:29, 52-55', label_supervision='none'),
        'fireants_syn': dict(version='as fireants_greedy', configuration='scales 4,2,1; iterations 200,100,50; lr .1; CC kernel 5; grad sigma .5, warp sigma .25', source='protocol.md:28; m_fireants.py:29, 57-58', label_supervision='none'),
        'ants_s': dict(version='ANTs 2.6.5 (fdce4d2) official binary', configuration='antsRegistrationSyN.sh -d 3 -t s -n 8, all defaults: rigid + affine MI, then SyN[0.2,3,0] with CC[...,1,2]; ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8; ANTS_RANDOM_SEED unset (not bitwise reproducible)',
                       source='protocol.md:29; env/ants/smoke_cmd.md:4, 9, 13, 21-23, 62; harness/drive.py:248-287 (command :257)', label_supervision='none'),
        'ants_so': dict(version='as ants_s', configuration='as ants_s, the SyN stage only (identity start)', source='as ants_s (smoke_cmd.md:15, 23)', label_supervision='none'),
        'niftyreg': dict(version='reg_f3d 1.5.69 (conda-forge 1.5.69.6)', configuration='defaults (NMI 64 bins, -sx -5, -be .001, -le .01, -ln 3, -maxit 150), OMP 8 threads', source='protocol.md:30; drive.py:290-327 (config :326)', label_supervision='none'),
        'deeds': dict(version='deedsBCV @ 00b4653', configuration='program defaults (alpha 1.6, 5 levels, G/L 8x7x6x5x4, Q 5x4x3x2x1), no affine', source='protocol.md:31; drive.py:330-382 (:382)', label_supervision='none'),
        'convexadam': dict(version='ConvexAdam @ b229e52 (package 0.2.0), env bl_convexadam', configuration='package defaults: MIND r 1, d 2; lambda 1.25; grid 6; disp_hw 4; Adam 80 iterations, grid 2; no smoothing; ic on; FP16; no mask; input in the L2R v1.1 layout',
                           source='protocol.md:32, 39-49; env/convexadam/smoke_cmd.md:5-28; harness/m_convexadam.py:21-28', label_supervision='none'),
        'unigradicon_io': dict(version='uniGradICON 1.0.6 (= @ 7678ceb), weights 1fcfa022…', configuration='default IO 50 steps, lr 2e-5, LNCC sigma 5', source='protocol.md:33; env/bl_unigradicon/smoke_cmd.md:5-11, 13-21', label_supervision='none (unsupervised)'),
        'unigradicon_noio': dict(version='as unigradicon_io', configuration='--io_iterations None', source='as unigradicon_io', label_supervision='none (unsupervised)'),
        'voxelmorph': dict(version='vxm_dense_brain_T1_3D_mse.h5 (8e5fe6bc…)', configuration='the released model, inference only', source='protocol.md:36; env/voxelmorph/smoke_cmd.md:8-10, 81-82', label_supervision='training data undocumented (image MSE)'),
        'fireants_greedy_eq': dict(version='as fireants_greedy', configuration='the 90 s tier: every scale\'s iterations x k, k = 22 (BL2 Amendment 6); the convergence test unchanged; no warm repeat', source='protocol.md:202-210; run_bl2val_eq_r3.sh:190', label_supervision='none'),
        'fireants_syn_eq': dict(version='as fireants_syn', configuration='the 90 s tier: k = 27 (BL2 Amendment 6)', source='protocol.md:202-210; run_bl2val_eq_r3.sh:190', label_supervision='none'),
        'unigradicon_io_eq': dict(version='as unigradicon_io', configuration='the 90 s tier: --io_iterations 123 (N*, BL2VAL README:16, 20)', source='BL2VAL README.md:16, 20', label_supervision='none (unsupervised)'),
        'convexadam_v4seg': dict(version='ConvexAdam @ b229e52, src/convexAdam/convex_adam_nnUNet.py sha256 9dab1091…, env bl_convexadam',
                                 configuration='convexAdam.convex_adam_nnUNet with its argparse defaults (lambda 1.25, grid_sp 6, disp_hw 4, selected_niter 80, selected_smooth 0, grid_sp_adam 2, ic True; convex_adam_nnUNet.py:166-172) on the two v4 argmax label maps (0..35; C22.1-4, the method channels\' probs36 and argmax rule), the class weights from this pair\'s maps (:25-36); input in the L2R v1.1 layout as m_convexadam.py; no self-configuring search',
                                 source='prereg §2.4; convex_adam_nnUNet.py:19-38, 41-159, 162-191', label_supervision='the features are a segmenter trained with labels (unet35_v4_ens, training set only): the same kind as the method',
                                 features=dict(xc.V4SEG_ANCHOR, rule='HI\'s provenance checked (segmenter_id, policy, registry_sha256, test_labels_seen false) and probs36\'s sha256 before the lock')),
    }
    arms = {}
    for arm, cfg in xc.ARMS.items():
        arms[arm] = {k: cfg.get(k) for k in xc.EXECUTABLE_KEYS}
        arms[arm].update(described[arm], family={'main': 'F1, F2 (prereg §4.4: the main family)', 'descriptive': 'F3 (described, not judged)', 'tier90': 'F3 (the 90 s tier, §2.6)'}[cfg['role']],
                         baseline_groups=cfg.get('baselines', []), clones=cfg.get('clones', []))
    return dict(
        schema='bayesreg.holdout_B_v1.baselines.v1', name='holdout_B_v1_baselines',
        written='2026-11-07, XB implementation subagent (branch xb-tools, runs/20261107-xb/任务书.md item 2); configs/frozen/ only grows: never edited',
        basis=dict(preregistration=xc.PREREG_REL, sections='2.3, 2.4, 2.6', adopted='BZ 2026-11-07 (docs/decision_log.md); BZ: "B1 和冻结求解器（配置 C）比也和外部基线（ANTs、FireANTs 等）比；B2 可以跑"',
                   rule='no tuning on the B-line hold-out, the 19 development pairs or T0: the configurations are paper 1\'s BL2 frozen ones (BL2 Amendment 3 items 3, 4)'),
        bl2_protocol=dict(path=xc.BL2_PROTOCOL, sha256=xc.BL2_PROTOCOL_SHA256),
        harness=dict(archive_dir=xc.ARCHIVE_HARNESS, drive_py_sha256=xc.ARCHIVE_DRIVE_SHA256, copies=HARNESS_COPIES, copy_dir='experiments/xb/harness'),
        env_records=dict(dir=xc.ENV_RECORDS_REL, sha256=ENV_RECORD_SHA256),
        environments=environments, baseline_files=baseline_files, baseline_root=str(xc.BASELINES), clones=clones, arms=arms, not_run=xc.NOT_RUN,
        unit=list(xc.UNIT_PROPERTIES), export_check=dict(threshold=xc.EXPORT_CORR_MIN, rule='drive.py check; corr < .999 is suspicious, not a failure; the third suspicious pair of an arm stops it (P19)'),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=str(xc.REPO/xc.FROZEN_CONFIG_REL))
    o = ap.parse_args(argv)
    out = Path(o.out)
    if out.exists():
        print(f'{out} exists: configs/frozen/ only grows; not written', file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(build(), indent=1, ensure_ascii=False)+'\n')
    print(f'wrote {out}: sha256 {xc.sha256(out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
