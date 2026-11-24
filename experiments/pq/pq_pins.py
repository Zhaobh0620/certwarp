"""PQ (contract C22.Q, revision 5 adopted by BZ 2026-11-07; runs/20261107-pq/plan.md, final revision 3): the pinned values of the package.  Nothing here reads a file.

The preregistration is pinned by its sha256 (PLAN_SHA256, the TTA way: tools/segprobe/flip_tta.py PLAN_SHA256); every unit checks it before it reads anything else.  The tables below are plan.md section 1 copied
(the generator parsed the plan's code blocks; tests/test_bayesreg/pq/test_pq_pins.py parses the plan again and compares every value):
  P3        19 pairs: the T88 directory / the repair directory, repair.npz's sha256, the state_hash of v (the top-level certificate of repair.json), field_0600.npz's sha256 (= source.field_sha256 = run.json
            field.sha256) and repair.json's sha256 (sha256sum of the file);
  RUN_JSON  the sha256 of the 19 T88 run.json (m3 reads labels_sha256 from them);
  V4        the 20 development subjects: probs36.npz and structure_groups.npz (provenance output_sha256) and provenance.json (sha256sum).
The anchors of C22.Q1-1 (C22.9-4 (xv)): unet35_v4_ens, all_folds_mean, the registry configs/bayesreg/segmenters_hs_v1.json with sha256 b7682421...; the seg4 table with sha256 273743b9... (C22.Q2-6).
The member files (D91 (a), C22.Q1-3 "only these 4 files, not their directory"): the four registered members of unet35_v4_ens in HS's order A0 A1 B0 B1, by path and sha256 (the registry's
member_model_sha256 and the provenance's members agree; plan.md section 1.2).
"""
from collections import namedtuple
from pathlib import Path
from types import MappingProxyType

PLAN_REL = 'runs/20261107-pq/plan.md'
PLAN_SHA256 = '4c2ac725be266ac1f0312b05096a3913dc62f9be974862c34e8808c4ee43742d'      # the final preregistration (revision 3, BZ 2026-11-07 "A按建议"); another plan needs this constant changed and reviewed
CONTRACT = 'docs/bayesreg/01_合同.md C22.Q (revision 5, adopted by BZ 2026-11-07; D87 = (c), D88 = (a), D89 = (a), D91 = (a), D92 = (a), D93, D94 = (a))'

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')      # data/ is an untracked link of the main checkout; a worktree has none
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')        # F: of C22.0-2 (boundary_taper, preopt.py:617)
DATA_IMAGES = Path('/mnt/d/data')                                     # <subject>/aligned_norm.nii.gz (the members unit only)
STAGE_REL = 'data/three_d/stage_L2R_val_v1'
P3_ROOT_REL = 'p3_v1/runs'                                            # under RESULTS
PRED_ROOT_REL = 'seg/unet35_v4_ens/pred'                              # under RESULTS
MODELS_ROOT_REL = 'seg/unet35_v4_full'                                # under RESULTS
PQ_ROOT_REL = 'pq_v1'                                                 # the new root of this package (plan.md section 12)
TAPER_L = 4

ANCHOR = MappingProxyType(dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                               registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'))
SEG4_TABLE_REL = 'configs/bayesreg/seg4_of_seg35_v1.json'
SEG4_TABLE_SHA256 = '273743b9b396668d8f0539da33008f2a163b1d659ce64245f24314795c22fa4e'
ALLOWLIST_REL = 'configs/bayesreg/dev_subjects.json'
FOLDS_REL = 'splits/bayesreg_v1/folds.json'
DEV_SOURCE = 'stage_L2R_val_v1'
SEGMENTER_SCHEMA_CANDIDATE = 'UF4'                                    # experiments/hs/hs_core.CANDIDATES key of unet35_v4_ens

SUBJECTS = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 458))
SHORTS = tuple(f'{n:04d}-{n+1:04d}' for n in range(438, 457))          # pair i (i = 0..18) = subject i (fixed) and subject i + 1 (moving): the chain of block_interval


def pair_name(short):
    """'0438-0439' -> 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1' (the stage directory)."""
    f, m = short.split('-')
    return f'OASIS_OAS1_{f}_MR1__OASIS_OAS1_{m}_MR1'


def subjects_of(short):
    f, m = short.split('-')
    return f'OASIS_OAS1_{f}_MR1', f'OASIS_OAS1_{m}_MR1'


P3Row = namedtuple('P3Row', 't88_dir repair_dir repair_npz state_hash field_npz repair_json')
V4Row = namedtuple('V4Row', 'probs36 provenance structure_groups')

P3 = MappingProxyType({
    '0438-0439': P3Row('C_first5', 'RC_c600', repair_npz='8d856b7041cb64be4f19f07f51c40ca6a16f1cdaae9971fcbd803de57ae28edc', state_hash='f1324dbdda76b2648f706d76ad7e1ca74ca607927caa85776b05d7c93c6adfba',
                       field_npz='2bb5dcc2db23e5de415854f4f92987791d56cf3efce752e4b610050677f0b54a', repair_json='83bfa56178dfbf95214482e073760b27f0a83a931a5f1a5f40153cc1154e3e9c'),
    '0439-0440': P3Row('C_first5', 'RC_c600', repair_npz='59a363e33f784011bcff19871b762f288fb044b0167befdfa0f90913a0511760', state_hash='36bf6299edb020c568781d8360abe61083ed2116a5a483e91d7aae98a300cdf1',
                       field_npz='d50b1635d0c4e7d338ed2cd45fc1a7cc00e410527256e7a15f0e8a11deef72fb', repair_json='3035c9d2eb4d928fcbdea77d163418e8dcf0a5ec6de4f1789e700c7d28fad212'),
    '0440-0441': P3Row('C_first5', 'RC_c600', repair_npz='a428464771d3ef8ef9f4a8c01020f7521cec3a557d1041ea81e00c685085e5c2', state_hash='5bd45e4ee16053432905eeb6191495fdaa6cfffadbfa553bc52d5362e402935f',
                       field_npz='8375e20f4d8f6be4c607fe1876b061a8fd179dab29d5ca0b575508749c5b1160', repair_json='347758146ea09257cf65a3ea49011fb4f8859f824bee1cd1f9334dc00d28a47f'),
    '0441-0442': P3Row('C_first5', 'RC_c600', repair_npz='14a548c264bcd38dd13a4815fc8e27696d284d235994e27fc9da40af11fa88a2', state_hash='95f765ade7ee2de69a2b14fb6818eb3af52ffc76fbb48aeb879e2b72ccd1fd72',
                       field_npz='ac56135a6e0a4330f52d5c2c0ac3bcac9de032dac4689917e28a66d214625c9c', repair_json='10553d80b3ba1c3a29734b3e879d8c1699db7cfda197e34e480bef1f12883393'),
    '0442-0443': P3Row('C_first5', 'RC_c600', repair_npz='87fa608783cb1821c9cdf9b73c50613a37c0750debad43091331a2d6f51e80a9', state_hash='921cd7c2ce6387f2da497fbc4f11ba7288df84ed96748f018c3e8af470877bdd',
                       field_npz='bf9a0ebb382d4c703ceeb08bfcf57c8d92f7c677719cd473378898b14755ba48', repair_json='fffd968a23a106c072f3e602d47afba51adb5e05f25d12429866168c2775d491'),
    '0443-0444': P3Row('C_rest600', 'RC_rest600', repair_npz='e61ef3ea2c19c35907a7b5230d266967a96f8ca41b5ec2c4b0b5267a87be277e', state_hash='98a5220c2ff628bea7cb09ffc4d02d4254e90b0f6166de01a97dc6013f0c6c68',
                       field_npz='fec447c75c2c4b6836340d201b9c03d3fb969fb2b2b07ff43a358233067786e3', repair_json='9d9067357c82c6234c3a816779c89b2ea05c313c059b10ad08d40df6c45d9d03'),
    '0444-0445': P3Row('C_rest600', 'RC_rest600', repair_npz='f7513ff34ef2a5f940d18f0d99881181be711ef926faa385afae6f01f66a7a51', state_hash='a73375969edf7434f15bdbf3e8fad681717f9691d0f308b8fd88fb44cf6bcd72',
                       field_npz='62201f74cd78081b657b2ae56e023c994b411f4fc6c1b17a429714adc93fceba', repair_json='f9a1a9140c57991aea7dc01913e09871baf40f5b687374f7d5cb3f24e80475e0'),
    '0445-0446': P3Row('C_rest600', 'RC_rest600', repair_npz='105b42f1cfa8ce2d6b0753f8de7c96c3754b3497f4b14ba54750f3b87868f6d6', state_hash='6ec28d8cb13d9537f40bf62e60b98ddcdfd2137ba56d295f7d219a644cf527ee',
                       field_npz='105efcbf4b4624553510b380abc01d15e4b976b17aba094485ba2c3d0f48b4ff', repair_json='bf028313f22acb4275c40a5648b66cd4b5a58142cec7d12a1096654135d9c418'),
    '0446-0447': P3Row('C_rest600', 'RC_rest600', repair_npz='8b4bdf46e190ead35f1740da02db85ba52f0386f724eec3dc99b6eff0b9fac31', state_hash='6263eac64318ccd8555df758cb5e4965ff5133fce036d3df4b2a6a4baaa24d9f',
                       field_npz='205ea118bbb348c72ea488e0e514fe1c80be2b29496b411118581e57e28e4e32', repair_json='25b1dfd28f099da490813315070849b3ac015063e49d30d70f5f44c6934326e1'),
    '0447-0448': P3Row('C_rest600', 'RC_rest600', repair_npz='e7ebe9955d694f1c6784020a44762fb1c818a5a546dfad67cfcc0f90b47b61cf', state_hash='6233588ce7cb60c5db4c77c36e8d29d9a6546ba9d82f15bce90595b49d044b4b',
                       field_npz='6dbce77d985278010469c562e6ef35469f657dacea71c1b6dab8c23d5ecdea12', repair_json='0e13b65720f5b7fc7cdc7f51a190a380ad33f5a5d9fc5f37f12d7caf16a343d6'),
    '0448-0449': P3Row('C_rest600', 'RC_rest600', repair_npz='b50e3af9863f00e0b4e7e5adb3ffa5a4643e2618561230940331421509834211', state_hash='5ec88df5757f69894a60416caeefcfea4b56776fcf143ce6fc715cd72b7fa5e6',
                       field_npz='0b1082f331f32ab1a78a4aee0f6ae05957c6d83f7caf290db36d11bbfaf40735', repair_json='60c3178ef763b2b437e9c662aee48972243e58a81c558cf43826e35a1b468d89'),
    '0449-0450': P3Row('C_rest600', 'RC_rest600', repair_npz='3e6c48f81893a9141a1017a7f9da64f2eb31b4412065f808cc670c99767e53de', state_hash='09d0dda3999ef1e31c950fa8680aa927a90dfcecdca374304cdd6539be8b06a7',
                       field_npz='a0abf6b6dafea15ce1a70531aeef60cdb452b59e983036ad2febfacb5c4b0b37', repair_json='0f48fce305fab571a3ce225a1bf5e577b32bcc8b9447311e6e1ca311f0d00500'),
    '0450-0451': P3Row('C_rest600', 'RC_rest600', repair_npz='2a867ab4e8a924328cd5a095cf9f875b32ef49fa801a0a9c8d7f1c86a3f04306', state_hash='269d77f0fde1e4a702c64d8803b4664c58821e40b69cbe7b3b0f78b813c72004',
                       field_npz='3137cee54d8b3e1c41e0db3f7cfd9214c1c7286eee7352a10c0ac723c0122297', repair_json='75c1bb09089b597615473651022011b0155c31062c6774c566a8287bc7b2052a'),
    '0451-0452': P3Row('C_rest600', 'RC_rest600', repair_npz='58fd25d4884eeb711ef0307497caa58369db0e43a079b3f685d63d60e06cd384', state_hash='54165fc1610d83ec142ed7dca0e37a8d04bf7ee888eb6f59379fc952d14c116f',
                       field_npz='8066f6741df7031b75a8d959cfa00c9a2c89288ed4b18d0f594fb51af9772526', repair_json='210445740a7c49c6b6ba5eaa1e95137dcb7cacfbec0d1e10700e83c44d852fcc'),
    '0452-0453': P3Row('C_rest600', 'RC_rest600', repair_npz='47efa85db45fb8c2e2bff58187a5ab5107e1dacfa30460f302f39a9bc134085c', state_hash='0ebf86e2fc5a0cc7fc4a2c8f263246e976080f3b8ecffa8e37a7487d9d924d5c',
                       field_npz='a4897380cf2981298cc1cc4d5c68617901972f61c40dc03d93375c957cf40ebc', repair_json='777e419214f7710185240ea936082f10dd222ca8f100fdb1e2dafa33e8043f87'),
    '0453-0454': P3Row('C_rest600', 'RC_rest600', repair_npz='0dc4b1be99d31e5413d6e9a3e32ec63e034e6e8e4e79494158a5a4bc76870d5c', state_hash='200ffaa4b7438e4153cb99dea9cbeecb14dafc2362befff023c629a5db723f43',
                       field_npz='193a283888e5feadc4fe73b90f10a32dded2a61891be1dc8af8df4abbaa31e4d', repair_json='5c64d7a72316dd6cc7ad91e4cbe5d4da1271a57e0d4411215f3725d608693fdf'),
    '0454-0455': P3Row('C_rest600', 'RC_rest600', repair_npz='73ef65014b3a7f3ab724297a195d8861d1f6a3a46d8e03ef59c00cd97f1696b6', state_hash='6436f21df455aabe538cc5c43bf3c6ee394398c8b5f9ea3434e2abd7c2d94117',
                       field_npz='2caa32e6b9726936037d7b2f554ca6988a00df4e7d239bff4ee194e7d2182350', repair_json='414dbad1b2b9da91ee881e640f8610752ba87a0740f3313d0201c37cd5bfedda'),
    '0455-0456': P3Row('C_rest600', 'RC_rest600', repair_npz='72642dbefd976e56bb8d29a69fb67c482cafe9a35cab88f56e19f67d61f316d4', state_hash='355d3182a8f144c2295ea3eaaf984de23bfcd627cd5d04d5bae5d160c2e203de',
                       field_npz='652f9ce0597b7c0acd29248478431d3b7311418396a9bdf6d93424ba01d96700', repair_json='8c5488d0e0f3c7f6de493b1cc3c027739a47f394461e039950c2c4ee753e1d26'),
    '0456-0457': P3Row('C_rest600', 'RC_rest600', repair_npz='6ffa6b5cdde5367c64ff95b7eeaea77a452461855a95a0748fb0c27f21073df8', state_hash='763e359e30aa52faa29a6d8b002c38ad7e209aadcb087220a878a80e3e96052d',
                       field_npz='1525d9cf5e73a87fd7619184788af28f6b511bdd882a26b3d02922105aecbc5f', repair_json='fade16846d87debc5a1e089bf9b5cb4c09bb5e634ebd9fbc7daa9066c77011c8'),
})

RUN_JSON = MappingProxyType({
    '0438-0439': 'fd1bb09ce6ac12ce7397924f58a1efb3a2a2042c6135f7ff3d44ce681db77b74',
    '0439-0440': 'c03e9a9f26105ae6ff1372ae34c032e4781f2c69f8fc6409c04aee94b61e3f67',
    '0440-0441': 'a3ac803822bccfb70a9205da2fb4219172d62e9789f83a2ab85627c375d531e3',
    '0441-0442': 'f06ce4b28156e29a9830dda73cd56462b9651627d16c9f97fd5a913c7dd01d71',
    '0442-0443': '5c30d75cc3268ccace8aaf6aaabf053437deda72e1ba6f0b6f492a8e8e6c612a',
    '0443-0444': 'ddd0eb6618d3e5a62f1869af5d01bee9ac056a39a98d3345ed272a3a4fdc3c9f',
    '0444-0445': '22594a1d3392a3a4672f805eb905f4687b3a33776a5357163d0cbd6df297e87a',
    '0445-0446': 'db209600fbcbc4beeb184fb5a0bc0247e12a79bc88b305cead1f1cb8bcc772b2',
    '0446-0447': '47e55ee699df8006b1686a5f136c1b7ab6822fec67200a3f4a91374094bae200',
    '0447-0448': 'f2285e8e5cf247339b666996757628495c9b9cfe4d12253b9e5d3c25fafc3ce3',
    '0448-0449': '66c78293bb33a5dbc5c4162d964720b4bbd2884cfa4eafecab7798cdc19f9761',
    '0449-0450': '753d60357cb09d6ea1f17d6593dfc50706a9aecfe0d46985e21e17b04060c830',
    '0450-0451': '80efec1b109b36a3a167ef29b6320920ce46011475fa758f3e14e51a8bbd1e7c',
    '0451-0452': 'd861ae54bcf98017b747a1f935dfc94e9b82af42509aa4f0c0a31efd6f842756',
    '0452-0453': '0347b20c16b07a3eda30be536da6e263d5f38328400b75446d7af2167fe23b67',
    '0453-0454': '9e10704ec9fa5c261f72c08b223073faef4dff870166aa9a809a97e1e7720a62',
    '0454-0455': 'e2a0d0659e8c6b262217c30526c074a173b38252763a6d9205219816625da816',
    '0455-0456': '24ad25ec16dafd045cf29fc7d8c960e93c822831c46112c0139be7fce0047712',
    '0456-0457': '7c3a15daefa1e323efd0577ff14d52a60e11a35d9bf798f1d80580b9c1e2ee06',
})

V4 = MappingProxyType({
    'OASIS_OAS1_0438_MR1': V4Row(probs36='9bc2d287484220f6c21cfba5fb4377468da73e2331b8ec5067c99bedcdcd7299', provenance='a92523d233d4bb8edb722d57fb35427bb48e0a3f45b1228ffdc35681056e3c3b', structure_groups='d9e35bfcd016e6691f388813f801dea448eee528ad6b56f538c884bead6ae4b6'),
    'OASIS_OAS1_0439_MR1': V4Row(probs36='ef75c356cb25c9504a25a2c32fab0252ec8fcd3e4a48570dc3f1ab9f6ff3fef8', provenance='ef1e7dc4658ac5ed6844bff82bd3f43287497213871788d8f101ac0e2f32cc2b', structure_groups='ee40eaea0dd447b6be9b99a446333651e246a0c9386859f8bf9c08809b45ae2a'),
    'OASIS_OAS1_0440_MR1': V4Row(probs36='6a24dbda418741ddd08f8c1d90a80fec58eae7cdbe33165f802f5cc390007723', provenance='2906bc7ea8738268a5d88e518bb7d54d1b99fd97af7eb87cfa0ba4c8fe6d744e', structure_groups='bdd04951838cfa1f6d3cd53a41050d908329be4c2e561a57d78545065f8840d4'),
    'OASIS_OAS1_0441_MR1': V4Row(probs36='a9e42ef598f89598a1b79c2be258bfe1ef8f1b19a8db45302ef494b6591079e9', provenance='bb552acf6ee42972a838db9415bfc847407892079d0a4135f212bfaed5f6066d', structure_groups='8e8c52cdd036b9e32c8c58851e6073ad47a7450f9aff9a5960458c92caddf5fb'),
    'OASIS_OAS1_0442_MR1': V4Row(probs36='d4f87b2200101ee6be8ba67881c21ea8a47751c290a75101c636ddd4debaf864', provenance='27025c3edee57215fd34fcf89858cd5b96140ffa1d3ed0207c18a90350f232e9', structure_groups='c999da479cc0f974868d5d5767c810f69d772d9ed1eaf4d78b28fcd7960cd903'),
    'OASIS_OAS1_0443_MR1': V4Row(probs36='e2233d63a5d1bfee2753dab4747a0f2e9d3acff119af4ec0bcefd35b3dc05f94', provenance='a113c3dc20b9363988c3b7d854ecf62a933bacb01b382a51af881744f052715b', structure_groups='e34b20c716871c6613779ca2d25655ba783a2cdceda60b06c23ed29d59913b5d'),
    'OASIS_OAS1_0444_MR1': V4Row(probs36='517b1a1fa9fb78ea9f10ecb64214250e973bcf127e6d5eb2e07c3ad3d83fe408', provenance='d5d37d86bfaba5f101a588ec661b47ee1676e218b0a06a6dfd3d30028eb58442', structure_groups='9909a299f137729d2372165d3809cd6ef58de440e2b9f06a6f183dc029fb66f2'),
    'OASIS_OAS1_0445_MR1': V4Row(probs36='e1c4269f6f5addacdfee0fed5f7c8ce0c07eb06dc1c872a0f7d0624fe78d029e', provenance='24d14e1704bbdb1e8647c049d810348919dd898cdce548b1e36460e440d6cb63', structure_groups='291124b4a2e4005d351ac91cb83c6460f3dc068f87276f2f14db15c3eab3354b'),
    'OASIS_OAS1_0446_MR1': V4Row(probs36='bed2dfbef5d59024ed44c23bfcf3926d233b7506f668b5c7c5ece502f17ec463', provenance='8e98c82dcf079bfab1a7f51ac8e30221e349480d10e4b9f51aec22dcc6ad1bb4', structure_groups='2b4acd5882740295a51d4364e3aa80bf405aa12eb841ad073af3d73ac8c6ed44'),
    'OASIS_OAS1_0447_MR1': V4Row(probs36='08e2ee304b63162f068375edd61b624f784aa39afda65cba05e5a887ff713fa6', provenance='dcf4364bff836de6502d8c0f47a21ad585ef4e8e203cbbd19152bfd88d59ff72', structure_groups='2619ab5569ab170ff0cf2244d82bb87603c07711d9fae05bec7fdc2397f6fe55'),
    'OASIS_OAS1_0448_MR1': V4Row(probs36='17bf85eb194787db051d5df029d75ed1ec80dfb6ea7bdb84a9d554472864f428', provenance='00e72262616db3423e0ef602e5a25a900cd4ece87fc2851d9d268e5a683742bc', structure_groups='3eb376e4b2174e50d7045892ad0eaa29f4307a107dae3c7053c5253a71aaa36a'),
    'OASIS_OAS1_0449_MR1': V4Row(probs36='8b73e966bd2fe51aa81167c2861136538ec27857c1b707f7f8df911d2ae3e7c8', provenance='9a7b859a2c7641f88bc5e37365c37ba5a625b828ba1ac3d15f8f28aedcf1ff8e', structure_groups='1da5b836a6db2426a3e77a6fec28edc4ab23f4bf00486f50eaa9cb62c4fff6b8'),
    'OASIS_OAS1_0450_MR1': V4Row(probs36='ef57e799de1aa555f04d7e08175e96afc303a7ca591d53b6431cd0378ac4b8d1', provenance='5b4eb856cb02e1704b26350d7aa32fd75d2b9ec75207461f4781c59b3f6e3aaf', structure_groups='6432bf2dcac1b7102792acf8031c0e231cdb3cc93c835cc7c7ddde44d2164543'),
    'OASIS_OAS1_0451_MR1': V4Row(probs36='135875a6517de9a9cfcaa266cf9c7a3d107d231ae61795a650b50da0cf7f48d0', provenance='01a180f7657191e935344527351c724bcb36f5a358e3f85e2ba02144e74f3453', structure_groups='1dda0165d72d95583e63176c2631bcbc64c34c41b792e87593cbb067bda6c8ca'),
    'OASIS_OAS1_0452_MR1': V4Row(probs36='9d66da8642b31de25f38e1536843cba88c12b36d52252f3756510a0b500775ba', provenance='6d76896508d4f504ddda5da5f3e6079bd8e34abb645e81cecd8bb58913e8b69b', structure_groups='7dc587adc9ba6bbbe08c3fcc8272f92b4bba39f5ff523bdd8066e0ed15a2cf0c'),
    'OASIS_OAS1_0453_MR1': V4Row(probs36='a99fb9d91924b3ccc204db570ffe5ded6c3383077890b33120a02326d34e7d2f', provenance='27806e95e12f4d3522477c5dfd114cb80842405bd55c65ccfd20726476006a81', structure_groups='85657aa3cebb0ef11f4b73ec2c58f6960144cda693df2620b7155b0504b5ce7e'),
    'OASIS_OAS1_0454_MR1': V4Row(probs36='5e3b5fca918e34ab65703eeeb0ac7cdbd58753ef2f2fc021371ce9120ce30072', provenance='a68182d633f5775e9c62a004ff8f0cf470679cb93c94b254eabd196a1b0a8e4a', structure_groups='eee01297dc4c18e950e7f6c4489f9557a8c37f00f443960a9f58d41be02b6911'),
    'OASIS_OAS1_0455_MR1': V4Row(probs36='ed2695599a406c36dd6050357c6a441e4f9157a6aaa25f9eea5c9981a9f6f753', provenance='84a417bda1b4f6a0640f5c41f4d967dca44adf3579a69ca282b5ab8a22d8ea13', structure_groups='43bcb4ce6eac8c7604498be89f7fe534ab982a2f5fc53ee791848bce7f3a1678'),
    'OASIS_OAS1_0456_MR1': V4Row(probs36='9be1b7c1076b267dfaebc810566ab2ef571676d8e801a86cf70df1e0fd552f80', provenance='79748b3089204ad9200f09469f992e742d4404f529ac229b26027a19682a85a1', structure_groups='e690a587b3696bdbe5b0d0791b8385ac68eee6ff93ba14db6a4bdbd38d0df482'),
    'OASIS_OAS1_0457_MR1': V4Row(probs36='72c2db0170754720698e0b2da4e13e625470c9d9e3e89cae221d45ecc90fe63c', provenance='6fcf372099666ff0f1e3352973f2a74c21b858c8a482390e5d2e992113bdecdd', structure_groups='b34fde2400bbfc1c8fac8dadeba7b7b8952375fdcf9ce4c9e4cf2b25ae553554'),
})

# D91 (a): the four registered members of unet35_v4_ens (configs/bayesreg/segmenters_hs_v1.json member_model_sha256; the provenance's members list fold and seed), HS's order A0 A1 B0 B1
MEMBERS = (('unet35_v4_ens_foldA', 'foldA_s0', '52522d1f9cb6ad46f295542b94746d48f4f504a2db07b2dad1d7e4154d2852fd'),
           ('unet35_v4_ens_foldA', 'foldA_s1', '15dea2900bc9acc453a423bb4405345557dd67f627e813cfc2b38c3715da55cb'),
           ('unet35_v4_ens_foldB', 'foldB_s0', '28d8e7e142ca0efa0c340014a158dae2d65b7c4dae9423f7af8f82271378ee46'),
           ('unet35_v4_ens_foldB', 'foldB_s1', '181e8473b09a4ca7bdf1157096ea3036ed32184e4d14dd0e515e88da4ed447ea'))
