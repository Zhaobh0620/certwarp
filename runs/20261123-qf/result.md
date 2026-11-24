# QF：定性图 fig_qual（中心差分 det 全正，Q1 插值里有 J ≤ 0）

> 做图运行员，2026-11-23 约 4:44 PM CST 写（`TZ=America/Chicago date`：Mon Nov 23 16:43:53 CST 2026）。任务书 runs/20261123-midl/任务书_QF.md（main 42181230）。分支 qf。
> **独立复核 0/0/7，读数成立**（docs/manuscript/reviews/posthoc3_QF_复核_20261123.md，裁决在 docs/decision_log.md 2026-11-23）；本文与图已照 m1、m6、m7 改，见文末修订 1。开发集一对（stage_L2R_val_v1 的 0438-0439），每个场一份、单次；没有读真值标签，没有读评测记录，不是选择读数。
> 没有新跑任何配准：三个场都是现成的开发集输出。没有跑测试。

## 1. 读到了什么

整幅场的计数（中心差分 det 的体素总数 6,452,160，是四周各裁 2 个体素之后的；单元总数 6,772,287）：

| 面板 | 场 | 中心差分 det ≤ 0 的体素 | 中心差分 det 最小值 | 有见证点的单元 | 53 个采样点上 J 的最小值 | 只看 Jacobian 的证书（BL1 配置） | 证书的不可行 / 未决单元 | 证书最小下界 |
|---|---|---|---|---|---|---|---|---|
| (b) | ANTs SyN（ants_s） | 0 | .1163 | 12 | −.4833 | 不通过（infeasible，JACOBIAN_VIOLATION） | 12 / 0 | −.4833 |
| (c) | CertWarp 修复前 | 23 | −.2239 | 19,817 | −4.602 | 不通过（infeasible，JACOBIAN_VIOLATION） | 19,830 / 0 | −4.602 |
| (d) | CertWarp 修复后（认证终态） | 0 | .0478 | 0 | 1.488e-4 | 通过（certified_feasible） | 0 / 0 | 1.2386e-08 |

- (a) 是固定图像的切片，没有计数。
- CertWarp 的两个场另跑了配置 C（C07-2，PrescreenCertificateConfig，δ = 0，max_depth 48，max_boxes 255，显式传）的全证书：修复前不通过（不可行 19,830，未决 0），修复后通过（最小下界 1.2386e-08）。两份的 state_hash 与运行自己记录的相同：修复前 f5d944bc…（repair.json 的 initial_certificate，它也记 19,830），修复后 79e3d2a6…（output/cert.json 的判定证书）。
- ANTs 的场六面位移不为零（最大 8.75 mm），所以只有只看 Jacobian 的证书；采样域核对 ok。
- 见证点计数是下界：修复前的场，证书判不可行的 19,830 个单元里有 19,817 个在 53 个采样点上有 J ≤ 0，另 13 个没有；没有反过来的情况。ANTs 的 12 个见证单元与证书的 12 个不可行单元是同一批。
- fixed > 0 之内（不读标签的脑内代理；固定图像 > 0 的体素 / 左下角节点 fixed > 0 的单元，都是 2,098,710）：ANTs 0 体素、12 单元；修复前 23 体素、19,804 单元；修复后 0、0。
- ANTs 的 12 个见证单元里，只有 1 个在 Gauss27 点上有 J ≤ 0（1 个点），12 个都在 closed27 点上有（17 个点）。修复前：Gauss27 8,346 个单元（18,652 个点），closed27 19,817 个单元（45,668 个点）。
- ANTs 这 12 个单元的 8 个角点体素上，中心差分 det 在 .948 到 2.334 之间（opponent_cells.json）：中心差分在这些地方读到的是不变或膨胀。
- 官方 SDlogJ（这一对，只描述）：ANTs .0648，修复前 .0684，修复后 .0683。

逐场记录：records/counts_ants_s.json、counts_certwarp_pre.json、counts_certwarp_final.json；图上用的数：fig_qual_counts.json。

## 2. 场的来源（都是现成的，没有新跑）

| 场 | 文件 | sha256 | 出处 |
|---|---|---|---|
| ANTs SyN | /home/rmcfz/seq_vp_tag_results/xb_dev_v1/work/ants_s/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1/disp.npz（键 d，float32） | 3b3d85f4613c497b4738b762ab25986e62201f3198e72b6047c52a0c7c006a0e | D3 回归核对的新单元 tag-hb-x-ants_s-0438-0439，XB 运行器的开发集入口、预登记的冻结配置，代码 935a0e49（P0′），2026-11-15 5:33–5:36 PM CST；runs/20261107-xb/regression_result_part2.md 第 3.2 节记的 disp.npz 3b3d85f4… 与此相同 |
| CertWarp 修复前 | /home/rmcfz/seq_vp_tag_results/he_smoke_v3/a/0438-0439/t88/field_0600.npz（键 u），再做 boundary_taper（L = 4） | 文件 b5be9924cf48e1722a8008a2ec7b61385d1ed894819a4111f0a84eee8e9c53b2；u b6f061da…；tapered u f5d944bce23008b7f2c0f0e4ba0e211e4b145ea18176b1bdcadc78558919d2b6 | D4 冒烟 he_smoke_v3 的 a 臂，方法 Hx24-RF05-F1200-v4（= CertWarp），冻结设置 414eccd3…，代码 935a0e49，2026-11-15 约 6:08–6:12 PM CST（文件时间）；runs/20261115-d4/result.md。修复前的场照预登记修订 2 第 4 项与 experiments/hb/hb_prerepair.py 的定义：修复的输入 = boundary_taper(field_0600 的 u)，三项核对（文件 sha256、u 的 sha256、tapered u 的 sha256）对 repair/repair.json 的 source 都成立 |
| CertWarp 修复后 | 同一运行的 output/disp.npz（键 d，float64） | ce551d06ecda2d60a1ae5bbff0493f79664fe15efb815e695d219c06a0d23d4d | 同上；d 与 output/final.npz（4c8fe761…）的 u 逐位互换，sha256(u) = output/cert.json（92d37880…）的 state_hash 79e3d2a6… |
| 固定图像 | data/three_d/stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1/pair.npz | pair_sha256 照 manifest.json（832ed612…）由 load_pair 核对 | 经 bayesreg.io.pair.load_pair 读（只取 fixed、moving，另核 initial_u 全零），受试者守卫用 configs/bayesreg/dev_subjects.json（9e1f050d…） |

- 脚本里每个输入都钉了 sha256，对不上就退出。
- 这一对上还有三个对手的现成场（fireants_greedy、fireants_syn、convexadam_v4seg 的 disp.npz）：只算了文件的 sha256、列了 npz 的成员名，**没有读数组，没有算计数**（见第 4 节）。
- 为了核来源另读过的小记录：xb_dev_v1/<臂>/<对>/result.json（四个臂，只看 arm、status、pair、disp 的 sha256、export_check）；he_smoke_v3 的 batch.json、a 臂的 context.json、t88/run.json（键名与 field 一项）、repair/repair.json、output/cert.json、output/line.json。a 臂的 evaluation.json 用 json 读进内存后只打印了顶层键名，里面的数没有看也没有用。
- 没有读：任何留出集目录、/mnt/d、LPBA40、T0；pair.npz 的 seg 键；真值标签。

## 3. 怎么算

脚本 docs/manuscript/midl/figures/fig_qual.py（定义在它的 docstring 里）。

- 中心差分 det：bayesreg.eval.metrics.central_det(d)（Learn2Reg SDlogJ 用的那个行列式，加 3 之前；四周各裁 2 个体素）。
- 有见证点的单元：u = d_to_u(d)；bayesreg.geom.q1.tile_geometry 分别用 gauss27(grid) 与 closed27(grid)，每个单元 53 个不同的采样点上的 J（两套点各 27 个，共用体心），有一点 ≤ 0 就标记。
- 证书：bayesreg.eval.holdout.certify_field_bl1(u, grid)（评测里每个臂都用的调用；ParallelEnclosureCertificateConfig，δ = 0，max_depth 48，max_boxes 255，tile 8192；CPU FP64）。CertWarp 的两个场另跑 bayesreg.eval.cert_eval.certify_field(mode="full")（配置 C）。两处的配置都是显式传的。
- 约定的独立核对（describe）：对 ANTs 的 12 个见证单元，直接从 disp.npz 的 8 个角点值写出三线性插值的 J，不经 bayesreg.geom 与 jd_curl；53 个采样点上的最小值与存下的值最大相差 1.4e-08（存的是 float32）。
- 两遍：第一遍在脚本还没提交时算（缓存 /home/rmcfz/seq_vp_tag_results/qf_v1，4:26–4:32 PM CST）；第二遍从提交 a1395c69 的干净检出重算（缓存 qf_v2，4:36–4:41 PM CST），图用的是第二遍。两遍的 15 个数组文件逐字节相同（records/cache_sha256.txt）；三份记录除了耗时、时刻、代码身份与第二遍新加的两个字段（state_hash_equals_run_certificate、regions_note）之外相同；选中的层与窗口相同。
- 图由 a1395c69 的脚本从 qf_v2 画出；之后的提交 118abb36 只加了 describe（65 行，只有新增），再之后只改了 SELECT_RULE 末句的字符串（见第 4 节）；用那时最后的脚本（611f40a5）重画，PDF 与 PNG 逐字节相同。修订 1（09cf1704）改了印进图里的文字与图例的行数，图从同一份 qf_v2 缓存重画，见文末。

## 4. 挑选经过（如实）

- **看过几对**：1 对。ANTs（和 FireANTs）在开发集上的现成场只有 0438-0439 这一对（D3 回归核对），冻结代码下留着修复前场的 CertWarp 运行（he_smoke_v3 的 a 臂）也是这一对，所以没有在对之间挑。convexadam_v4seg 有 19 对的场，HS 有 19 对的修复前场，都没有看。
- **对手**：照任务书的次序先看 ANTs SyN。它在这一对上中心差分 det 全为正、又有见证单元，所以没有再算 ConvexAdam-Seg 与 FireANTs。
- **切片与窗口**：规则写在脚本的 SELECT_RULE 里。时序：先算了 ANTs 的场、列出它的 12 个见证单元（坐标、min J、角点的中心差分 det）；然后写下规则，那时 CertWarp 两个场的计数正在后台算，它们的输出我还没有读（此前只从 repair.json 知道修复前整幅有 19,830 个不可行单元，不知道在哪里）；规则写完之后才读 CertWarp 的计数、跑 select。规则没有在看到 CertWarp 的计数之后改过。records/selection.json 里记的规则末句是当时的写法（"before the CertWarp fields were counted"），不够准确，脚本里已改成上面这个说法（只改了这一句字符串）。
  - 候选是含 ANTs 见证单元的每一层单元（两张相邻节点面之间的一层），三个方向共 19 层（横断 7、冠状 5、矢状 7）；窗口是以该层里 ANTs 见证单元的包围盒为中心的 W × W 个单元，W 取 24、32、48 里第一个可行的。
  - 取 ANTs 见证单元最多的层（a = 3，有 4 层：冠状 y = 136、155，矢状 x = 107、108），其中取同一窗口里修复前见证单元最多的（b = 4、17、31、13）。
  - 选中：矢状层 x = 107–108，窗口 z 140–163、y 124–147（24 × 24 个单元，图上是 25 × 25 个体素），a = 3，b = 31，修复后 0，窗口节点里 fixed > 0 的占 .92。全表（19 层 × 3 个窗口尺寸）在 records/selection.json。
  - 所以图上的 3 个单元是 12 个里的一簇（12 个分三簇：2、5、5 个）。
- **窗口里的中心差分 det**（节点面 x = 107）：ANTs .238–2.361，修复前 .165–3.298，修复后 .204–3.267，三个场在窗口里都没有 ≤ 0 的体素。修复前整幅有 23 个，都不在这个窗口里，所以图上没有画"体素 det ≤ 0"的标记（脚本里留着这种标记，有才画）。
- **看过的图**：固定图像三个方向的中间切片各一张（定方向用，不含场）；选中窗口的图三稿（第一稿标题与计数行重叠、方框用白色衬底把格子盖住了；第二稿标题仍相碰；第三稿即定稿，另转了一份灰度看）。没有渲染别的层或别的窗口。

## 5. 图

- docs/manuscript/midl/figures/fig_qual.pdf（26492b11…）与 fig_qual.png（8a73281a…，300 dpi）。宽 6.0 in（MIDL 的 \textwidth），高 5.77 cm，字号 7 pt，DejaVu Sans（PDF 内嵌子集）。修订 1 之前的那一版是 06d80d20… / b158386b…，高 5.42 cm。
- 灰度可读：底图本来就是灰度；单元用"浅色方框加黑色衬边"，形状与明暗都能认，不只靠颜色（转灰度看过）。
- 图里与两个图文件里都没有对名或受试者号。
- 中心差分 det 的灰度范围 0–2.5，更大的截断（色标上端的箭头）。
- 面板下三行是整幅场的计数（中心差分那一行不含四周 2 个体素，图里的脚注写明）；"J ≤ 0: … cells"写的是有见证点的单元数（下界），不是证书的不可行单元数。

### 图注草稿（英文；数字出自 fig_qual_counts.json 与 opponent_cells.json）

> **Figure X.** Voxel-grid Jacobian statistics can miss non-positive Jacobians of the interpolated map (one development pair, selected for illustration; one field per method). (a) Sagittal slice of the fixed image; the box marks the 25 × 25-voxel region shown in (b)–(d). (b)–(d) Gray scale: determinant of the central-difference Jacobian of the displacement at the voxels of this slice, the quantity behind SDlogJ. Squares: cells between this slice and the next in which the trilinear (Q1) interpolant of the exported displacement has J ≤ 0 at one of 53 sample points per cell (the 27 Gauss points and the 27 points of {0, ½, 1}³, which share the cell centre). (b) ANTs SyN: no voxel has a non-positive central-difference determinant (2-voxel border excluded, as for SDlogJ; minimum 0.12), yet 12 cells contain a sample point with J ≤ 0 (minimum −0.48; at the corner voxels of these cells the central-difference determinant is between 0.95 and 2.33), and the Jacobian-only certificate fails. (c) CertWarp before local repair: 23 voxels (none in the region shown) and 19,817 cells; the certificate reports 19,830 cells that contain a point with J ≤ 0. (d) CertWarp output: no such voxel or cell, and the certificate passes on all 6,772,287 cells. The numbers under the panels are whole-volume counts (voxels: 2-voxel border excluded; cells: all); cells found by sampling are a lower bound on the cells in which the interpolant has J ≤ 0 somewhere.

稿件里用的时候：这是开发集一对、每个场一份的观测，不能写成各方法的一般性质；ANTs 的命令不设随机种子，D3 回归核对里同一命令的两次输出不逐位相同（最大差 1.32 体素），"12 个单元"只是这一份场的数。

## 6. 偏离与没做成的地方

1. **脚本放在 docs/manuscript/midl/figures/**（任务书没写位置；与图放在一起）。这个目录不在 SG 静态普查（tests/test_bayesreg/sg/_census.py 的 SCOPE：bayesreg、experiments、scripts、runs）之内。脚本读 pair.npz 走的是 load_pair 与开发集白名单。要挪进 runs/ 或 experiments/ 的话，普查要补行；交总负责人定。
2. **boundary_taper 用的是带出处的复制**（eb808be jd_curl/three_d/preopt.py:617–624，本仓库在 archive/ 里），不是 mc0_core.frozen_api（那要把冻结树放到 sys.path 最前面）。复制只在 tapered u 的 sha256 等于 repair.json 的 source.tapered_u_sha256 时才被接受，这次相等。
3. **算了两遍**（见第 3 节）；第一遍的缓存 qf_v1 留着（209 MB），第二遍 qf_v2（209 MB）是图的来源。都在 /home/rmcfz/seq_vp_tag_results/ 下，不进 git。
4. 证书记录里按区域分的"brain"在这里是 fixed > 0，不是评测里的 seg4 > 0（没有读标签）。
5. 计数没有放进 systemd 单元：只是对现成场的事后计算（每个场约 1.5–2 分钟，nice 19，NUMBA_NUM_THREADS = 4），没有新跑配准，没有用 GPU。计算期间机器上另有一个 nice 19 的 CPU 进程（另一项工作的 posthoc_minj.py），没有 tag-* 或 seq-vp-* 单元。这里不报任何耗时结论。
6. 没有跑测试；没有改项目状态记录、docs/decision_log.md、runs/INDEX.md；没有 push。
7. 稿件正文与 figures_plan.md 没有动；图注只是草稿。

## 7. 文件

- docs/manuscript/midl/figures/：fig_qual.py、fig_qual.pdf、fig_qual.png、fig_qual.md（图旁的说明）。
- runs/20261123-qf/：result.md（本文件）、fig_qual_counts.json、opponent_cells.json、records/（三份 counts、selection.json、cache_sha256.txt、logs/）。
- 重画：`python -I -B docs/manuscript/midl/figures/fig_qual.py plot --out <目录>`（读 qf_v2 的缓存）；缓存没了就先 `count --field ants_s --cells`、`count --field certwarp_pre --cells`、`count --field certwarp_final --cells`、`select`（要 `--cache` 指一个新的 qf_* 目录，旧的不覆盖）。

## 修订 1（2026-11-23 约 6:33 PM CST，`TZ=America/Chicago date`：Mon Nov 23 18:33:03 CST 2026）

依据：独立复核 docs/manuscript/reviews/posthoc3_QF_复核_20261123.md（0/0/7，计数都成立）的 m1、m6、m7，裁决在 docs/decision_log.md 2026-11-23 最后一行；总负责人的指示。分支 qf 先并入 main（534d5ddb，快进：main 已含 611f40a5）。**没有重新计数，没有读新的文件**；图从现成的 qf_v2 缓存重画。上文已照下面三条改过，改之前的写法在 611f40a5。

- **m1（53 个采样点，不是 54 个）**：Gauss27 与 closed27 共用体心，每个单元是 53 个不同的点（27 + 27 次求值）。改了：fig_qual.py 印进图例的那一行与 docstring、注释；本文第 1 节表头、第 1 节第 4 条、第 3 节两处；图注草稿；fig_qual.md。计数不变：体心在两套点里是同一个点，"有一点 ≤ 0"的单元集合与 54 次求值的并集相同。缓存文件 minj54.npy 与 opponent_cells.json 的键 min_J_54_points、min_J_54_points_independent 的名字没有改（describe 没有重跑）；records/ 下的记录与日志是当时的原样。
- **m6（中心差分的计数不含四周 2 个体素）**：图里的脚注改成两行，"Under (b)–(d), whole-volume counts: voxels with central-difference det ≤ 0 (2-voxel border excluded); / cells marked as above (all cells); cell-wise certificate of J > 0."，图例因此多一行，图高从 5.42 cm 变成 5.77 cm（仍在 5–6 cm 内），宽不变。图注草稿两处照复核的改法补上。
- **m7（图注措辞）**：标题句 "do not describe the interpolated map" 改成 "can miss non-positive Jacobians of the interpolated map"；末句 "the cells where the interpolant folds" 改成 "the cells in which the interpolant has J ≤ 0 somewhere"；"the same displacement" 改成 "the exported displacement"。fig_qual.py docstring 里同一处的 "folds" 也照改。
- **图**：重画用提交 09cf1704 的脚本（脚本 sha256 f721b28d…，干净检出），只换了字符串与图例的行数，面板的画法与数没有动；fig_qual_counts.json 重写了一次，与之前只差图高、代码身份、时刻。用 Read 看过新的 PNG：三行图例不挤、没有重叠。

| 文件 | 修订 1 之前（611f40a5） | 修订 1（现在） |
|---|---|---|
| fig_qual.pdf | 06d80d20d0d2930994268cd7fa5ea2b9a0e05192e43e8ab46a2ea2cc78f5a9e3 | 26492b11bfcf4ecfcf628e363be6a2a9aa1af59993eba2fe4a45a99b4b96b74f |
| fig_qual.png | b158386bc4e9a769076392a435147ffd00b6064480c56f5683d8757751432937 | 8a73281a79e71d529fe39061a917a89a65cf7ea9724ae12e7694493a8c39495c |
| fig_qual_counts.json | 5df39913c63fc577beacbda6d6b44722f83b9e80130be6ed08e2dd1423349464 | 574a4e230e5625c6cea9c82f4178f8736cc219e6f8d052e7189930e8d736f65e |

- 没有改的：SELECT_RULE 的文字（它是记录在 selection.json 里的规则原文；其中 "in the whole volume" 指的也是裁掉 2 个体素之后的中心差分体积）；第 4 节的挑选经过；第 6 节的各条。
