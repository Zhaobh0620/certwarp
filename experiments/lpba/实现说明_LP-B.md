# LP-B 实现说明：各入口的允许清单（spy）

依据：定稿预登记（06121278…）第 8.5 节第 7 条"各入口的 spy 用允许清单"；合同 C26.5-1、C26.8-5、C26.9-5；LP-B 复核 I3、I4、次要 1，差异复核 I3-a、次要 1–3（decision_log 2026-11-19 总负责人的两次裁决，约 3:13 PM 与 3:51 PM CST）。只写 LPBA40 的两个模式（lpba40_test、lpba40_feas）。spy 只在 LPBA 分支装，留出集与开发集的进程不装、记录不变。

## 一、方法臂各进程与配置 C 臂：lpba_arms.ArmPolicy + ArmSpy

实现：experiments/lpba/lpba_arms.py（ArmPolicy、ArmSpy、arm_spying），装在 run_hx.lpba_spied（监督、t88、repair、第三次认证）与 hb_configc.lpba_spied_run（配置 C 的驱动）。机制是 hp_core 的审计钩子（open、os.listdir、os.scandir），另外拦 NpzFile.__getitem__。装之前核守卫的封存名单缓存是空的（C23.6-5），不空就拒。

**判法，按这个次序**（复核 I3："只管数据根与结果根，库文件不管"；差异复核 I3-a：代码树里的记录不算代码）：
1. 本进程自己的目录（write_inside）与锁文件（write_exact）：可读可写，不过名字规则（冻结 worker 的 labels.json、冻结树的私有副本都在这里）。
2. 逐个列出的输入（read_exact），以及只读的输入目录（read_inside）：只许读。
3. 禁读根，一律拒：
   - 数据根 /mnt/d/data，以及**整个 /mnt/d**（含留出集的 D: 库 /mnt/d/seq_vp_tag_holdout_B_v1、论文一的 /mnt/d/seq_vp_3d_results）；
   - seq_vp_3d 树；
   - 各代码树与主检出的 data/、archive/、docs/bayesreg/reviews/：archive/ 禁读，守卫读的三份封存名单逐个放行（它们在第 2 条的 read_exact 里，先于本条判；LP-G）；
   - 冻结树（结果根下的 frozen_eb808be）的 results/、docs/、figures/。
4. 代码树（运行中的检出，不管它在哪：正式与可行性都放在 `<正式根>/code/`，C26.9-5；bayesreg 的代码树；冻结树）：
   - 代码后缀（hp_core.CODE_SUFFIXES：.py、.pyc、.so、.pth）与 `__pycache__` 里的文件（字节码、numba 缓存）算代码，读时不过名字规则；
   - 必要的配置文件列在 lpba_arms.CODE_CONFIGS，也不过名字规则；现在是空的，t88 读取链、配置 C 驱动、第三次认证读到的配置文件名字都不碰名字规则（见第五节）；
   - 代码树里其余的文件（json、tsv、txt、md 等）要过第 6 条的名字规则；
   - 写只许进 `__pycache__`（同代码身份的缓存例外）；
   - 记录：.py 的读取只计数；代码树里其余的打开（非 .py 的读取、`__pycache__` 的写）逐条记。
5. Python 环境与系统目录：只读。
6. 名字规则：hp_core.FORBIDDEN_NAME_PARTS（aligned_seg、seg4、seg35、dice、metrics、evaluation、truth）加 'label'，命中就拒。
7. 结果根拒（不是逐个允许的）。
8. 其余不管：缓存、/tmp、/dev。

pair.npz 的成员只认 fixed、moving、initial_u（自己目录里的派生对除外）；其他归档不管（预测、场）。越界就停（C26.8-5）：
- 子进程（t88、repair）写 `<目录>/spy_breach_<k>.json`，退出码 7；
- 监督进程读到后写 line.json hx_implementation_error，退出码 9，队列停下；
- 方法队列在删任何文件之前（certified_check 之前）先查格子里有没有 `spy_breach_*.json`，有就停，line.json 怎么写都一样；这时 repair/certified.npz 不删，step4.json 的 certified 记 kept（差异复核次要 1）；
- 第三次认证越界退出码 7，复核单元照 SPY_BREACH_EXIT 判成 STOP 28；
- 配置 C 在锁之前越界，写拒收记录（stage spy）；锁之后越界，记 infra_failure（kind spy_breach）。被某段代码吞掉的越界也一样：结果改记 infra_failure（kind spy_breach，原来的状态记在 superseded），中间提交（committed_*.npz、level_*.npz）不删（差异复核次要 1）。同时写 `spy_breach.json`，配置 C 队列停下。

**越界不走补跑通道，一律停下交 BZ**（差异复核次要 3）：LPBA 模式下，格子里有越界记录时，补跑授权与复核重做的核对都拒：
- 方法臂：hb_method.load_resume；
- 配置 C：hb_configc.rerun_authorization、load_resume（hb_configc.spy_breach_problem）；
- 复核单元：重做的核对（hb_review.lpba_spy_stop_problem：格子里的 spy_breach_*.json，或 kind 为 spy、third_spy 的停止记录）；
- 外部臂：xb_drive 的补跑与续跑（xb_drive.lpba_spy_breach_problem：stage spy_breach 的拒收记录，或 spy_passed 为假的记录）。

留出集与开发集的这些核对不变。BZ 的"未批补跑"（no_resume）仍照旧可以写。

spy 给计时的臂多加了一点开销（每次打开都过审计钩子）。时间只作描述，不影响判定（差异复核次要 4），结果文档里说明。

每个进程的记录：`opened_files_<阶段>_<k>.json`。代码、系统和不管的位置只计数，不逐条记；其余逐条记。

已知局限（LP-H 修订 1；修订 1 核对次要 2 改准）：
- 认不出相对目录描述符的打开：`os.open(名字, dir_fd=…)`、shutil.rmtree 基于描述符的那一支、os.fwalk。审计事件 open 只带裸名、不带 dir_fd，各 spy 的钩子把裸名照工作目录补全。hp_core 的 Spy、ArmSpy、XB 的 DriverSpy 与 spyrun、lpba_eval 的 spy 都是如此。
- 按描述符列目录（`os.scandir(fd)`）：hp_core 的 Spy 与 ArmSpy 看列目录事件，参数是整数时当作'不是路径'放行。XB 的 DriverSpy 与 lpba_eval 的 spy 根本不看列目录事件，任何列目录都不判。
- 因此，**本代码树里**凡在 spy 下运行的代码都不用 shutil.rmtree、os.fwalk、tempfile.TemporaryDirectory 或 dir_fd。静态测试 tests/test_bayesreg/lpba/test_lpba_spy_no_dir_fd.py 守着这一条，例外只有 lpba_eval 删单文件临时目录的两处，见第七节。它只扫本代码树（各个入口的静态导入闭包，含按名字加载的模块）。管不到的有三类：
  - 方法进程里由 runpy 跑的第三方代码，例如 convexAdam.convex_adam_nnUNet；
  - 方法进程两个 conda 环境（bl_fireants、bl_convexadam）里的库；
  - 跨文件系统的 shutil.move：搬目录时会走 copytree 再 rmtree。闭包里只有 decide_ef、decide_mc0、decide_mc0p 三处 shutil.move，都不在 LPBA 的 spy 进程里跑。

### 各进程的允许清单（`<out>` = `<根>/work/method/<对>`，`<cache>` = context 的 paths.cache）

| 进程 | 读写（自己的） | 只读（逐个） | 只读目录 |
|---|---|---|---|
| 监督（run_hx.run_pair，LPBA 模式） | `<out>`、`<cache>` | stage 的 `<对>/pair.npz`、`<对>/manifest.json`；封存名单四份（holdout_common.sealed_list_paths） | — |
| t88 子进程 | `<out>`、`<cache>` | 同上 | 两侧受试者的 HI 预测目录 `<pred_root>/<s>/`、Atropos 目录 `<atropos_root>/<s>/`，登记成员的根（context 的 models_root） |
| repair 子进程 | `<out>`、`<cache>` | 同上 | — |
| 第三次认证（复核单元的子进程里） | `<out>`、`<cache>` | 同上 | — |
| 配置 C 驱动（hb_configc.run，LPBA） | `<根>/work/configc/<对>` 与 `.resume`、`<根>/configc/<对>`、本次的 batch 目录（差异复核次要 3：别的批次不写）、`<根>/work/configc/cache` | stage 的 `<对>/pair.npz`、`manifest.json`；封存名单四份（holdout_common.sealed_list_paths） | `<根>/batches`（只读：以前的拒收记录，用于序号与补跑核对） |

配置 C 驱动另有可写的锁文件：`<根>/locks/hb__configc__<对>.lock` 与 `.resume`。

各进程共有的允许：
- 代码树：本检出（`<正式根>/code/`）、sealed.REPO、冻结树 frozen_eb808be（它的 results/、docs/、figures/ 除外）；
- 名单、LPBA 白名单、预登记、冻结文件、c 记录：都在代码树里，照第 4 条放行，过名字规则，逐条记；
- 封存名单四份：archive/ 禁读，守卫读的三份封存名单逐个放行（LP-G）。ArmPolicy 在构造时把 hp_core.allowed_list_paths()（= holdout_common.sealed_list_paths，prep 与 XB 用的同一来源）并进 read_exact，按 resolve 之后的路径逐个比对，第 2 条先于禁读根判；另一份在 splits/bayesreg_v1/ 下，以前照第 4 条放行，现在也走 read_exact。只放行这四个文件：写它们、plan/ 下的其他文件、plan/ 的目录列举、archive/ 的其余部分都照旧拒。记录里 policy.sealed_lists 列出这四份，每次读取逐条记（why 为 input）；
- 第三次认证在复核单元的子进程里另有复核单元自己的 spy（下面第二节），两层叠加，装卸先进后出。

## 二、复核单元：hb_review.ReviewSpy + lpba_policy

禁读清单式的 spy（C22.H1-8 的写法），留出集原样；LPBA 模式改的地方：
- 禁读根：
  - forbidden_roots()；
  - 主检出与代码树的 data/；
  - 本模式的 eval/；
  - 另一个 LPBA 模式的根；
  - 留出集的根 holdout_B_v1、D: 的两个目录；
  - 其他臂在本模式根下的记录与 work 格子，它们的锁与批次（前缀规则）。
- 开发产物规则：结果根下只认本模式根的名字；seg/ 下只认含本模式根名的项。
- keep：代码树 hb.REPO（复核 I4，照 dev_policy；禁读根仍在 keep 之前判）。
- 检出的位置（差异复核次要 2，C26.9-5）：正式与可行性都从 `<正式根>/code/` 跑。可行性模式把正式根列为禁读根，但它下面的 code/ 不算在这个根里（ReviewPolicy.deny_except）；code/ 里的 data/ 与 docs/bayesreg/reviews/ 仍由各自的禁读根拒，正式根的其余部分仍拒。留出集与开发集的 deny_except 是空的。
- 封存名单（LP-G，C22.H4-3：这四个文件对复核单元不算禁读）：代码树 hb.REPO 的 archive/ 是禁读根，archive/ 禁读，守卫读的三份封存名单逐个放行。两种 LPBA 模式的 deny_except 都给这个禁读根（_paths 的两种写法）配上落在它下面的每份名单的精确路径（hb_review.sealed_list_exceptions；名单取 holdout_common.sealed_list_paths 与 sealed.REPO/<path> 两种写法）。只是这三个文件：plan/ 下的其他文件、archive/ 的其余部分仍拒，其他禁读根照旧判。复核记录 opened_files.json 的 policy.deny_except 列出这些对；留出集与开发集的 deny_except 仍是空的。
- 名字规则（复核次要 1）：原有的标签文件、评测记录、评测锁、结果文档规则不变，另加 `label[^/]*\.nii(\.gz)?$`。
  - LPBA40 的真值是 `<s>/delineation/label.nii.gz`，旧 stage 的副本是 `S??_label.nii.gz`。
  - 不用"名字含 label 就拒"：第三次认证退到配置 C 时要复制冻结树，里面的 scripts/check_labels_stage_3d_e10.py 会被拦住。
- 写（LP-G 修订 1，LP-G 复核次要 1）：这个 spy 只看禁读清单，原本不分读写；LP-G 之后三份名单不再被 archive/ 这个禁读根拦，所以另加 read_only：两种 LPBA 模式下，四份封存名单（两种写法）只许读，以写方式打开（open、io.open、Path.open 的模式含 w、a、x、+）一律拒（hb_review.opens_for_writing）。留出集与开发集的 read_only 是空的，判定不变。

## 三、外部臂

xb_drive 的 DriverSpy 是允许清单式的，原样沿用。LPBA 下换的地方：
- 代码树与主检出的 archive/、docs/bayesreg/reviews/ 是禁读根（LP-C2 整合第 7 项）。archive/ 禁读，但 driver 自己要读的 archive/ 文件逐个放行（LP-G 修订 1）：
  - 守卫读的三份封存名单；
  - check_frozen 与 check_environment 打开的 BL2 冻结件（xb_core.archive_reads：harness/drive.py、3d_stage_BL2_v1/plan/protocol.md、本臂各组的 smoke_cmd.md、四个方法脚本、15 份环境记录；check_environment 读的本臂环境的 conda_list 与 pip_freeze 就在这 15 份里，四个 LPBA 臂每臂一共 22 个文件）。为了算这份清单，driver 在装 spy 之前先读一次冻结配置（configs/frozen/holdout_B_v1_baselines.json，核 sha256）；这次读不进 spy 的记录，装 spy 之后 check_frozen 再读的那次会记进去。
  - 这些都放进 xb_policy 的 read_exact（规则 1a），只许读、不许写；LPBA 模式下 allowed_exact 留空，逐个放行的输入也都只许读。留出集与开发集的策略没有 read_exact，行为不变。
- holdout_root 换成本模式的根，所以本模式的 stage/ 与 atropos/ 都在禁读根里，只有逐个列出的 manifest.json 与 pair.npz 放行；
- HI 的 provenance.json、probs36.npz 与受试者记录逐个放行；
- 授权目录只在 holdout_common 认得这个模式时才放行；
- 检出在 `<正式根>/code/` 时照 xb_policy 的规则 3a 读作代码树（与留出集检出在 holdout_B_v1/code 下同理）；
- 越界不走补跑通道（见第一节末）。

## 四、可行性的第 7 步（复核 I2，总负责人裁决 (a)）

- 可行性模式的输入锁写到可行性根：`<results>/lpba40_feas_v1/hb_input_lock.json`。不进代码树、不提交、不删场，可行性链全程在同一个提交上（C26.9-3）。
- 正式模式照旧：锁写进 runs/lpba40_test_v1/hb_input_lock.json，提交之后 lpba-delete-fields 只删 hx_certified 那些对的 field_0600.npz。
- 两个函数都自己核正式模式的规则一（次要 7）。

## 五、t88 读取链在合成世界里的重跑（差异复核 I3-a）

测试 tests/test_bayesreg/hb/test_hb_lpba.py::test_i3a_the_t88_read_chain_runs_under_the_spy_without_a_breach。合成世界用 LP-A 的 stage、Atropos 与 HI。t88 子进程照真实的读取链走：准入 → 分割器锚点 → read_predicted（结构层、组织层、对齐 QC）→ 颜色表。走到 prepare_stacks 停下，那里开始 T88 的算术，之后不再读文件。

替身只有下面几处：
- 合成登记表提交进一个临时 git 仓库，加进 FORMAL_REGISTRIES；
- 锚点的两次核对换成通过（合成登记表不是合同的那份）；
- 对齐 QC 照算，判成通过（合成影像没有对齐）。

读数（一次运行）：
- 打开 124 次，逐条记的 112 条；越界 0；
- 记录按类：输入 45、代码树文件 56、代码 9（.pyc）、自己目录的写 2；
- pair.npz 只读了 fixed、moving、initial_u 三个成员；
- 代码树里非 .py 的读取：颜色表 colour_table_a5_v1.json、structure_groups_v1.json、folds.json、configC_reference.json、LPBA 白名单、臂表、holdout_B_v1_baselines.json、预登记、封存名单（守卫读）、c 记录、对表与受试者名单。名字都不碰名字规则，所以 CODE_CONFIGS 是空的。

注（LP-G）：这一节的合成世界把封存名单放在 lists/ 下，不在 archive/ 里，所以没有测出守卫读的三份名单被 archive/ 这个禁读根拦下（可行性链 11-20 在监督进程停下的原因，decision_log 2026-11-20）。

## 六、封存名单按生产布局的测试（LP-G）

测试 tests/test_bayesreg/hb/test_hb_lpg.py。合成世界同上，另把四份合成名单搬到生产的相对路径（tests/test_bayesreg/hb/_lpba_world.py 的 production_lists：路径取 bayesreg.io.sealed 模块常量，内容与 sha256 都是合成的，不读真实名单）：三份在 archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan/ 下，一份在 splits/bayesreg_v1/ 下；plan/ 里另放一个不是名单的合成文件，archive/ 里另放一个文件。
- (i) ArmPolicy：方法四个进程与配置 C、两种模式，四份名单的读为 input（原路径、resolve 之后、经符号链接三种写法），写为 forbidden；plan/ 的其他文件、名字只是前缀相同的文件、plan/ 下的 .py、archive/x.json 都拒；plan/ 与 archive/ 的目录列举都拒；document() 的 read_exact 与 sealed_lists 含这四份。
- (ii) 读取链：t88（同第五节）、监督进程的准入（两种模式）、repair、第三次认证、配置 C 驱动，以及监督进程与第三次认证在本进程里走的配置 C 退路（derive_pair、防火墙核对、候选再认证；冻结运行与 firewall_c.prepare 用配置 C 测试的替身），都 0 越界，opened_files 记着四份名单的读取，archive/ 下没有别的读取。去掉修法（hp_core.allowed_list_paths 不给 ArmPolicy 任何路径）时，监督进程停在 H1 那份名单（stage supervisor，line.json hx_implementation_error），t88、repair、第三次认证、配置 C（锁之前，拒收记录 stage spy）也都停在 H1。
- (iii) 复核单元：hb.REPO 指向持有名单的代码树（生产里是同一份检出），两种模式的 lpba_policy 放行四份名单（三种写法），其余照旧拒；review_pair 准入并定稿、复核单元里的第三次认证（ReviewSpy 与 ArmSpy 两层）都通过；去掉 deny_except 的名单例外时 review_pair 停在 H1（STOP 28）。留出集与开发集的 deny_except 是 ()。
- (iv) 跨入口（LP-G 修订 1 扩成两条，LP-G 复核次要 2）。对每个装了 spy 的 LPBA 入口核两条：archive/ 里它正当要读的文件都放行；archive/ 里其余的都不放行。
  - 入口：ArmPolicy 各使用者、复核单元两种模式、XB 的 LPBA 驱动（四个臂、两种模式）、lpba_common.Policy 的使用者（prep、Atropos、HI、准入干跑），以及 lpdev_a、lpdev_b、lpdev_c、评测入口 lpba_eval 的 spy、make_eval_manifest_lpba 的守卫。代码树取持有名单的那份。
  - 正当要读的：三份在 archive/ 下的名单；XB 另加 check_frozen 与 check_environment 打开的 BL2 冻结件（xb_core.archive_reads）。
  - 其余的：plan/ 的其他文件（含经链接的写法）、plan/x.py、名字只是前缀相同的文件、archive/x.json、BL2 冻结件（XB 以外的入口）；XB 另加别组的 smoke_cmd.md 与 harness/ 下另一个文件。
  - 两类入口照现状写进测试，不是 LP-G 改的：
    - hp_core.Policy（lpba_common.Policy 与 lpdev_common.Policy）没有 archive/ 这个禁读根。archive/ 下非代码后缀的文件判为 unlisted，等于拒；但它自己代码树里的 .py 照代码放行。测试在 hp_core.CODE_TREE 上只按路径核这一点。
    - 评测入口 lpba_eval 的 spy 与 make_eval_manifest_lpba 的守卫只看数据根与结果根下的留出集：四份名单在它们的放行名单里，archive/ 的其余部分不在它们监视的位置，既不记也不拒。
- (v) LP-G 修订 1 另加两条：
  - 复核单元的 spy 拒绝以写方式打开四份名单（open、Path.open、write_text），文件不变；留出集与开发集的 read_only 为空，why(path, write=True) 与 why(path) 相同。
  - 检出放在 `<正式根>/code` 下时，可行性模式的两层例外一起作用：正式根除了 code/，code/archive/ 除了名单。名单两层都过；plan/ 其余、archive/ 其余、正式根其余照旧拒；ArmPolicy 两种模式在这种布局下也只放行名单。
  - XB 的那组测试在 tests/test_bayesreg/xb/test_xb_lpg.py，见第三节。

## 七、私有副本的删除与工作目录（LP-H）

可行性 run3（F''' 43c99313）停在配置 C 的 S01←S02 一格：spy 拦下了对 `$CODE/data` 的一次 open（读），判为 forbidden（代码树的 data/），结果记 infra_failure（kind spy_breach），exit 20。检出里没有 data 链接（R1.2），这个路径并不存在。decision_log 2026-11-21 约 2:58 PM CST。

- 根因：firewall_c.prepare 做完冻结的 prepare 之后，用 shutil.rmtree 删冻结树的私有副本（`<work>/frozen_copy`）。在 Linux 上，rmtree 走的是基于描述符的那一支（shutil._rmtree_safe_fd），每个子目录用 `os.open(子目录名, O_RDONLY, dir_fd=父目录描述符)` 打开。审计事件 open 只带裸名，不带 dir_fd，hp_core 的钩子就把裸名照工作目录补成绝对路径。
  - LPBA 的各单元从检出跑，--working-directory 就是检出（hb_core.unit_argv）。于是副本里的 scripts、jd_curl、experiments、three_d、solvers（两次）、backends 被记成检出下的同名路径，照"代码树文件"放行。**读 run3 记录的人请注意**：opened_files.json 里这 7 条放行记录（`<code>/scripts`、`<code>/jd_curl`、`<code>/experiments`、`<code>/three_d`、`<code>/solvers` 两次、`<code>/backends`）被认错了路径，实际打开的是副本里的同名目录，不是检出里的。副本的 data/（冻结 prepare 的槽位 data/three_d/stage_L2R_val_v1/<对>）被记成 `$CODE/data`，撞上禁读根。
  - 真正被打开的是工作格子自己的 `frozen_copy/data`，没有越过防火墙；是 spy 把路径认错了。
- 为什么以前没撞上：留出集的进程不装 ArmSpy；复核单元的 ReviewSpy 只包 builtins.open 与 io.open，看不见 os.open；LP-B、LP-G 的测试用的是 firewall_c.prepare 的替身（fake_prepare），没有走真的 rmtree；run3 方法臂两格都是 hx_certified，监督进程没有走到配置 C 退路。
- 修法（不改任何策略）：firewall_c.remove_tree 按绝对路径自底向上删（os.walk，topdown=False，不跟链接；os.unlink、os.rmdir），不打开任何文件；列目录走 os.scandir 的绝对路径，判为本进程自己的目录。删掉的文件与 rmtree 相同。三处都用它：配置 C 驱动、监督进程的配置 C 退路、第三次认证的配置 C 退路。
- 测试：tests/test_bayesreg/hb/test_hb_lph.py。生产布局：检出在 `<正式根>/code`，没有 data/，工作目录就是检出。用合成冻结树跑真的 firewall_c.prepare（私有副本、冻结 prepare 的子进程用一个替身脚本、副本的删除；check_prepared 用替身）。
  - 换回 shutil.rmtree 时，配置 C 一格停得与 run3 相同：infra_failure、spy_breach、`<检出>/data`、write false、exit 20。监督进程（stage supervisor）与第三次认证（stage third）的退路也停在同一路径。
  - 用 remove_tree 时，以上各处以及复核单元子进程里的第三次认证退路都是 0 越界，副本删净，记录里没有按裸名认错的路径。
- 静态守卫（LP-H 修订 1，复核次要 7）：tests/test_bayesreg/lpba/test_lpba_spy_no_dir_fd.py 从各个装 spy 的 LPBA 入口出发，静态地收集它们在代码树里导入的全部模块（约 200 个文件，函数内的导入、core.load_module 按路径加载的模块都算），逐个找 rmtree、fwalk、TemporaryDirectory 与 dir_fd=。找到的必须恰好是登记过的例外：lpba_eval 的 private_temp_dir（刚建的空目录）与 read_label（只有一个文件、没有子目录的私有标签目录：rmtree 只按绝对路径打开目录本身，文件按 dir_fd 用 unlink 删，unlink 不是 open 事件）。新增一处就失败，例外消失也会提示更新。
- 另外两组生产布局的端到端测试，都是 0 越界、没有拒：
  - tests/test_bayesreg/xb/test_xb_lph.py：ants_s、fireants_greedy、fireants_syn 在可行性模式下走 xb_drive.run_cell。真的 check_frozen 与 check_environment，方法步骤用替身（FakeAnts、FakeMethod）。
  - tests/test_bayesreg/lpba/test_lpba_eval_lph.py：lpba_eval 两种模式。
