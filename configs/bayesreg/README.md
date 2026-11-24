# configs/bayesreg

本目录放 bayesreg 的输入配置。这里**只说明格式**：真实的 `dev_subjects.json` 要读 data/ 下的开发集清单才能生成，等允许运行时按 BZ 的指定生成并提交；`segmenters.json` 等结构层模型训练完再登记。读取与核对的代码在 `bayesreg/io/guard.py`、`bayesreg/io/pair.py`。

## 规范受试者 id（`bayesreg/io/guard.py`）

逐字符精确匹配（ASCII、区分大小写），不做模糊匹配，也不做任何规范化；写法不规范一律拒绝，状态 `subject_sealed`。

| 数据集 | 正则（fullmatch） | 来源 |
|---|---|---|
| OASIS | `OASIS_OAS1_[0-9]{4}_MR1` | 主线 manifest 的 fixed_subject / moving_subject（受试者目录名） |
| LPBA40 | `S(?:0[1-9]\|[1-3][0-9]\|40)` | 主线 LPBA40 manifest；S37–S40（H2）写进任何白名单都会被拒绝 |
| IBSR18 | 待定（TODO） | 目录名约定未核实；IBSR18 全部是留出集（H3），在定下之前任何 IBSR18 id 都按不规范拒绝 |

## dev_subjects.json（开发阶段受试者白名单）

```json
{
  "schema": "bayesreg.dev_subjects.v1",
  "subjects": {"OASIS": ["OASIS_OAS1_0438_MR1", "..."], "LPBA40": ["S01", "...", "S36"]},
  "sources": [
    {"name": "stage_T0_tune_v1", "dataset": "OASIS",
     "path": "<仓库相对 POSIX 路径，例如 data/three_d/stage_T0_tune_v1/selection.json>",
     "sha256": "<该文件的 64 位小写十六进制>",
     "never_for_training": true,
     "subjects": ["OASIS_OAS1_....", "..."]}
  ]
}
```

- 顶层只允许 `schema`、`subjects`、`sources`（可另加字符串 `comment`）；JSON 里重复的键会被拒绝。
- `subjects[数据集]` 必须**恰好等于**该数据集各来源 `subjects` 的并集；id 必须规范，同一列表内不得重复。
- 来源：`stage_L2R_val_v1`（20 人）、`stage_T0_tune_v1`（`never_for_training: true`）、`stage_E11_native_v1`、LPBA40 S01–S36；B 线训练集名单在抽签提交后作为新来源加入。`path` 只能是仓库相对路径（不得以 `/` 开头，不含 `.`、`..` 段或空白）。`verify_allowlist_sources(allowlist, root)` 重算每个来源文件的 sha256 —— 它会读 data/ 下的清单，只在允许时调用。
- 只有白名单上的受试者能被 `load_pair` 读入；其余（H1–H4、H4 候选池、未列出的、写法不规范的）一律 `subject_sealed`，并且在打开 pair.npz 之前就拒绝。
- 解封由 BZ 执行，代码只留接口：`load_pair(..., unseal_token=...)` 只记录 token 的 sha256，不放行任何受试者。

## segmenters.json（结构层模型登记表，按折级模型登记）

```json
{
  "schema": "bayesreg.segmenters.v1",
  "models": {
    "<model_id>": {
      "segmenter_id": "<集成的 id>",
      "fold": 0,
      "member_model_sha256": ["<每个集成成员权重文件的 sha256>", "..."],
      "label_contact_subjects": ["<规范 id>", "..."],
      "label_contact_subjects_sha256": "<subject_list_sha256(列表)>"
    }
  }
}
```

- `label_contact_subjects`：该模型接触过标签的**全部**受试者（训练、验证、早停、温度标定）。其中每个人都必须在白名单上，且不得是 `never_for_training`（T0）受试者，否则用到该模型时拒绝。
- `subject_list_sha256`：UTF-8 文本，每个 id 一行、按列表顺序、每行以 `\n` 结尾，取 sha256。
- 每条恰好这五个键；`fold` 是非负整数（不接受布尔）；同一 `segmenter_id` 的折号不重复；成员 sha256 不重复。
- 模型 id 与 segmenter_id：ASCII 标识符 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`，不含空白和换行；含 `oracle`、`truth`、`seg4`、`seg35`（大小写无关的任意子串）的一律拒绝；`atropos`、`fast` 不能作结构层的 segmenter_id。
- 加载时记录登记表文件本身的 sha256，写进接受记录。

## structure_groups_v1.json（结构层分组表，schema bayesreg.structure_groups.v1）

结构层分割器（docs/bayesreg/04_分割器.md）的网络输出 seg35 的 36 类（类 c = 标签 c，0 为背景）；分组表把标签划分成背景（通道 0）和 K_S 个命名的组（通道 1..K_S），推断后对类概率求和得到组概率，所以改分组不必重训。读取与检查在 `bayesreg/seg/unet/groups.py`。

- 默认 `structure13_v1`（K_S = 13，左右合并；`rule_exceptions` 列出留在通道内部、没有强度边界的三处相接：蚓部处左右小脑皮层、脑干与小脑白质、胼胝体中线处左右白质，以及待核实的"不相邻"论断）；`tissue4_ref` 是仿 seg4 的对照分组，seg35→seg4 的对应**未核实**（状态 `reference_unverified`，用前先跑 `python -m bayesreg.seg.unet.cli check-tissue4`，至多读 3 名训练受试者）；`identity35` 是每标签一个通道的消融分组。
- **状态**：推断只接受 `default`、`ablation`、`verified`；其余（如 `reference_unverified`）默认拒绝，`--allow-unverified-grouping` 显式放行并写进 provenance。改状态是需复核的改动。
- 每个分组的背景标签与各组必须恰好划分 0..35；组名是合规标识符（不得为 `background`），不区分大小写不重复。P_S provenance 的 `class_names` 就是该分组的组名（按顺序），并记录分组名与本文件的 sha256。
- 结构层模型的登记：集成 `segmenter_id = unet35_v1_ens`，折级模型 `unet35_v1_foldA`（fold 0）、`unet35_v1_foldB`（fold 1），各列两个种子成员（member.pt：权重 + 温度）的 sha256；由 `python -m bayesreg.seg.unet.cli register` 在一折的种子都标定完后写入 segmenters.json，只增不改。写入前 `registry.verify_members` 逐项核对：每个成员的 label_contact 恰为 folds 中该折 train + calib，各成员 folds_sha256 相同且等于 folds 文件（真实数据为钉死值 98e9e676…），训练与标定都有标签守卫记录且读取 ⊆ contact（标定恰为本折 calib），member.pt 内的折、contact、温度与记录一致，contact 全在白名单与训练来源 `bayesreg_train_v1` 内（docs/bayesreg/04_分割器.md §7）。真实训练链条先登记到结果目录的暂存表，总负责人复核后再写进本目录的 segmenters.json。

## 真实数据运行的钉死输入（`bayesreg/seg/unet/pins.py`）

结构层分割器的 cli 在真实数据上（默认）要求：代码是已核验的 git-archive 快照（train、calibrate）或干净的 git 工作树（其余命令）；`splits/bayesreg_v1/folds.json` 的 sha256 为 98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245，每折 45 train + 8 calib；白名单就是本目录的 `dev_subjects.json`，且 `verify_allowlist_sources` 对源仓库通过；训练来源按名字 `bayesreg_train_v1` 指定，人员集合等于 folds。`--synthetic` 不走这些检查，只接受合成受试者（OASIS_OAS1_0900_MR1 及以上）与非真实数据根目录。
