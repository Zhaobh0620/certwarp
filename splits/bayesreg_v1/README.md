# splits/bayesreg_v1 — B 线受试者划分（冻结）

由 `experiments/split/draw_split.py` 生成（种子 20261030），规则、来源、核对与披露见 `docs/bayesreg/06_数据划分.md`。
本目录**只能新增，不能修改**（CLAUDE.md git 规则：冻结文件只能新增）；每个文件的 sha256 在 SHA256SUMS。

| 文件 | 内容 |
|---|---|
| pool_206.txt | B 线候选池，206 人，按规范 id 排序 |
| holdout_B_subjects.txt | B 线留出集，100 人，按 id 排序。**只由 BZ 解封** |
| holdout_B_pairs.tsv | B 线留出集 50 对，列 fixed、moving，按抽签顺序；配对名 `<fixed>__<moving>` |
| train_subjects.txt | 训练集，106 人，按 id 排序 |
| folds.json | 训练集每人的折（A/B）与角色（train = 该折训练子集；calib = 该折标定子集，用于早停与温度标定） |
| eb_pairs.tsv | 群体经验贝叶斯训练对 30 对，列 fixed、moving、fold；同折配对，受试者不重复 |
| draw_log.json | 种子、numpy/Python 版本、输入文件路径与 sha256、各集合人数与核对结果、脚本 sha256、UTC 时间 |

使用规则（摘要；以 06_数据划分.md 与 CLAUDE.md 硬约束 6 为准）：
- 留出集与 H1 同等对待：不用于任何开发（调参、冒烟、诊断、分割、看结果）；只由 BZ 解封；解封后每对每方法一次，全部报告。
- 训练集只用于结构层分割器训练（按折、标定子集只作早停与温度标定）与群体经验贝叶斯（eb_pairs.tsv，用另一折训练的分割器预测）。
- T0 的 24 人不在本划分里，永不用于训练任何学习部件。

复现：`python -I -B experiments/split/draw_split.py --out <新目录>`，除 draw_log.json 的 utc_time 与 SHA256SUMS 中
draw_log.json 那一行外，输出应逐字节相同（numpy 2.0.2；numpy 的 Generator 不保证跨版本流兼容，以本目录文件为准）。
