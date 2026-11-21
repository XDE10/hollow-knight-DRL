# ADR 0026 用短期运动预测改进战斗姐妹的避让策略

在各自的胜局里，规划器对照每秒打出 14.6 点伤害，每分钟损失 0.92 格血。纯神经走位版的 8 场胜局分别为 14.9 点和 2.06 格，动作选择版的 9 场胜局分别为 15.0 点和 2.04 格。三组的种子与对局数不同。这个差距使本轮研究集中在“保持输出时，能否少挨打”。[规划器逐场记录](../../models/sisters-world-v2/evaluation.json) · [走位版逐场记录](../../models/sisters-native-iql/timing-executed-movement/evaluation.json) · [动作选择版逐场记录](../../models/sisters-native-iql/timing-learned-options/evaluation.json)

用两个不同初始化的神经世界模型从同一局面做八步预测，每一步代表 0.04 个游戏秒。训练时输入实际按键，用后续实测位置、速度、无敌状态、灵魂和冻结策略的状态表示校正预测。两种候选操作分别让小骑士走出一条预测轨迹。另一个输出只从起始观测预测 0.16 秒和 0.32 秒后本体和飞镖的碰撞框分布。同一个模型比较两种操作时共用这一份敌人预测。模型在两条轨迹各自经过的位置读取接近程度，并计入骑士当时处于无敌状态的可能性。[训练实现](../../scripts/train_sisters_imagined_keys.py)

这里读取的是敌方根物体的碰撞框。它与攻击实际命中的时刻有差距。用开发训练集中能明确归因、前后相邻且小骑士当时可受伤的一格掉血记录检查，449 次本体伤害中有 435 次、319 次飞镖伤害中有 314 次，在掉血前后至少一次观测到骑士距该类框不超过 1.5 个游戏单位。仅看掉血前一刻，距离中位数分别为 0.55 和 0.66；同场匹配的可受伤对照状态为 3.07 和 11.97。这个距离能定位险情。实际损血仍由原生血量与受伤事件计分。[游戏中的碰撞框读取](../../mod/HKRLEnvMod/Observation/EntityReadHelpers.cs)

强化学习的奖励来自真正的随机操作实验。每次触发时，两种操作各有一半概率被选中。每个选项执行后都会回到原神经策略。用整场划分的训练集，比较两种选项在相似局面下四秒内的实际奖励，拟合神经策略的选择权重。八秒实测输出和失血用于检查较晚出现的后果。短期运动预测提供选择特征。运行游戏时，只需在本机计算冻结的神经网络。[强化学习更新](../../scripts/train_sisters_compound_options.py)

世界模型先用旧开发集学习。留出的随机水平移动试验比较从同一起点预测两种动作时，各自在 0.16 秒后的位置误差。旧试验使用 75 种按键，新走位策略使用 79 种。读取旧试验时，训练器核对当时的冻结策略和每一次实际按键，并预测旧策略接下来会按的键。把实际随机选中的动作换成另一种之后，位置误差应在多数留出对局增大。独立的联合操作实验共 24 场，其中 18 场用于学习，6 场检查实测回报、伤害和失血。原有随机投影方法也使用这六场开发数据。最终两个课程目标按各自全新的百场实机成绩验收。[旧随机试验的冻结策略](../../models/sisters-native-iql/timing-preferences/) · [两阶段验收要求](../../configs/experiments/sisters_acceptance.json)

Colab L4 上的两个世界模型各完成 1,800 次更新。四场留出的旧水平移动试验共有 69 次符合时序要求的抽签。按场等权，输入实际选中方向时，0.16 秒后骑士横向位置的平均绝对误差为 0.18 个游戏单位；换成另一方向时为 1.30 个单位。四场均呈现这一差别。相应窗口只记录到 3 格实际失血，四场中按预测接近程度换动作的失血差估计均为零。这个结果确认了模型对走位方向的敏感性。能否凭它减少损血，仍需用后续联合操作的实测结果判断。[冻结世界模型和逐场误差](../../models/sisters-native-iql/timing-imagined-motion/)

在仓库根目录，可以用下列命令从保存的走位策略及开发数据重训。第一个数据包的生成方式见 ADR 0025。第二个数据包的风险权重、行为策略和实机抽签须对应同一次采集。[开发数据合并方法](0025-sisters-recurrent-control.md)

```sh
base=models/sisters-native-iql/timing-executed-movement
old=models/sisters-native-iql/timing-preferences
python scripts/train_sisters_imagined_keys.py --archive runs/sequence-replay.zip \
  --parent "$base/learner.pt" --behavior "$old/learner.pt" \
  --behavior-policy "$old/world_iql.bin" --output runs/sisters-motion-world
python scripts/train_sisters_compound_options.py --data runs/compound-trials/development.zip \
  --risk runs/compound-trials/risk.bin --parent "$base/world_iql.bin" \
  --learner "$base/learner.pt" --action-world runs/sisters-motion-world \
  --output runs/sisters-motion-policy
```
