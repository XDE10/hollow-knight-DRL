# 战斗姐妹的离线强化学习实验

网络在游戏物理循环中直接决定移动、上下瞄准和按键。第一轮只用开发对局训练。在 Colab L4 上先模仿示范，再用真实游戏奖励训练双 Q 网络、状态价值网络和策略。世界模型预测下一个决策时刻的实体状态，参与估计未来回报。[算法与数据划分](../../docs/adr/0020-sisters-native-offline-rl.md)

## 第一轮实机结果

两份权重冻结后各打 10 场，从同样的竞技场开场，原速、原始伤害、9 格血、0 灵魂。胜负按整场结算，过第一只按三姐妹同时登场统计。掉血按对应来源的格数累计。

- 模仿策略 0 场过首只、0 场通关，场均伤害 208.5，最高 351。十场共被本体打掉 43 格、飞镖 36 格、地刺 13 格。
- 世界模型辅助 IQL 策略 2 场过首只、0 场通关，场均伤害 578.8，最高 1337。十场共被本体打掉 64 格、飞镖 37 格、地刺 1 格。两组十场都打出伤害，两组的冲刺键计数都为零。

这两组来自一次训练。IQL 在模仿起点上多做了 4500 次策略更新，场均输出差异不能单独归因给世界模型。本体掉血总数还受战斗长短影响。[完整逐场成绩](round1/evaluation.json) · [两组冻结权重与训练文件指纹](round1/sha256.json)

训练用了 88 场、336158 次原生控制决策，另留出 27 场、97189 次决策。每次决策包含 180 个输入，镜像扩充后训练目录有 135 个不同按键组合。动力学对留出集下一时刻物理特征的平均绝对误差为 0.01628，沿用当前状态为 0.02276。模仿策略与 IQL 的留出按键完全匹配率分别为 78.86% 和 79.55%。[训练记录](round1/training.json)

## 复现

开发集只包含数值特征、按键、事件和场次清单。解压到新目录，使用安装有 NumPy、PyTorch 的环境，从仓库根目录运行以下命令。训练器在可用时选择 CUDA。

```powershell
Expand-Archive models/sisters-native-iql/round1/development.zip -DestinationPath runs/native-round1-data
python scripts/train_sisters_native_iql.py --data runs/native-round1-data --output runs/native-round1-retrained
```

`bc.bin` 和 `world_iql.bin` 用于游戏内推理，`learner.pt` 包含 actor、双 Q、状态价值网络、动力学网络和动作目录。实机复测需要在装有本仓库 mod 的游戏进程启动前设置 `HKRL_NATIVE_ACTOR` 为所选 bin 的绝对路径，`HKRL_NATIVE_ACTOR_LEDGER` 为一个可写的记录文件，清除 `HKRL_TACTICS_POLICY` 和 `HKRL_NATIVE_DEMO_DIR`，设置骨钉倍率 1。启动已配置战斗姐妹存档的游戏后运行评估，并让端口与游戏配置一致。

```powershell
python scripts/eval_sisters_native.py --checkpoint models/sisters-native-iql/round1/world_iql.bin --port 5555 --episodes 10 --seed 9461000 --output runs/native-round1-evaluation
```

历史上 100 场赢 99 场、7 场无伤的是带战术控制器的工程对照。此处的独立评估由网络控制全部操作。[工程对照记录](../sisters-world-v2/README.md)
