# 战斗姐妹世界模型与 PPO

用于《深度强化学习》课程项目。实体世界模型在 Colab L4 上从 68 场真实开发对局拟合，另有 23 场留出。PPO 同时训练 actor 和 critic。实机由 actor 直接选择 42 种原始按键组合，走位和攻击均由策略决定。

## 修正版在原速原伤下的成绩

四份冻结策略分别独立从开场打 10 场，开局 9 格血和 0 灵魂。四组过首只、通关、无伤通关均为 0/10。伤害按每场输出的游戏血量点数统计，踩刺按十场累计掉血格数统计。

- 同一份模仿预训练起点，场均伤害 **224.1**，10 场均打出伤害，累计踩刺 **10** 格。
- 从该起点做 115200 步想象 PPO，场均 **62.7**，8 场打出伤害，累计踩刺 **41** 格。
- 想象 PPO 后接 48 场实机 PPO，场均 **16.8**，6 场打出伤害，累计踩刺 **52** 格。[权重](combat/real_ppo.pt)
- 从同一起点直接做 48 场实机 PPO，场均 **132.9**，10 场均打出伤害，累计踩刺 **24** 格。[权重](combat/real_only_ppo.pt)

两个实机训练组各做了 12 次更新，分别采集 8522 和 4765 次决策。预算按整场固定，实际控制步数和游戏随机过程各有差异。这个设置下，跳过想象训练的 PPO 实战表现较好，两组的场均输出均低于模仿起点。每个配置只有一次训练和十场独立评估。[全部逐场成绩与更新记录](evaluation.json) · [各版本权重指纹](weights-sha256.json)

## 第一轮模型文件

- `before_rl.pt` 保存行为模仿及真实折扣回报预训练之后的 actor、critic 和实体编码器。
- `imagined_ppo.pt` 保存 150 轮想象 PPO 后的策略。每轮从 192 个真实状态开始，每条轨迹展开 5 步，共 144000 个模型转移。
- `real_ppo.pt` 在此基础上使用 24 场真实对局、3259 个来自自身策略的决策，完成 6 次实机 PPO 更新。
- `dynamics.pt` 保存三个拟合动力学网络及事件概率的训练集校准偏置。状态标准化参数在策略检查点中。
- [Colab 训练记录](training.json) 和 [训练日志](train.log) 记录硬件、训练配置、数据指纹和留出结果。
- [训练方法与复现实验](../../docs/adr/0019-sisters-model-based-ppo.md) 包含实机评估和继续 PPO 更新的入口。

原有 [99 胜工程对照](../sisters-world-v2/README.md) 使用独立的混合控制器。它的通关率与无伤场数只归属于该版本。

在已配置好游戏 Mod 的 Windows 项目中，可从仓库根目录运行一场。第一条命令使用项目的 3 号存档启动游戏并检查连接。第二条命令加载 actor 并写出本场数据，输出目录需使用新名称。

```powershell
.\scripts\windows\start_sisters.ps1 -CheckOnly
.\.venv\Scripts\python.exe scripts/run_sisters_latent_ac.py --checkpoint models/sisters-course-ac/combat/real_ppo.pt --port 5555 --episodes 1 --seed 9410000 --output runs/course-demo-001
```

## 第一轮独立实测

三个冻结检查点分别从开场打 10 场，原速、原始伤害、9 格血、0 灵魂，三组均未过第一只。模仿预训练策略场均打出 23.1 点伤害，想象 PPO 为 4.2，接实机 PPO 后为 12.6。三组总踩刺掉血分别为 27、35、4 格。实机 PPO 的输出高于想象 PPO，仍低于预训练起点。本轮提供了策略能否从预测迁移到实机的一次反例。

训练集有 13321 个观测间隔包含挥刀，其中 7384 个间隔内挥刀按键持续不足一半。最初按按住时长选最近的组合，会漏掉这种短按。第二轮同时保留短按事件、显式提供最近本体和飞镖的相对位置，并提高低频示范动作的预训练权重。这三项作为一个整体实测，无法从组间比较拆分各自贡献。[按键统计](combat/pulse-audit.json) · [第二轮 GPU 训练](combat/training.json)

## 世界模型在新策略身上预测得怎样

用修正版 actor 在实机训练期间产生的 48 场、8522 步做事后检查。冻结的动力学模型未用这些轨迹拟合。给它实际选出的动作，并沿用想象训练时每步 0.12 秒的约定，用三个成员分别预测五步。标准化状态的平均绝对误差为 0.568 至 0.592，保持起始状态的误差为 0.639。实际按键间隔会浮动。

奖励估计偏乐观。8522 步的预测即时回报均值为 +0.015，游戏事件算出的均值为 −0.171。558 个挥刀决策的两项均值分别为 +0.206 和 −0.272。这些数字描述该策略实际访问的状态与动作，也包含同一间隔的受伤、死亡等事件。[逐组误差](combat/transfer-audit.json) · [复查脚本](../../scripts/audit_sisters_model_transfer.py)

跳过想象训练的对照组也有同方向的误差。其 48 场、4765 步的平均预测回报为 +0.026，平均实机回报为 −0.307。613 个挥刀决策分别为 +0.182 和 −0.450。两组完整的 24 批实机训练数据均可下载，在本地用同一脚本复算。[对照组误差](combat/transfer-audit-real-only.json) · [实机 PPO 采样轨迹](combat/on-policy-rollouts.zip)

```powershell
Expand-Archive models/sisters-course-ac/combat/on-policy-rollouts.zip -DestinationPath runs/course-replay
.\.venv\Scripts\python.exe scripts/audit_sisters_model_transfer.py --dynamics models/sisters-course-ac/combat/dynamics.pt --rollouts runs/course-replay/combat-real-train --output runs/course-replay/transfer-audit.json
```
