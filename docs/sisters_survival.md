# 战斗姐妹的生存训练

在 3 号槽的神居战斗姐妹竞技场，用同一版 Mod、相同的评估参数分别运行三套模型各 10 场。训练种子均为 0，评估采用各自冻结的策略。Unity 内部攻击序列仍会变化。

| 指标 | 旧进攻模型 | 上轮生存模型 | 本轮避刺与反击 |
|---|---:|---:|---:|
| 对 Boss 造成伤害的场数 | 10/10 | 0/10 | 10/10 |
| 平均对 Boss 伤害 | 113.4 | 0 | 77.7 |
| 地刺扣血总量 | 26 | 70 | 5 |
| 全程未被地刺扣血 | 0/10 | 0/10 | 6/10 |
| 本体扣血总量 | 46 | 7 | 54 |
| 飞镖扣血总量 | 18 | 13 | 31 |
| 来源未知的扣血总量 | 0 | 0 | 0 |
| 平均存活物理步 | 1207.8 | 1481.4 | 1334.6 |
| 平均首次受伤物理步 | 78.8 | 88.7 | 100.8 |
| 每千物理步损失 HP | 7.45 | 6.08 | 6.74 |
| 每千物理步地刺扣血 | 2.15 | 4.73 | 0.37 |
| 胜场 | 0/10 | 0/10 | 0/10 |

本轮在 10 场中都打出了实际伤害。与旧进攻模型相比，地刺少扣 21 HP，平均多活 126.8 个物理步，每场少打 35.7 点伤害。与上轮生存模型相比，地刺少扣 65 HP，每场恢复 77.7 点输出，平均少活 146.8 个物理步。两组旧权重都保留在原目录。三姐妹合击阶段尚待实战验证。

首伤时间从回合开始计数，没有受伤则计到回合结束。每千物理步损失 HP 按全部回合的总掉血与总物理步计算。同一回合可能被多个来源击中，每次实际扣血单独归因。

## 训练与核查

- 生存模型额外采集了 12 场完整示范，共 4,039 个决策步，按整场分为 9 场训练、3 场验证。模仿学习后再跑 10,240 步 PPO。此前的进攻模型保留为对照。
- 其中 8 场实际记录到两枚不同 ID 的敌方飞镖同屏。带伤害的两侧地刺各占一个实体。该批示范最多观察到 1 名活跃 Boss。三名 Boss 同时出现时的避让和实体顺序变化另有合成回归测试。
- 示范策略可在边缘停步、移动或起跳。这三类动作均由游戏当前的动作掩码决定。已核查的受伤按本体、飞镖、地刺及未知分组。脚本入口为 `scripts/collect_sisters_demos.py --teacher survival` 和 `scripts/clone_sisters_policy.py --objective survival`，任务配置为 `configs/experiments/sisters_survival.yaml`。
- 对照权重 SHA-256 为 `438b56a856262e2c27359f711d200cf7416a1aac67cffeff94655f7fbfe62e00`，生存模型为 `5adaebf5f390c500c7679aa7eda81533e03a27a971afb17e46fe2f5994a6d9f5`。模型、存档、逐步游戏回放和本机连接配置保存在本地。

## 松开方向键的复核

同样的 300 个验证帧里，示范者松开了方向键。给两套权重输入相同的示范状态与上一动作，模仿学习正确复现 250 帧，接续 PPO 后正确复现 11 帧。正式评估中，PPO 的 3,325 个基础移动决策包含 1 次松开方向键。相同状态下的复核能把 PPO 阶段的行为变化和不同回合的攻击序列区分开来。

## 同时训练落地避刺和反击

在相同竞技场重新采了 10 场脚本示范，共 3,388 个决策步。普通挥刀与同一 Boss 的 HP 下降匹配 43 次，10 场均有命中，地刺合计扣 1 HP。其中 13 次从左侧挥刀命中、24 次从右侧命中，6 次出刀时水平位置重叠。10 次先转向、接近再命中，左右两个方向都有。示范分为 8 场训练（2,590 步）和 2 场留出（798 步）。这组示范最多同时出现 1 个 Boss、2 枚敌方飞镖。

从旧进攻模型权重开始，用新的 Adam 优化器。每一次 PPO 小批量更新同时抽取 3 段示范，复习移动、起跳和出刀。击中一次造成 21 点伤害对应 +6.3 奖励，自己掉 1 HP 对应 −12。持续存活或单纯按攻击键均不加分。模型额外使用两侧刺区与小骑士的运动方向估算落点。低空接近危险区域时降低向外走的概率，空中足够高时恢复原有分布，各方向、起跳和普通挥刀仍由策略选择。详见 [ADR-0012](adr/0012-sisters-landing-prior-and-demonstrations.md)。

### 冻结权重的 10 场评估

训练完成 20 次更新，共 10,240 个决策步，采样失败和丢弃的步数均为 0。下表按回合结束顺序列出真实 HP 变化。最终检查点 SHA-256 为 `304bf07096c6ba45c2ed6a798004e3b32c85b123abcb8088d565306058eb8ef7`。

| 场次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 对 Boss 伤害 | 84 | 42 | 84 | 84 | 42 | 84 | 42 | 84 | 126 | 105 |
| 地刺扣血 | 2 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

逐步回放中有 37 个决策步造成 Boss 扣血，其中 24 个在前后各 12 个物理步内均未记录到自己掉血，每场至少出现一次。时间取相应决策步的结束时刻。共计向左 968 次、松开方向键 1,394 次、向右 763 次，另有 1 次宏动作。

在两场留出示范的 283 个停步帧上，本轮模型跟对 242 帧，上轮生存模型跟对 9 帧。9 个示范短跳时机，本轮均未复现。正式评估中记录到 86 个长按跳跃的决策步，基础动作里短跳和冲刺均为 0。下一轮可围绕本体突进和低位飞镖采集成功闪避、落地后回身命中的连续片段，增加这类稀有片段在示范回放中的占比。用本体与飞镖扣血、命中场次和踩刺情况共同验收。

### 复现本轮训练

在游戏和训练 Mod 已就绪的本机上，用以下 PowerShell 命令在独立目录复现实验。旧进攻权重保存在原运行目录。

```powershell
. ./scripts/windows/game_runtime.ps1
$null = Initialize-HKRLGameRuntime (Get-Location).Path
$py = '.venv/Scripts/python.exe'
$run = 'runs/training/sisters-counter-replica'
$cfg = 'configs/train/sisters_counter_v2.yaml'
$task = 'configs/experiments/sisters_counter.yaml'
$control = 'runs/training/sisters-gru-bc-ppo-seed0/checkpoints/checkpoint_v000021.pt'
& $py scripts/collect_sisters_demos.py --teacher counter --config $cfg --task $task `
    --episodes 10 --output "$run/demos"
& $py scripts/train.py --config $cfg --task $task --run-dir $run `
    --init-weights $control --demo-replay "$run/demos" --updates 20 `
    --checkpoint-every 1 --reset-timeout 90 --save-rollouts
& $py scripts/run_eval.py --policy model --train-config $cfg --tasks $task `
    --checkpoint-dir "$run/checkpoints" --local-session $run --episodes 10 `
    --seeds 10001 10002 10003 --output "$run/eval.json" --replay-jsonl "$run/eval-steps.jsonl"
```

从已有检查点接续时可以省略两项初始化数据参数，训练程序会按清单复核示范文件的哈希。游戏存档、逐步回放和权重保存在本机。

## 为什么先做短训练，以及怎样进入长训练

一次决策步包括读取游戏状态、选择一次动作并取得下一次观测。一个动作可能持续多个物理步。一批 512 个决策步，累计 20 批才是前面各轮的 10,240 步。单个检查点可以一直续训；不同实验的步数能否合并，要看它们继承了哪些权重和优化器状态。进攻与生存对照分别训练，上一节的避刺与反击模型继承了进攻模型的权重。

前几轮依次核对真实伤害事件、调整 GRU 输入、采集示范和测试奖励。生存轮的 0/10 命中暴露了退化，加大该轮预算缺少依据。避刺与反击轮恢复到 10/10 命中，建立了下一阶段的起点。本阶段固定同一套奖励、GRU 和 PPO 配置，比较加入闪避示范并连续训练后，实战能走多远。

- 在 3 号槽录下螳螂突进前的跳跃、上方落砸时的侧移或冲刺，以及躲过飞镖后的出刀。每四场留出一整场做模仿验证。
- 用游戏记录检查动作是否实际起跳或进入冲刺、附近是否掉血。训练中额外抽取包含这些动作的时间段。18 个物理步内无伤是本实验的筛选窗口；动作后 50 个物理步内无换血且出现已核实的普通挥刀命中，记为后续反击。它们描述观察到的结果，无法证明单个动作避免了原本必中的攻击。
- 从上轮正式检查点微调模仿策略，再用新的 PPO 优化器继续实机训练。新阶段的首个检查点在 4,096 个决策步，随后可在同一目录恢复全部权重、优化器和采样状态，累计到 32,768 步。每个里程碑冻结权重，单独实打 10 场。10 场里命中至少 8 场、场均输出至少 60、本体与飞镖合计每千物理步掉血不高于 7、地刺每千步掉血不高于 0.8，或出现胜场，才扩大这一路的预算。这些数值用于开发筛选。上一轮相应为 10 场、77.7、6.37 和 0.37。
- 判断模型能否用于通关，最终要看独立对局的胜率、能否打进合击阶段、受伤与用时，以及不同游戏攻击序列下的表现。累计步数和训练奖励都不能替代这项验收。

复用上节的 PowerShell 环境和 `$cfg`、`$task`，选择新的运行目录。以下更新数均指本目录累计完成的 PPO 批数。

```powershell
$run = 'runs/training/sisters-dodge-stage'
$control = 'runs/training/sisters-counter-v2-seed0/checkpoints/checkpoint_v000021.pt'
& $py scripts/collect_sisters_demos.py --teacher dodge_counter --config $cfg --task $task `
    --episodes 12 --output "$run/demos"
& $py scripts/fit_sisters_dodges.py --config $cfg --task $task --demos "$run/demos" `
    --output "$run/imitation" --init-weights $control
& $py scripts/train.py --config $cfg --task $task --run-dir $run --updates 8 `
    --init-weights "$run/imitation/bc-weights.pt" --demo-replay "$run/imitation/demos" `
    --reset-timeout 90 --checkpoint-every 4 --save-rollouts
```

每个里程碑按上一节的评估命令使用新的结果文件名。通过实战筛选后，同一训练命令改为 `--updates 64` 即可从第 8 批接续。
