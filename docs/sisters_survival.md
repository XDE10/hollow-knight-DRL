# 战斗姐妹的生存训练

在 3 号槽的神居战斗姐妹竞技场，用同一版 Mod 和同一组评估参数分别运行旧模型与生存模型各 10 场。两者训练种子为 0，评估使用固定策略。Unity 内部的攻击序列仍会变化。这一组数据适合开发对照，三姐妹同时出战的表现需要另外实测。

| 指标 | 上一轮进攻模型 | 生存示范 + PPO |
|---|---:|---:|
| 平均存活物理步 | 1207.8 | 1481.4 |
| 平均首次受伤物理步 | 78.8 | 88.7 |
| 每千物理步损失 HP | 7.45 | 6.08 |
| 地刺扣血场数（共 10 场） | 10 | 10 |
| 地刺扣血总量 | 26 | 70 |
| 本体扣血总量 | 46 | 7 |
| 飞镖扣血总量 | 18 | 13 |
| 来源未知的扣血总量 | 0 | 0 |
| 对 Boss 造成伤害的场数 | 10 | 0 |
| 平均对 Boss 伤害 | 113.4 | 0.0 |
| 胜场 | 0 | 0 |

这 10 场的平均存活时间提高了约 23%，但两侧地刺造成的掉血从 26 增至 70，进攻也退化到 10 场都没有命中。新权重保留作生存实验的对照，上一轮仍是已经能稳定打出伤害的模型。

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
