<p align="right"><a href="README.md">English</a> | <b>简体中文</b></p>

# HK-RL — 空洞骑士深度强化学习

[![CI](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml)
[![C# Mod Build](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml)

> 面向《空洞骑士》（Godhome / 神居）的结构化游戏状态深度强化学习框架，以及经过验证的原生控制冠军策略。基于 **HKRLEnvMod (C#)** + **PyTorch** + **FlatBuffers** 构建。

基于 [myouo/hk_Rl](https://github.com/myouo/hk_Rl)（保留 MIT 许可证与提交历史）。

---

## 战斗姐妹（Sisters of Battle）冠军亮点

仓库内附带一个**开箱即用的冻结原生 C# 神经网络策略**（[`models/sisters_champion/sisters_champion.bin`](models/sisters_champion/sisters_champion.bin)，`17.38 MB`），通过三阶段流程训练：**离线 IQL + 3 集成世界模型 → 在线世界模型 PPO → Radiant 1-Hit GRPO-PPO + 低温退火**。

评测条件：**100 个完全未见过的确定性种子（`19900001..19900100`）**，**1.0× 原始游戏速度、1× 骨钉伤害、9 个初始面具、0 初始灵魂**，使用未修改的原版 `HKRLEnvMod.dll`，无任何外部推理服务器：

| 阶段 / 候选 | 100 场胜率 | 全程无伤胜利 | 三人阶段无伤 | 平均每场受伤 |
| :--- | :---: | :---: | :---: | :---: |
| 阶段 1 · 离线 IQL + AWR + 3 集成世界模型 | 100%（验证集 30/30） | 0% | 0% | 3.27 面具 |
| 阶段 2 · 在线世界模型正则化 PPO（Goal 1） | **100 / 100 (100%)** | 31 / 100 (31%) | — | 1.33 面具 |
| **阶段 3 · Radiant GRPO-PPO + 退火（`sisters_champion.bin`）** | **100 / 100 (100%)** | **65 / 100 (65%)** | **67 / 100 (67%)** | **0.47 面具** |

* 完整模型卡与校验和：[`models/sisters_champion/README.md`](models/sisters_champion/README.md)
* 完整方法、世界模型谱系与规模消融：[`docs/sisters_champion.md`](docs/sisters_champion.md)
* 100 场验证汇总与逐场 JSON 回执：[`models/sisters_champion/formal100_summary.json`](models/sisters_champion/formal100_summary.json)

---

## 5 分钟快速上手（克隆并观看冠军）

### 1. 克隆仓库并安装 Python 环境

```bash
git clone https://github.com/XDE10/hollow-knight-DRL.git
cd hollow-knight-DRL

# 方案 A：Conda（推荐，与 CI 一致）
conda env create -f environment.yml
conda activate hkrl

# 方案 B：pip / venv（Python 3.10+）
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux: source .venv/bin/activate
pip install -e "python[dev]"
```

### 2. 构建并将 `HKRLEnvMod` 安装到空洞骑士

需要 Hollow Knight（1.5.78，已安装 HK Modding API）、.NET 8 SDK 与 `flatc 23.5.26`：

```bash
# 生成 C# FlatBuffers 绑定（需要 flatc 23.5.26）
flatc --csharp -o mod/HKRLEnvMod/Schema schema/hkrl.fbs

# 针对你的 Hollow Knight Managed 目录构建
dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj -c Release \
  -p:HollowKnightManaged="/path/to/Hollow Knight/hollow_knight_Data/Managed"

# 将构建出的 DLL 复制到 Hollow Knight 的 Mods 文件夹
cp mod/HKRLEnvMod/bin/Release/HKRLEnvMod.dll \
  "/path/to/Hollow Knight/hollow_knight_Data/Managed/Mods/HKRLEnvMod/"
```

请确保存档位 3 停在 Godhome（万神殿），并拥有 9 个面具 / 纯净骨钉（1×）。

### 3. 观看冠军击败战斗姐妹

首次运行时，`scripts/sisters/play.py` 会自动从 GitHub Release [`v1.0.0-sisters-champion`](https://github.com/XDE10/hollow-knight-DRL/releases/tag/v1.0.0-sisters-champion) 下载 `sisters_champion.bin`（`17.38 MB`）并进行 SHA-256 校验：

```bash
# 自动检测 Steam 版 Hollow Knight，并以可见的 1280x720 窗口开战
python scripts/sisters/play.py

# 或指定游戏路径 / 运行 5 场确定性战斗并保存 JSON 回执
python scripts/sisters/play.py \
  --game-root "D:/Steam/steamapps/common/Hollow Knight" \
  --episodes 5 --seed 19900001 --output runs/sisters_play.json

# Windows PowerShell 快捷方式
powershell -ExecutionPolicy Bypass -File scripts/windows/play_sisters.ps1 -Episodes 1
```

---

## 复现训练与 100 场正式评测

所有战斗姐妹相关脚本都位于 `scripts/sisters/`：

```bash
# 并行多实例 Formal-100 验证（无头 Linux / 多核）
python scripts/sisters/eval_parallel.py \
  --mode eval \
  --actor-bin models/sisters_champion/sisters_champion.bin \
  --run-dir runs/formal100 \
  --first-seed 19900001 --episodes 100 --workers 32 --nographics

# 阶段 1：离线 IQL + AWR + 3 集成世界模型 + 占据/运动预测头
python scripts/sisters/train_offline_wm_iql.py --help

# 阶段 2 & 3：在线世界模型 PPO -> Radiant 1-Hit GRPO-PPO + 低温退火
python scripts/sisters/train_online_ppo.py --help
```

---

## 仓库结构

```text
HKRL/
├── schema/                   ★ C#/Python FlatBuffers 的唯一事实来源 (hkrl.fbs)
├── mod/
│   ├── HKRLEnvMod/           C# 环境服务器 + 原生 V2 神经网络 actor 运行时
│   ├── ci-stubs/             用于无头 CI 构建的最小 Unity / HK / Modding API 桩
│   └── ci-tests/             C# 输入注入与动作历史冒烟测试
├── python/
│   ├── hkrl/                 核心 Gymnasium 环境、模型、PPO/APPO、worker/learner、评测
│   └── tests/                单元测试（核心框架 + 冠军完整性检查）
├── models/
│   └── sisters_champion/     ★ 冻结的原生冠军 (.bin)、世界模型与 Formal-100 回执
├── scripts/
│   ├── sisters/              ★ 战斗姐妹回放、并行评测与三阶段 DRL 流程
│   ├── windows/              Windows 回放与游戏 PC 辅助脚本
│   ├── linux/                Linux 游戏 PC + SSH 远程 GPU 辅助脚本
│   ├── remote/               远程 GPU learner 引导脚本
│   └── *.py / *.sh           HKRL 核心框架 CLI（train、eval、phase8、mod CI/发布）
├── configs/                  44 个 Godhome Boss 任务、训练配置与战斗姐妹冠军配置
├── docs/                     架构规范、ADR (0001–0011) 与战斗姐妹冠军报告
├── notebooks/                Colab / Kaggle 远程 learner 配置 notebook
├── Makefile                  开发质量门禁（`make check`、`make test`、`make lint`）
└── AGENTS.md                 项目工程章程与六条架构不变量
```

---

## HKRL 核心框架架构与常用命令

```text
Hollow Knight + HKRLEnvMod (C#)        游戏 PC                   远程 GPU
  ├─ 观测采集                           ┌──────────────┐         ┌──────────────┐
  ├─ 动作应用 (FixedUpdate)      <───>  │ GameWorker   │ ──────> │ Learner      │
  ├─ 奖励事件                           │ 本地推理     │ rollout │ PPO / APPO   │
  └─ 干净的 Episode 生命周期            │ Gym Env      │ <────── │ Checkpoints  │
       │ FlatBuffers over TCP           └──────────────┘ 权重    └──────────────┘
       └────────────────────────────────────┘
```

常用框架命令（参见 [`docs/architecture.md`](docs/architecture.md) 与 [`docs/linux_ssh_deployment.md`](docs/linux_ssh_deployment.md)）：

```bash
# 运行完整本地质量门禁（schema 代码生成 + ruff 格式化/检查 + mypy + pytest）
make check

# 运行 C# mod CI 构建与冒烟测试（需要 flatc 23.5.26 + .NET 8）
bash scripts/build_mod_ci.sh

# 在不重置场景的情况下 ping 运行中的 HKRLEnvMod 实例
python scripts/check_env.py --config configs/train/ppo_mlp.yaml --task configs/tasks/gruz_mother.yaml

# 训练 / 评测通用多 Boss 策略
python scripts/train.py --config configs/train/ppo_attention_gru.yaml --task configs/tasks/gruz_mother.yaml --smoke
python scripts/run_eval.py --policy scripted --tasks configs/tasks/gruz_mother.yaml --episodes 5

# 离线分布式联通冒烟测试（无需运行游戏）
make phase8-smoke
```

## 许可证

MIT — 见 [`LICENSE`](LICENSE)。
