# HKRL — Hollow Knight 战斗姐妹深度强化学习项目

[![CI](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml)
[![C# Mod Build](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml)

基于《空洞骑士》（Hollow Knight）内部结构化状态环境 `HKRLEnvMod (C#)` + `PyTorch` + `FlatBuffers`，通过 **离线 IQL + 三集成世界模型 → 在线世界模型正则 PPO → 辐辉 1 伤早停 GRPO-PPO + 低温退火** 三阶段深度强化学习，训练出可直接部署于游戏内原速运行的**战斗姐妹（Sisters of Battle）纯神经网络策略**。

> 本项目基于开源框架 [myouo/hk_Rl](https://github.com/myouo/hk_Rl) 扩展，保留原 MIT 许可与提交历史。

---

## 核心实机战绩

在 **1.0× 原速、1× 骨钉伤害、9 格血、0 灵魂、零外挂推理服务** 的严格条件下，于 **100 个全新未见随机种子（`19900001..19900100`）** 上实机验收：

| 训练阶段 | 100 场胜率 | 全场无伤 | 三姐妹阶段无伤 | 场均受伤 |
| :--- | :---: | :---: | :---: | :---: |
| 阶段 1 · 离线 IQL + AWR + 三集成世界模型 | 100%（30/30 验证） | 0% | 0% | 3.27 格 |
| 阶段 2 · 在线世界模型正则 PPO | **100 / 100 (100%)** | 31 / 100 (31%) | — | 1.33 格 |
| **阶段 3 · 辐辉 GRPO-PPO + 低温退火（最终冠军）** | **100 / 100 (100%)** | **65 / 100 (65%)** | **67 / 100 (67%)** | **0.47 格** |

* 完整算法原理、世界模型血统与消融对比：[`docs/sisters_champion.md`](docs/sisters_champion.md)
* 详细环境配置与存档准备指南：[`docs/setup_guide.md`](docs/setup_guide.md)
* 百局逐场验收回执与校验和：[`models/sisters_champion/README.md`](models/sisters_champion/README.md)

---

## 组员 3 步快速上手（无需安装 .NET / flatc）

### 第 1 步：克隆仓库并安装 Python 依赖

需要 Python 3.10+（推荐 Conda 或 venv）：

```bash
git clone https://github.com/XDE10/hollow-knight-DRL.git
cd hollow-knight-DRL

# 方式 A：pip + venv（最轻量）
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
pip install -e "python[dev]"

# 方式 B：Conda（与 CI 环境一致）
conda env create -f environment.yml
conda activate hkrl
```

### 第 2 步：安装 Mod（一键自动定位游戏目录）

前提：电脑已通过 Steam 安装正版《空洞骑士》（1.5.78），并已通过 [Lumafly](https://themulhima.github.io/Lumafly/) 安装 Modding API，且 **3 号存档**已解锁神居诸神堂（详见 [`docs/setup_guide.md`](docs/setup_guide.md)）。

运行以下一键脚本，会自动通过 Steam 注册表 / `libraryfolders.vdf` 搜索任意盘符下的游戏目录，从 GitHub Release 下载校验好的预编译 `HKRLEnvMod.dll` 并安装：

```bash
python scripts/install_mod.py

# 如需手动指定游戏目录：
python scripts/install_mod.py --game-root "/path/to/Hollow Knight"
```

### 第 3 步：一键运行观看 AI 无伤通关战斗姐妹

首次运行会自动从 GitHub Release [`v1.0.0-sisters-champion`](https://github.com/XDE10/hollow-knight-DRL/releases/tag/v1.0.0-sisters-champion) 下载 `sisters_champion.bin`（`17.38 MB`）并完成 SHA-256 校验，随后弹出 `1280×720` 游戏窗口实时演示：

```bash
# Windows PowerShell 快捷启动
.\scripts\windows\play_sisters.ps1

# 跨平台 Python 启动（支持自定义局数、随机种子、无头评估、保存回执）
python scripts/sisters/play.py
python scripts/sisters/play.py --episodes 5 --seed 19900001 --output runs/play_results.json
```

---

## 复现训练与百局并行评估

所有战斗姐妹相关脚本统一收口在 `scripts/sisters/`：

```bash
# 100 局多进程并行原速正式验收（适合多核 Linux / 无头评测）
python scripts/sisters/eval_parallel.py \
  --mode eval \
  --game-root "/path/to/Hollow Knight" \
  --actor-bin models/sisters_champion/sisters_champion.bin \
  --run-dir runs/formal100 \
  --first-seed 19900001 --episodes 100 --workers 16 --nographics

# 阶段 1：离线 IQL + AWR + 三集成世界模型训练
python scripts/sisters/train_offline_wm_iql.py --help

# 阶段 2 & 3：在线世界模型正则 PPO + 辐辉 1 伤 GRPO + 低温退火
python scripts/sisters/train_online_ppo.py --help
```

---

## 极简项目结构

```text
HKRL/
├── configs/
│   ├── sisters_train.yaml         # 扁平独立训练/通信配置（无嵌套继承）
│   ├── sisters_task.yaml          # 扁平独立战斗姐妹竞技场配置
│   └── sisters_acceptance.json    # 百局验收标准配置
├── docs/
│   ├── setup_guide.md             # 组员配置与排障手册（Modding API / 存档 / 源码编译）
│   └── sisters_champion.md        # 三阶段世界模型 + RL 算法原理与实验结果
├── models/
│   └── sisters_champion/          # 冠军模型元数据、百局回执与自动下载落盘目录
├── mod/
│   ├── HKRLEnvMod/                # C# 环境服务端 + 原生 V2 注意力 GRU 推理内核
│   ├── ci-stubs/                  # CI 编译桩程序集
│   └── ci-tests/                  # C# 输入注入冒烟测试
├── python/
│   ├── hkrl/                      # Gymnasium 环境封装、FlatBuffers 协议与工具库
│   └── tests/                     # 精简核心单测（含配置完整性与零硬编码守护）
├── schema/
│   └── hkrl.fbs                   # C# 与 Python 共用的 FlatBuffers 唯一协议定义
└── scripts/
    ├── install_mod.py             # 一键自动查找 Steam 并安装预编译 HKRLEnvMod.dll
    ├── build_mod_ci.sh            # C# Mod CI 编译脚本
    ├── gen_schema.sh              # FlatBuffers 绑定生成脚本
    ├── windows/play_sisters.ps1   # Windows PowerShell 一键启动脚本
    └── sisters/                   # 演示、并行评估与三阶段训练流水线
```

---

## 本地质量检查（开发者 / CI）

```bash
# 运行全部 Python 检查（生成协议绑定 + ruff 格式/规范 + mypy + pytest）
make check

# 验证 C# Mod CI 编译（需要 .NET 8 SDK 与 flatc 23.5.26）
bash scripts/build_mod_ci.sh
```

## License

MIT — 详见 [`LICENSE`](LICENSE)。
