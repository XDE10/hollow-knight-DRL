<p align="right"><b>English</b> | <a href="README.zh-CN.md">简体中文</a></p>

# HK-RL — Hollow Knight Deep Reinforcement Learning

[![CI](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/ci.yml)
[![C# Mod Build](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml/badge.svg)](https://github.com/XDE10/hollow-knight-DRL/actions/workflows/mod-build.yml)

> A structured game-state Deep RL framework and verified native-control champion for *Hollow Knight* (Godhome / Hall of Gods), built on **HKRLEnvMod (C#)** + **PyTorch** + **FlatBuffers**.

Based on [myouo/hk_Rl](https://github.com/myouo/hk_Rl) (MIT License and commit history preserved).

---

## Sisters of Battle Champion Highlights

The repository ships a **ready-to-run frozen native C# neural policy** ([`models/sisters_champion/sisters_champion.bin`](models/sisters_champion/sisters_champion.bin), `17.38 MB`) trained via a 3-stage **Offline IQL + 3-Ensemble World Model → Online World-Model PPO → Radiant 1-Hit GRPO-PPO + Low-Temperature Annealing** pipeline.

Evaluated on **100 completely unseen deterministic seeds (`19900001..19900100`)** at **1.0× original game speed, 1× nail damage, 9 initial masks, 0 initial soul**, with untouched original `HKRLEnvMod.dll` and zero external inference servers:

| Stage / Candidate | 100-Fight Win Rate | Full No-Hit Wins | Trio-Phase No-Hit | Avg Damage / Fight |
| :--- | :---: | :---: | :---: | :---: |
| Stage 1 · Offline IQL + AWR + 3-Ensemble World Model | 100% (30/30 Val) | 0% | 0% | 3.27 masks |
| Stage 2 · Online World-Model Regularized PPO (Goal 1) | **100 / 100 (100%)** | 31 / 100 (31%) | — | 1.33 masks |
| **Stage 3 · Radiant GRPO-PPO + Annealing (`sisters_champion.bin`)** | **100 / 100 (100%)** | **65 / 100 (65%)** | **67 / 100 (67%)** | **0.47 masks** |

* Full model card & checksums: [`models/sisters_champion/README.md`](models/sisters_champion/README.md)
* Full methodology, World Model lineage & scaling ablation: [`docs/sisters_champion.md`](docs/sisters_champion.md)
* 100-fight verified summary & per-episode JSON receipts: [`models/sisters_champion/formal100_summary.json`](models/sisters_champion/formal100_summary.json)

---

## 5-Minute Quick Start (Clone & Watch the Champion)

### 1. Clone & Install Python Environment

```bash
git clone https://github.com/XDE10/hollow-knight-DRL.git
cd hollow-knight-DRL

# Option A: Conda (recommended, matches CI)
conda env create -f environment.yml
conda activate hkrl

# Option B: pip / venv (Python 3.10+)
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux: source .venv/bin/activate
pip install -e "python[dev]"
```

### 2. Build & Install `HKRLEnvMod` into Hollow Knight

Requires Hollow Knight (1.5.78 with HK Modding API installed), .NET 8 SDK, and `flatc 23.5.26`:

```bash
# Generate C# FlatBuffers bindings (requires flatc 23.5.26)
flatc --csharp -o mod/HKRLEnvMod/Schema schema/hkrl.fbs

# Build against your Hollow Knight Managed directory
dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj -c Release \
  -p:HollowKnightManaged="/path/to/Hollow Knight/hollow_knight_Data/Managed"

# Copy built DLL into Hollow Knight Mods folder
cp mod/HKRLEnvMod/bin/Release/HKRLEnvMod.dll \
  "/path/to/Hollow Knight/hollow_knight_Data/Managed/Mods/HKRLEnvMod/"
```

Ensure save slot 3 is parked in Godhome (Hall of Gods) with 9 masks / pure nail (1×).

### 3. Watch the Champion Defeat Sisters of Battle

`scripts/sisters/play.py` automatically downloads and SHA-256-verifies `sisters_champion.bin` (`17.38 MB`) from GitHub Release [`v1.0.0-sisters-champion`](https://github.com/XDE10/hollow-knight-DRL/releases/tag/v1.0.0-sisters-champion) on first run:

```bash
# Auto-detects Steam Hollow Knight and launches a visible 1280x720 fight
python scripts/sisters/play.py

# Or specify game path / run 5 deterministic fights & save JSON receipts
python scripts/sisters/play.py \
  --game-root "D:/Steam/steamapps/common/Hollow Knight" \
  --episodes 5 --seed 19900001 --output runs/sisters_play.json

# Windows PowerShell shortcut
powershell -ExecutionPolicy Bypass -File scripts/windows/play_sisters.ps1 -Episodes 1
```

---

## Reproducing Training & 100-Fight Formal Evaluation

All Sisters of Battle scripts live cleanly under `scripts/sisters/`:

```bash
# Parallel multi-instance Formal-100 verification (headless Linux / multi-core)
python scripts/sisters/eval_parallel.py \
  --mode eval \
  --actor-bin models/sisters_champion/sisters_champion.bin \
  --run-dir runs/formal100 \
  --first-seed 19900001 --episodes 100 --workers 32 --nographics

# Stage 1: Offline IQL + AWR + 3-Ensemble World Model + occupancy/motion heads
python scripts/sisters/train_offline_wm_iql.py --help

# Stage 2 & 3: Online World-Model PPO -> Radiant 1-Hit GRPO-PPO + Low-Temp Annealing
python scripts/sisters/train_online_ppo.py --help
```

---

## Repository Structure

```text
HKRL/
├── schema/                   ★ Single source of truth (hkrl.fbs) for C#/Python FlatBuffers
├── mod/
│   ├── HKRLEnvMod/           C# Environment Server + native V2 neural actor runtime
│   ├── ci-stubs/             Minimal Unity / HK / Modding API stubs for headless CI builds
│   └── ci-tests/             C# input-injection & action-history smoke tests
├── python/
│   ├── hkrl/                 Core Gymnasium env, models, PPO/APPO, worker/learner, eval
│   └── tests/                Unit test suite (core framework + champion integrity checks)
├── models/
│   └── sisters_champion/     ★ Frozen native champion (.bin), World Model, & Formal-100 receipts
├── scripts/
│   ├── sisters/              ★ Sisters of Battle playback, parallel eval, & 3-stage DRL pipeline
│   ├── windows/              Windows playback & game PC helper scripts
│   ├── linux/                Linux game PC + SSH remote GPU helpers
│   ├── remote/               Remote GPU learner bootstrap scripts
│   └── *.py / *.sh           Core HKRL framework CLIs (train, eval, phase8, mod CI/release)
├── configs/                  44 Godhome boss tasks, train configs, & Sisters champion config
├── docs/                     Architecture specs, ADRs (0001–0011), & Sisters Champion report
├── notebooks/                Colab / Kaggle remote learner setup notebooks
├── Makefile                  Developer quality gates (`make check`, `make test`, `make lint`)
└── AGENTS.md                 Project engineering charter & six architectural invariants
```

---

## Core HKRL Framework Architecture & Commands

```text
Hollow Knight + HKRLEnvMod (C#)        Game PC                  Remote GPU
  ├─ Observation collection             ┌──────────────┐         ┌──────────────┐
  ├─ Action apply (FixedUpdate)  <───>  │ GameWorker   │ ──────> │ Learner      │
  ├─ Reward events                      │ Local infer  │ rollout │ PPO / APPO   │
  └─ Clean Episode Lifecycle            │ Gym Env      │ <────── │ Checkpoints  │
       │ FlatBuffers over TCP           └──────────────┘ weights └──────────────┘
       └────────────────────────────────────┘
```

Common framework commands (see [`docs/architecture.md`](docs/architecture.md) and [`docs/linux_ssh_deployment.md`](docs/linux_ssh_deployment.md)):

```bash
# Run full local quality gate (schema codegen + ruff format/lint + mypy + pytest)
make check

# Run C# mod CI build & smoke test (requires flatc 23.5.26 + .NET 8)
bash scripts/build_mod_ci.sh

# Ping a live HKRLEnvMod instance without resetting the scene
python scripts/check_env.py --config configs/train/ppo_mlp.yaml --task configs/tasks/gruz_mother.yaml

# Train / evaluate general multi-boss policies
python scripts/train.py --config configs/train/ppo_attention_gru.yaml --task configs/tasks/gruz_mother.yaml --smoke
python scripts/run_eval.py --policy scripted --tasks configs/tasks/gruz_mother.yaml --episodes 5

# Offline distributed wiring smoke (no live game required)
make phase8-smoke
```

## License

MIT — see [`LICENSE`](LICENSE).
