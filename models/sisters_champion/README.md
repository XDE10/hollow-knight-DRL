# Sisters of Battle — Frozen Native Champion Model

This directory ships the **final verified frozen neural actor** for *Hollow Knight: Sisters of Battle* (Godhome), ready to run directly inside `HKRLEnvMod` at **1.0× original game speed, 1× nail damage, 9 initial masks, and 0 initial soul** with zero external inference servers.

## Formal 100-Fight Acceptance Result

Evaluated across **100 completely unseen deterministic seeds (`19900001..19900100`)** on untouched original `HKRLEnvMod.dll` (`sha256: abf9f0ca1b109384...`):

| Candidate | Wins | Full No-Hit Wins | Trio-Phase No-Hit | Avg Damage / Fight | Evidence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`sisters_champion.bin` (Plan B R06)** | **100 / 100 (100%)** | **65 / 100 (65.0%)** | **67 / 100 (67.0%)** | **0.47 masks** | [`formal100_summary.json`](./formal100_summary.json) · [`formal100_receipts.tar.gz`](./formal100_receipts.tar.gz) |
| Runner-up control (Plan B R12) | 100 / 100 (100%) | 61 / 100 (61.0%) | 64 / 100 (64.0%) | 0.50 masks | [`summary.json`](./summary.json) |
| Stage-2 Goal-1 Baseline (WM-PPO R10) | 100 / 100 (100%) | 31 / 100 (31.0%) | — | 1.33 masks | Stage-2 milestone |

## Architecture & 3-Stage World Model + RL Lineage

1. **Model (`AttnRecurrentActor`, V2 binary magic `0x3F4C5149`, ~3.5M params)**:
   - 6-token entity self-attention (`Hero + 3 Bosses + 2 Discs`) + 4-lag causal motion history (`0.08s, 0.16s, 0.32s, 0.48s`) + Pre-LN ResFFN + 2-layer Residual GRU (`d_model=320`).
   - Dual-phase frozen `.bin` (`17.38 MB`) bundling the dedicated Phase-1 opener and Phase-2 Trio actor for native C# execution inside `SistersScaledRecurrent.cs`.
2. **Stage 1 — Offline IQL + AWR + 3-Ensemble World Model (`scripts/sisters/train_offline_wm_iql.py`)**:
   - Trained offline Actor/Critic jointly with a **3-ensemble action-conditioned recurrent world model** (`world_model/world.pt`, `world_model/validation.json`) plus 54-dim future occupancy (`occ_head`) and future motion (`fmov_head`) auxiliary heads.
   - Achieved `30/30` (100%) win baseline before any online RL (avg damage `3.27`).
3. **Stage 2 — Online World-Model Regularized PPO (`scripts/sisters/train_online_ppo.py`)**:
   - Joint on-policy GPU PPO with live `world_aux` dynamics regularization from the verified 3-ensemble world model, Smooth-L1 Critic warmup, and monotonic KL trust region.
   - Passed Goal 1 (`100/100` wins, `31/100` full no-hit, avg damage `1.33`).
4. **Stage 3 — Radiant 1-Hit Stop + Group-4 GRPO-PPO + Low-Temperature Annealing**:
   - Annealed offline KL / world-anchor weight (`world_aux_decay → 0`), added protocol-level 1-hit early termination, per-seed Group-4 GRPO contrast, and low-temperature policy annealing (`T = 0.08..0.22`, entropy stabilized at `~0.22`).
   - Slashed average damage by 65% (`1.33 → 0.47`) and more than doubled blind no-hit rate (`31% → 65%`, peak 32-game holdout `22/32 = 68.8%`).

> See [`docs/sisters_champion.md`](../../docs/sisters_champion.md) for the full technical write-up, including the scaling ablation comparing this 3.5M native V2 actor against an 8.65M external TCP GPU actor with an online latent world-model head.

## Artifacts & Checksums

| File | Description | SHA-256 (prefix) | Distribution |
| :--- | :--- | :--- | :--- |
| `sisters_champion.bin` | Frozen native dual-phase C# actor (`17.38 MB`) | `f97d6aba7911a498` | GitHub Release `v1.0.0-sisters-champion` (auto-downloaded by `play.py`) |
| `world_model/world.pt` | Stage-1 3-ensemble recurrent world model (`1.74 MB`) | Verified in `validation.json` | GitHub Release `v1.0.0-sisters-champion` |
| `formal100_summary.json` | Aggregated 100-fight strict 1.0× evaluation summary | — | Tracked in Git |
| `formal100_receipts.tar.gz` | All 100 individual per-episode JSON receipts | — | Tracked in Git |
| `training_history.json` | Stage-3 R0–R12 progression log | — | Tracked in Git |
| `summary.json` | Head-to-head R06 vs R12 comparison & full SHA-256 digests | — | Tracked in Git |

*(Full 49.6 MB PyTorch training checkpoint `FINAL_BEST_B_CHAMPION.pt`, sha256 `7165edaa0a825010...`, is archived externally to keep Git clone fast.)*

## Run Immediately

```bash
# Watch 1 fight in a visible Hollow Knight window (auto-detects Steam install)
python scripts/sisters/play.py

# Run 5 deterministic fights and save receipts
python scripts/sisters/play.py --episodes 5 --seed 19900001 --output runs/play_receipts.json

# Windows PowerShell shortcut
powershell -ExecutionPolicy Bypass -File scripts/windows/play_sisters.ps1 -Episodes 1
```
