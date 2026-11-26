# Sisters of Battle — Methodology, World Model Lineage & Results

This document records the complete engineering and algorithmic lineage behind the frozen native **Sisters of Battle** champion shipped in [`models/sisters_champion/`](../models/sisters_champion/README.md).

## 1. Problem Formulation & Strict Evaluation Protocol

- **Environment**: *Hollow Knight* Godhome — Sisters of Battle (Phase 1 single Mantis Lord → Phase 2 simultaneous Trio + boomerang discs + spike edges).
- **Control Interface**: Native discrete primitive action catalogue executed at 25 Hz (`0.04s` step) inside `HKRLEnvMod` (`SistersScaledRecurrent.cs`), with zero vision shortcuts, zero invulnerability hacks, and zero external Python servers at evaluation time.
- **Strict Acceptance Conditions**:
  - `1.0×` original game speed, `1×` nail damage, `9` initial masks, `0` initial soul;
  - Untouched original `HKRLEnvMod.dll` (`sha256: abf9f0ca1b109384...`);
  - 100 completely unseen deterministic seeds (`19900001..19900100`).

## 2. Model Architecture (`AttnRecurrentActor`, V2 Magic `0x3F4C5149`)

Implemented in `scripts/sisters/_lib/wp22_attn_model.py` (re-exported via `scripts/sisters/models.py`) and executed natively in C# by `mod/HKRLEnvMod/Action/SistersScaledRecurrent.cs`:

- **Observation**: 192-dim causal timing state (`SistersTimingState.cs`) augmented with **4-lag causal motion history** (`Δt = 0.08s, 0.16s, 0.32s, 0.48s`).
- **Entity Self-Attention**: 6 structured entity tokens (`Hero` + `3 Bosses` + `2 Discs`) with explicit entity validity masking so inactive slots leak nothing.
- **Spatial-Temporal Backbone**: Pre-LN ResFFN blocks + **2-layer Residual GRU** (`d_model = 320`, ~3.5M parameters total).
- **Dual-Phase Packaging**: Single self-contained frozen binary (`sisters_champion.bin`, `17.38 MB`) embedding both the dedicated Phase-1 opener and the Phase-2 Trio actor.

## 3. Three-Stage World Model + Reinforcement Learning Pipeline

### Stage 1 — Offline IQL + AWR + 3-Ensemble World Model (`train_offline_wm_iql.py`)
- Trained offline Actor/Critic via Implicit Q-Learning (IQL) and Advantage-Weighted Regression (AWR) on audited trajectories.
- Explicitly trained a **3-ensemble action-conditioned recurrent dynamics world model** (`models/sisters_champion/world_model/world.pt`, validated in `validation.json`) predicting multi-step kinematics and damage, alongside **54-dim future spatial occupancy (`occ_head`)** and **future movement (`fmov_head`)** auxiliary heads.
- **Result**: `30/30` (100%) win rate at zero online updates, `3.27` average damage, `0/30` no-hit.

### Stage 2 — Online World-Model Regularized PPO (`train_online_ppo.py`)
- Closed-loop on-policy GPU PPO with 32 parallel headless Linux workers.
- Loaded the verified Stage-1 3-ensemble world model on every update to supply `world_aux` dynamics regularization (converging steadily from `0.00236` to `0.00185`), paired with Smooth-L1 Critic warmup and dual-KL monotonic trust region.
- **Result**: Cleared Goal 1 in Formal-100 evaluation with **`100/100` wins (100%)**, **`31/100` full no-hit**, and **`1.33` average damage**.

### Stage 3 — Radiant 1-Hit Stop + Group-4 GRPO-PPO + Low-Temp Annealing (`train_online_ppo.py`)
- Annealed offline KL and stale offline world-model anchor weight (`world_aux_decay → 0`) once online skill exceeded the offline dataset.
- Introduced protocol-level **1-hit early termination** during Trio rollouts, **same-seed Group-4 GRPO relative outcome contrast**, hard-seed replay pool, and **low-temperature annealing** (`T = 0.08..0.22`, stabilizing policy entropy around `~0.22`).
- **Result**: Slashed average damage by 65% (`1.33 → 0.47`), reached a peak unseen 32-game holdout of **`22/32 (68.8%)`** full no-hit (`23/32 = 71.9%` Trio no-hit, `0.406` avg dmg) at Round 6, and won the final 100-fight head-to-head acceptance test.

## 4. Final 100-Fight Head-to-Head Acceptance (Unseen Seeds `19900001..19900100`)

| Candidate | Wins | Full No-Hit | Trio No-Hit | Avg Damage | P1 Hits | Trio Hits (Body / Disc / Spike) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage-3 R06 (`sisters_champion.bin`)** | **100 / 100** | **65 / 100 (65%)** | **67 / 100 (67%)** | **0.47** | 6 | 41 (27 / 18 / 2) |
| Stage-3 R12 (`r12_champion.bin`) | 100 / 100 | 61 / 100 (61%) | 64 / 100 (64%) | 0.50 | 6 | 44 (28 / 21 / 1) |

## 5. Scaling & Architecture Ablation Summary (Plan A vs Plan B)

We also evaluated an **8.65M-parameter external GPU actor (`PlanADeepActor` + online multi-horizon latent `PlanAWorldModelHead`)** driven via a loopback binary TCP bridge at 25 Hz:

- **Finding**: Under identical `1.0×` original-speed sample budgets (64 rollouts/round across 32 workers), the 8.65M external model plateaued at `50%–56%` unseen holdout no-hit (`16–18/32`, entropy `~0.29`, `avg_dmg 0.63–0.66`) despite clean convergence of its online world-model head (`wm_dyn: 0.147 → 0.097`, `wm_risk_bce: 0.118 → 0.024`).
- **Takeaway**: Matching model capacity (~3.5M native V2) to the real-time `1.0×` sample budget and annealing policy entropy (`~0.22`) outperformed raw parameter scaling while preserving 100% native zero-dependency C# deployment.
