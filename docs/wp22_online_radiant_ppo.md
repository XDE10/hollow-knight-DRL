# WP22 Scaled Entity-Attention & High-Concurrency Online Radiant PPO

This document summarizes the WP18–WP22 / V4 Radiant PPO architecture, native C# inference extensions, cloud Linux runner, and current empirical status for Sisters of Battle.

## 1. Native C# Inference Extensions (`mod/HKRLEnvMod/Action/SistersScaledRecurrent.cs`)

- **V1 (`0x3E4C5149`)**: 2-lag motion history (0.16s, 0.48s) + entity MLP max/sum pooling + Pre-LN ResFFN + residual GRU.
- **V2 (`0x3F4C5149`)**: 4-lag motion history (0.08s, 0.16s, 0.32s, 0.48s) + 6-token Multi-Head Entity Self-Attention (player + 3 bosses + 2 spinning discs) + Pre-LN ResFFN + residual GRU.
- Dispatched in `SistersOptionRouter.cs` as a zero-allocation `NativeSistersNetwork` running strictly in-process at 1x original game speed (`HKRL_DEVELOPMENT_SPEED=1`).
- Also adds `SistersTeacherActionHistory.cs` and `SistersTeacherHistorySmoke.cs` for audited teacher-command parity.

## 2. Two-Phase Policy & Training Pipeline (`scripts/wp22/`)

- **Phase-1 Opener (`train_wp22_opener.py`)**: Dedicated V2 Entity-Attention 4-lag opener (`opener-wp22-attn139.pt`) trained on 139 clean 99%-teacher victories (zero legacy 77% contamination). Currently frozen during online Trio PPO; P1 accounts for ~3%–9% of total damage events and is flagged for next-stage pure-0-hit reinforcement.
- **Trio Backbone & Offline Pretrain (`wp22_attn_model.py`, `train_wp22_scaled_drl.py`)**:
  - `g_attn_320_2lgru`: V2 6-token attention + 2-layer GRU (`d_model=320`).
  - `j_disc_320_1lgru_hi`: V2 disc-focused attention + 1-layer GRU (`d_model=320`).
  - `m_disc_384_2lgru_pure`: V1/V2 high-capacity disc model (`d_model=384`).
- **On-G4 Closed-Loop Online PPO (`train_wp22_gpu_ppo.py`, `colab_linux_online_ppo.py`)**:
  - Runs 32 parallel native Linux headless Unity instances (`-batchmode -nographics`, per-lane isolated `hollow_knight.x86_64` copy + symlinks to bypass `unity.lock`, `steam_appid.txt=367520`) on a single 48-core / 96GB RTX PRO 6000 Colab G4 VM.
  - Strictly verifies C# `SistersPpoSampler` RNG uniform & log-probability parity on CPU FP32 (`allow_tf32=False`, threshold `3e-4`) before GPU updates (`actor`, `critic`, and frozen `worlds` set to `.train()` for cuDNN GRU backward with `dropout=0.0`).
  - **V3 Continuous Evolution**: Decouples continuously advancing `latest.pt` from archived `champion.pt`, with collapse rollback (`wins < 27/30` or `trio_avg_dmg > best + 1.50`), Smooth-L1 critic warmup, decaying offline KL → running-champion anchor KL, first-hit cliff penalty, 1.3s post-hit iframe damage-reward zeroing, and 32-episode unseen `HOLDOUT_AUDIT` every 4 rounds.
  - **V4 True Radiant Early-Stop Curriculum (`observe-collect.py`, `colab_linux_multi_runner.py`, `remote_launch_v4_radiant.py`)**:
    - Protocol-level zero-mod-change Radiant early stopping via `HKRL_RADIANT_MAX_TRIO_HITS` (`2` hits in Stage 1/2 → `1` hit in Stage 3), using native `game.pause()` + `ppo_timeout_prefix()`.
    - Marks Radiant-stopped final valid frame as `terminated=True` in `native_gae` with terminal failure penalty (`-4.5`), preventing post-hit 9-mask win returns from masking early mistakes while cutting wasted post-hit rollout wall-time (~110s vs ~220s).
    - Caps online elite BC (`0.035 -> 0.025 -> 0.020`) and samples preferentially from the most recent 4 rounds of 0-hit wins to avoid locking the policy onto noisy early trajectories.

## 3. Empirical Milestones (as of 2026-11-25)

All evaluations are 1x original speed, 9 initial masks, 0 initial SOUL, frozen native neural actor (`teacher_fraction=0`):

| Track / Model | Baseline R0 (30-ep) | Online PPO Best (30-ep val) | Unseen 32-ep `HOLDOUT_AUDIT` |
|---|---|---|---|
| `g_attn_320_2lgru` | 30/30 wins, 0 no-hit, 3.267 avg dmg | R3: 30/30, 7/30 no-hit (23.3%), 1.733 avg dmg | R8: **32/32 wins, 7/32 no-hit (21.9%), 1.781 avg dmg** (`p1=1, trio=56`) |
| `j_disc_320_1lgru_hi` | 30/30 wins, 1 no-hit, 3.033 avg dmg | R8: 30/30, 3/30 no-hit (10.0%), 1.933 avg dmg | R8: 32/32 wins, 3/32 no-hit (9.4%), 2.500 avg dmg (`p1=1, trio=79`) |
| `m_disc_384_2lgru_pure` | 30/30 wins, 0 no-hit, 3.633 avg dmg | R4: 30/30, 2/30 no-hit (6.7%), 2.833 avg dmg | Superseded on `colab-d` by `g_attn_radiant` (V4 2-hit→1-hit Radiant) |

## 4. Open Review Points

1. **Frozen Phase-1 Opener**: P1 currently incurs ~1–4 hits per 30–32 fights, creating a structural ceiling below 95/100 full-fight no-hit unless reinforced/retrained on pure 0-hit P1 trajectories.
2. **Discount Horizon (`GAMMA=0.995`)**: Over ~1,250 Trio steps (50s at 25 Hz), terminal `no_hit_bonus` decays by `0.995^1250 ≈ 0.0019`; credit assignment relies heavily on `first_hit_pen`, pre-hit advantage weighting, and V4 Radiant early termination.
3. **Validation Seed Rotation**: Periodic 32-episode unseen `HOLDOUT_AUDIT` guards against overfitting, while per-round gating still evaluates fixed `11100001..11100030`.
4. **Formal 100-Episode Goal 1 Acceptance**: 94/94 + 32/32 win streaks are established; dedicated 100-episode formal acceptance (`>=98/100`) is scheduled on fresh seeds.
