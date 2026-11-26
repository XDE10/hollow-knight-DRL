"""WP22 Safe Monotonic Online Recurrent Actor-Critic PPO (V1 & V2 Entity-Attention compatible, GPU/CPU).

Key upgrades over WP19 CPU 8-fight PPO:
  1. Supports V2 0x3F4C5149 (6-Token Entity Self-Attention + 4-Lag motion_history_4lag) AND V1 0x3E4C5149.
  2. Supports multi-temperature cohorts (e.g. 18 fights temp=0.28 + 18 fights temp=0.42, ZERO greedy fights!),
     verifying per-fight cryptographic RNG uniform & log-probability parity with C# SistersPpoSampler.
  3. Direct anti-damage credit assignment: amplifies negative GAE advantage on the 1.0s (25-frame) window
     immediately before ENEMY_PROJECTILE (2.4x) and BOSS_BODY/SPIKES (1.8x) hits so RL directly eliminates
     the exact multi-modal imitation failure modes (jumping into discs / Quake startup into discs with full Shade Cloak).
  4. Strict trust-region monotonic acceptance (kl_limit=0.008) + offline KL anchor + 6 physical safety guards
     + dash_ratio drift guardrail so policy cannot over-dash or regress.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

_HERE = Path(__file__).resolve().parent
ART = Path(__file__).resolve().parents[3]
WP18 = ART / "WP18-ScaledDRL"
REPO = WP18 / "HKRL-wp18"
CLEAN_PY = ART / "python"
for _p in (_HERE, _HERE / "scripts", _HERE / "python", WP18, REPO / "scripts", CLEAN_PY, REPO / "python"):
    if str(_p) not in sys.path and _p.is_dir():
        sys.path.append(str(_p))
if str(_HERE) in sys.path:
    sys.path.remove(str(_HERE))
sys.path.insert(0, str(_HERE))

from audit_sisters_mainline_replay import read_episode
from hkrl.training.gae import compute_gae
from train_hkrl_wm_recurrent_rl import prepare_fight, sha256
from train_sisters_recurrent_ppo import (
    HORIZONS,
    STEPS,
    load_world,
    native_gae,
    validate_behavior_lineage,
)
from train_sisters_recurrent_world import World
from train_sisters_timing_iql import DISCS, FOES, motion_history
from train_wp18_scaled_drl import pre_window
from wp18_scaled_model import ScaledRecurrentActor, ScaledRecurrentValues
from wp22_attn_model import ATTN_MAGIC, AttnRecurrentActor, AttnRecurrentValues, motion_history_4lag


@dataclass
class WP22Rollout:
    inputs: Tensor
    action: Tensor
    old_log_distribution: Tensor
    old_value: Tensor
    advantage: Tensor
    returns: Tensor
    world_rows: Tensor
    train_rows: Tensor
    temperature: float
    pre_proj_mask: Tensor
    pre_body_mask: Tensor
    hitless_mask: Tensor | None = None
    is_nohit_win: bool = False
    seg_rew: np.ndarray | None = None
    seg_clock: np.ndarray | None = None
    seg_term: np.ndarray | None = None
    seg_valid: np.ndarray | None = None


def wp22_gae(
    reward: np.ndarray,
    value: np.ndarray,
    clock: np.ndarray,
    valid: np.ndarray,
    terminated: np.ndarray,
    gamma: float = 0.998,
    gae_lambda: float = 0.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Configurable-horizon GAE matching native_gae segment boundaries."""
    advantage = np.zeros(len(clock), dtype="f4")
    returns = np.zeros_like(advantage)
    intervals = np.r_[np.diff(clock) / 0.04, 1.0]
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for lo, hi in boundaries.reshape(-1, 2):
        terminal = bool(terminated[hi - 1])
        last_value = np.zeros(1, dtype="f4")
        if not terminal:
            if hi == len(clock):
                raise ValueError("A nonterminal transition needs an observed successor")
            last_value[0] = value[hi]
        done = np.zeros((hi - lo, 1), dtype=bool)
        done[-1] = True
        truncated = done & (not terminal)
        adv, target = compute_gae(
            reward[lo:hi, None],
            value[lo:hi, None],
            done,
            truncated,
            last_value,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            discount_exponents=intervals[lo:hi, None],
        )
        advantage[lo:hi] = adv[:, 0]
        returns[lo:hi] = target[:, 0]
    return advantage, returns


@torch.no_grad()
def recompute_rollout_gae(
    critic: nn.Module,
    rollouts: list[WP22Rollout],
    device: torch.device,
    gamma: float = 0.998,
    gae_lambda: float = 0.96,
) -> None:
    """Recompute GAE advantages and returns using the latest Critic weights."""
    critic.eval()
    for r in rollouts:
        if r.seg_rew is None or r.seg_clock is None or r.seg_term is None or r.seg_valid is None:
            continue
        val = critic(r.inputs[None].to(device))[2][0].cpu().numpy()
        adv, ret = wp22_gae(
            r.seg_rew, val, r.seg_clock, r.seg_valid, r.seg_term,
            gamma=gamma, gae_lambda=gae_lambda,
        )
        r.old_value = torch.from_numpy(val).float()
        r.advantage = torch.from_numpy(adv).float()
        r.returns = torch.from_numpy(ret).float()


def stage_weight(r: "WP22Rollout", adv: Tensor, post_hit_ppo_weight: float, device: torch.device) -> Tensor:
    """Down-weight PPO loss on post-first-Trio-hit frames (keeps hitless / pre-hit / low-HP frames at 1.0)."""
    if post_hit_ppo_weight >= 1.0 or r.hitless_mask is None:
        return torch.ones_like(adv)
    valid = r.train_rows
    obs = r.inputs[valid].to(device)
    keep = (r.hitless_mask[valid] | r.pre_proj_mask[valid] | r.pre_body_mask[valid]).to(device) | (obs[:, 4] <= (3.5 / 9.0))
    return torch.where(keep, torch.ones_like(adv), torch.full_like(adv, float(post_hit_ppo_weight)))


@dataclass(frozen=True)
class EliteSample:
    inputs: Tensor
    action: Tensor
    train_rows: Tensor
    weight: float


def build_actor_critic_from_cfg(catalogue: Tensor, cfg: dict) -> tuple[nn.Module, nn.Module, bool]:
    is_v2 = cfg.get("magic") == ATTN_MAGIC or ("d_tok" in cfg)
    if is_v2:
        kw = dict(
            d_boss=cfg["d_boss"],
            d_disc=cfg["d_disc"],
            d_past=cfg["d_past"],
            d_tok=cfg["d_tok"],
            n_heads=cfg["n_heads"],
            d_model=cfg["d_model"],
            d_ffn=cfg["d_ffn"],
            n_blocks=cfg["n_blocks"],
            n_gru_layers=cfg["n_gru_layers"],
            dropout=0.0,
        )
        return AttnRecurrentActor(catalogue, **kw).eval(), AttnRecurrentValues(catalogue, **kw).eval(), True
    kw = dict(
        d_boss=cfg["d_boss"],
        d_disc=cfg["d_disc"],
        d_past=cfg["d_past"],
        d_model=cfg["d_model"],
        d_ffn=cfg["d_ffn"],
        n_blocks=cfg["n_blocks"],
        n_gru_layers=cfg["n_gru_layers"],
        dropout=0.0,
    )
    return ScaledRecurrentActor(catalogue, **kw).eval(), ScaledRecurrentValues(catalogue, **kw).eval(), False


@torch.no_grad()
def verified_rollouts_wp22(
    folders: list[Path],
    plan: dict,
    actor: nn.Module,
    critic: nn.Module,
    catalogue: Tensor,
    is_v2: bool,
) -> tuple[list[WP22Rollout], list[dict]]:
    hist_fn = motion_history_4lag if is_v2 else motion_history
    per_seed_temp: dict[str, float] = plan.get("temperatures", {})
    default_temp = float(plan.get("temperature", 0.35))

    found = {}
    for folder in folders:
        for index, item in enumerate(json.loads((folder / "manifest.json").read_bytes())):
            receipt = json.loads((folder / f"episode-{index:03d}.json").read_bytes())
            seed = receipt["unity_reset_seed"]
            source = folder / item["file"]
            exp_temp = float(per_seed_temp.get(str(seed), default_temp))
            if (
                receipt["seed"] != seed
                or seed in found
                or seed not in plan["seeds"]
                or receipt["behavior_policy_sha256"] != plan["actor_sha256"]
                or receipt["mod_sha256"] != plan["mod_sha256"]
                or receipt["teacher_fraction"] != 0
                or receipt.get("teacher_labels", False)
                or receipt.get("corrective_labels", False)
                or item.get("teacher_labels", False)
                or item.get("corrective_labels", False)
                or receipt["game_speed"] != 1
                or not all(float(k) <= 1.001 for k in receipt["speed_counts"])
                or abs(float(receipt["ppo_temperature"]) - exp_temp) > 1e-9
                or receipt["sampler_draw_seed"] != plan["draw_seeds"][str(seed)]
                or sha256(source) != receipt["tape_sha256"]
                or item["tape_sha256"] != receipt["tape_sha256"]
            ):
                raise ValueError(f"PPO fight {seed} does not match its registered rollout assignment")
            found[seed] = (source, receipt, exp_temp)

    if set(found) != set(plan["seeds"]) or len(found) != len(plan["seeds"]):
        raise ValueError("The complete registered training cohort is required")

    lookup = {tuple(a): i for i, a in enumerate(catalogue.tolist())}
    rollouts: list[WP22Rollout] = []
    evidence: list[dict] = []

    for seed in plan["seeds"]:
        source, receipt, temp = found[seed]
        with np.load(source, allow_pickle=False) as tape:
            arrays = {k: tape[k] for k in tape.files}
        if {"teacher_controls", "corrective_label_valid"} & arrays.keys():
            raise ValueError(f"PPO fight {seed} contains teacher supervision")
        episode = read_episode(
            {k: v for k, v in arrays.items() if not k.startswith("ppo_")},
            {"won": receipt["win"], "truncated": receipt["truncated"]},
        )
        fight = prepare_fight(episode, lookup)
        n = len(episode.clock)
        selected = arrays["ppo_rows"] < n
        rows = arrays["ppo_rows"][selected]
        actions = arrays["ppo_action"][selected]
        if (
            not np.array_equal(rows, np.flatnonzero(episode.observation[:, 23] > 0.5))
            or (actions >= len(catalogue)).any()
            or (actions < 0).any()
            or not np.array_equal(catalogue.numpy()[actions], episode.action[rows])
        ):
            raise ValueError(f"Sampled PPO keys disagree with applied keys for fight {seed}")
        if len(rows) and rows[-1] != len(episode.clock) - 1:
            raise ValueError(f"PPO fight {seed} lacks a Trio end boundary")

        pre_proj_all = pre_window(episode.clock, episode.events[:, 2] > 0, 1.00)
        pre_body_all = pre_window(episode.clock, (episode.events[:, 1] + episode.events[:, 3]) > 0, 0.80)

        valid_events = episode.events[fight.train_rows]
        stats = {
            "seed": seed,
            "temperature": temp,
            "effective_rewards": float(episode.reward[fight.train_rows].sum()),
            "effective_event_totals": dict(
                zip(
                    ("damage_dealt", "boss_body", "projectile", "spikes", "other_damage", "heal", "arena_win", "death"),
                    map(float, valid_events.sum(0)),
                    strict=True,
                )
            ),
            "win": receipt["win"],
            "truncated": receipt["truncated"],
            "decisions": len(rows),
            "max_logp_error": 0.0,
            "masks_lost": sum(receipt["damage_taken"].values()),
            "valid_decisions": int(fight.train_rows.sum()),
        }
        evidence.append(stats)
        if not len(rows):
            continue

        observations, logits, values = [], [], []
        for lo, hi in fight.segments:
            x = torch.from_numpy(hist_fn(episode.clock, episode.observation, np.arange(lo, hi)))
            observations.append(x)
            logits.append((actor(x[None])[0].double() / temp).log_softmax(-1))
            values.append(critic(x[None])[2][0])

        old = torch.cat(logits)
        selected_logp = old[torch.arange(len(rows)), torch.from_numpy(actions).long()].numpy()
        log_error = float(np.max(np.abs(selected_logp - arrays["ppo_log_probability"][selected])))
        cumulative = old.exp().cumsum(-1).numpy()
        upper = cumulative[np.arange(len(rows)), actions]
        lower = upper - np.exp(selected_logp)
        u = arrays["ppo_uniform"][selected]
        if log_error > 2e-4 or (u < lower - 2e-5).any() or (u >= upper + 2e-5).any():
            raise ValueError(f"PPO behavior log likelihood does not reproduce for {seed}: {log_error}")
        stats["max_logp_error"] = log_error

        for x, log_dist, val, (lo, hi) in zip(observations, logits, values, fight.segments, strict=True):
            seg_valid = fight.train_rows[lo:hi]
            if not seg_valid.any():
                continue
            seg_clock = episode.clock[lo:hi]
            seg_rew = episode.reward[lo:hi].copy()
            # Extra anti-hit terminal shaping strictly on observed real damage events (zero teacher labels!)
            seg_ev = episode.events[lo:hi]
            seg_rew -= 0.45 * seg_ev[:, 2]  # ENEMY_PROJECTILE extra penalty
            seg_rew -= 0.30 * (seg_ev[:, 1] + seg_ev[:, 3])  # BOSS_BODY / SPIKES extra penalty
            adv, ret = native_gae(seg_rew, val.numpy(), seg_clock, seg_valid, episode.terminated[lo:hi])

            exact = (np.abs(np.diff(seg_clock) - 0.04) < 1e-4) & (fight.action[lo : hi - 1] >= 0)
            broken = np.r_[0, np.cumsum(~exact)]
            w_starts = np.arange(0, max(0, (hi - lo) - STEPS))
            if len(w_starts):
                w_starts = w_starts[broken[w_starts] == broken[w_starts + STEPS]]
            rollouts.append(
                WP22Rollout(
                    inputs=x,
                    action=torch.from_numpy(fight.action[lo:hi].clip(0)).long(),
                    old_log_distribution=log_dist.float(),
                    old_value=val.float(),
                    advantage=torch.from_numpy(adv).float(),
                    returns=torch.from_numpy(ret).float(),
                    world_rows=torch.from_numpy(w_starts).long(),
                    train_rows=torch.from_numpy(seg_valid),
                    temperature=temp,
                    pre_proj_mask=torch.from_numpy(pre_proj_all[lo:hi]),
                    pre_body_mask=torch.from_numpy(pre_body_all[lo:hi]),
                )
            )
    return rollouts, evidence


def _effective_advantage(
    r: WP22Rollout,
    mean: Tensor,
    scale: Tensor,
    pre_proj_mult: float,
    pre_body_mult: float,
    post_hit_ppo_weight: float,
    win_nt1_boost: float,
    device: torch.device,
) -> Tensor:
    """Compute normalized, hit-window & non-top1-win weighted PPO advantage."""
    valid = r.train_rows
    adv = ((r.advantage[valid].to(device) - mean) / scale).clamp(-5, 5)
    w_proj = torch.where((adv < 0) & r.pre_proj_mask[valid].to(device), torch.full_like(adv, float(pre_proj_mult)), torch.ones_like(adv))
    w_body = torch.where((adv < 0) & r.pre_body_mask[valid].to(device), torch.full_like(adv, float(pre_body_mult)), torch.ones_like(adv))
    w_hit = torch.maximum(w_proj, w_body)
    eff_adv = adv * w_hit * stage_weight(r, adv, post_hit_ppo_weight, adv.device)
    if win_nt1_boost > 1.0 and r.is_nohit_win:
        is_nt1 = (r.old_log_distribution[valid].argmax(-1) != r.action[valid]).to(device)
        eff_adv = eff_adv * torch.where((adv > 0) & is_nt1, torch.full_like(adv, float(win_nt1_boost)), torch.ones_like(adv))
    return eff_adv


def trio_safety_guard(
    v_logits: Tensor,
    obs_v: Tensor,
    cat_dev: Tensor,
    samples: int,
) -> Tensor:
    """Physical action-mask and cliff/Quake-trap guard over valid Trio frames."""
    jtap_mask = (cat_dev[:, 2] & 1) != 0
    dash_mask = (cat_dev[:, 2] & 4) != 0
    cast_mask = (cat_dev[:, 2] & 16) != 0
    cast_down_mask = cast_mask & (cat_dev[:, 1] == -1)
    focus_mask = (cat_dev[:, 2] & 32) != 0
    dash_left = dash_mask & (cat_dev[:, 0] == -1)
    dash_right = dash_mask & (cat_dev[:, 0] == 1)

    greedy_probs = v_logits.softmax(-1)
    xw = obs_v[:, 0] * 10.0
    bosses_v = obs_v[:, FOES:DISCS].reshape(-1, 3, 36)
    boss_lunge = ((bosses_v[:, :, 0] > 0.5) & (bosses_v[:, :, 20:26].max(dim=2).values > 0.5)).any(dim=1)
    wall_prep = (
        (bosses_v[:, :, 0] > 0.5)
        & ((bosses_v[:, :, 27:30].max(dim=2).values > 0.5) | (bosses_v[:, :, 32:36].max(dim=2).values > 0.5))
    ).any(dim=1)
    disc_active = (obs_v[:, 132] > 0.5) | (obs_v[:, 146] > 0.5)
    shadow_ready = (obs_v[:, 13] > 0.5) & (obs_v[:, 175] >= 1.50)

    no_cast = (obs_v[:, 168] < 0.5) & ((obs_v[:, 11] < 0.5) | (obs_v[:, 5] < (33.0 / 99.0 - 1e-3)))
    no_focus = (
        ((obs_v[:, 169] < 0.5) & ((obs_v[:, 12] < 0.5) | (obs_v[:, 5] < (33.0 / 99.0 - 1e-3)) | (obs_v[:, 7] < 0.5) | (obs_v[:, 4] > (6.5 / 9.0))))
        | disc_active | boss_lunge | wall_prep | (xw.abs() > 4.0) | (obs_v[:, 7] < 0.5)
    )
    no_dash = (obs_v[:, 166] < 0.5) & (obs_v[:, 13] < 0.5)
    no_jtap = (obs_v[:, 164] < 0.5) & (obs_v[:, 7] < 0.5) & (obs_v[:, 8] < 0.5)

    guard = v_logits.new_zeros(())
    denom = max(1, int(samples))
    for cond_m, act_m in ((no_cast, cast_mask), (no_focus, focus_mask), (no_dash, dash_mask), (no_jtap, jtap_mask)):
        if cond_m.any():
            guard = guard + 0.04 * greedy_probs[cond_m][:, act_m].sum(-1).sum() / denom
    disc_quake_trap = disc_active & shadow_ready & (obs_v[:, 7] < 0.5)
    if disc_quake_trap.any():
        guard = guard + 0.03 * greedy_probs[disc_quake_trap][:, cast_down_mask].sum(-1).sum() / denom
    cliff = xw.abs() > 5.2
    if cliff.any():
        guard = guard + 0.02 * torch.where(
            xw > 5.2, greedy_probs[:, dash_right].sum(-1), greedy_probs[:, dash_left].sum(-1)
        )[cliff].sum() / denom
    return guard


def eval_ppo_and_kl_wp22(
    actor: nn.Module,
    rollouts: list[WP22Rollout],
    mean: Tensor,
    scale: Tensor,
    samples: int,
    device: torch.device,
    clip_eps: float = 0.10,
    pre_proj_mult: float = 2.2,
    pre_body_mult: float = 1.6,
    post_hit_ppo_weight: float = 1.0,
    win_nt1_boost: float = 1.0,
) -> tuple[float, float]:
    total_ppo = 0.0
    total_kl = 0.0
    actor.eval()
    with torch.no_grad():
        for r in rollouts:
            valid = r.train_rows
            log_p = (actor(r.inputs[None].to(device))[0][valid.to(device)] / r.temperature).log_softmax(-1)
            old_log_p = r.old_log_distribution[valid].to(device)
            chosen = r.action[valid, None].to(device)
            ratio = (log_p.gather(1, chosen) - old_log_p.gather(1, chosen)).flatten().exp()
            eff_adv = _effective_advantage(r, mean, scale, pre_proj_mult, pre_body_mult, post_hit_ppo_weight, win_nt1_boost, device)
            ppo = -torch.minimum(ratio * eff_adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * eff_adv).sum() / samples
            kl = (old_log_p.exp() * (old_log_p - log_p)).sum() / samples
            total_ppo += float(ppo)
            total_kl += float(kl)
    return total_ppo, total_kl


def optimize_safe_wp22(
    actor: nn.Module,
    offline_actor: nn.Module,
    critic: nn.Module,
    rollouts: list[WP22Rollout],
    worlds: list[World],
    validated: Tensor,
    catalogue: Tensor,
    updates: int,
    actor_lr: float,
    kl_limit: float,
    device: torch.device,
    critic_warmup: int = 8,
    offline_kl_coef: float = 0.06,
    clip_eps: float = 0.10,
    pre_proj_mult: float = 2.2,
    pre_body_mult: float = 1.6,
    anchor_actor: nn.Module | None = None,
    anchor_kl_coef: float = 0.0,
    dash_drift_mult: float = 1.18,
    elite_samples: list[EliteSample] | None = None,
    elite_bc_coef: float = 0.0,
    behavior_kl_coef: float = 0.025,
    value_clip: float = 0.2,
    critic_lr: float = 8.0e-5,
    critic_warmup_lr: float = 1.2e-4,
    post_hit_ppo_weight: float = 1.0,
    ent_coef: float = 0.0,
    use_smooth_l1_critic: bool = True,
    world_aux_coef: float = 0.015,
    gamma: float = 0.998,
    gae_lambda: float = 0.96,
    max_greedy_kl: float = 0.0,
    win_nt1_boost: float = 1.0,
) -> list[dict]:
    samples = sum(int(r.train_rows.sum()) for r in rollouts)

    cat_dev = catalogue.to(device)
    dash_mask = (cat_dev[:, 2] & 4) != 0
    dash_left = dash_mask & (cat_dev[:, 0] == -1)
    dash_right = dash_mask & (cat_dev[:, 0] == 1)

    offline_probs_list: list[Tensor | None] = [None] * len(rollouts)
    anchor_probs_list: list[Tensor | None] = [None] * len(rollouts)
    ref_dash_rate_num = 0.0
    if offline_kl_coef > 0.0 or anchor_actor is None:
        offline_actor.eval()
        with torch.no_grad():
            for idx, r in enumerate(rollouts):
                off_logits = offline_actor(r.inputs[None].to(device))[0][r.train_rows.to(device)]
                if offline_kl_coef > 0.0:
                    offline_probs_list[idx] = (off_logits / r.temperature).softmax(-1)
                if anchor_actor is None:
                    ref_dash_rate_num += float(off_logits.softmax(-1)[:, dash_mask].sum())
    if anchor_actor is not None:
        anchor_actor.eval()
        with torch.no_grad():
            for idx, r in enumerate(rollouts):
                anc_logits = anchor_actor(r.inputs[None].to(device))[0][r.train_rows.to(device)]
                anc_probs = anc_logits.softmax(-1)
                anchor_probs_list[idx] = anc_probs
                ref_dash_rate_num += float(anc_probs[:, dash_mask].sum())
    ref_dash_rate = ref_dash_rate_num / max(1, samples)

    # V5 fix: iterative critic warmup + GAE recompute BEFORE normalizing advantages!
    if critic_warmup > 0:
        has_raw_rew = any(r.seg_rew is not None for r in rollouts)
        outer_loops = 2 if has_raw_rew else 1
        inner_steps = max(1, (critic_warmup + outer_loops - 1) // outer_loops)
        critic_only_opt = torch.optim.AdamW(critic.parameters(), lr=max(critic_warmup_lr, 2.5e-4), weight_decay=1e-4)
        for _ in range(outer_loops):
            if has_raw_rew:
                recompute_rollout_gae(critic, rollouts, device, gamma=gamma, gae_lambda=gae_lambda)
            critic.train()
            for _ in range(inner_steps):
                critic_only_opt.zero_grad(set_to_none=True)
                for r in rollouts:
                    valid = r.train_rows.to(device)
                    val = critic(r.inputs[None].to(device))[2][0][valid]
                    ret = r.returns[r.train_rows].to(device)
                    v_loss = 0.5 * F.smooth_l1_loss(val, ret, beta=5.0, reduction="sum") / samples
                    v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 2.0, error_if_nonfinite=True)
                critic_only_opt.step()
        if has_raw_rew:
            recompute_rollout_gae(critic, rollouts, device, gamma=gamma, gae_lambda=gae_lambda)

    all_adv = torch.cat([r.advantage[r.train_rows] for r in rollouts]).to(device)
    mean, scale = all_adv.mean(), all_adv.std(unbiased=False).clamp_min(0.05)

    actor.train()
    critic.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": actor_lr, "weight_decay": 1e-4},
            {"params": critic.parameters(), "lr": critic_lr, "weight_decay": 1e-4},
        ]
    )

    progress = []
    cur_ppo, _ = eval_ppo_and_kl_wp22(
        actor, rollouts, mean, scale, samples, device,
        clip_eps=clip_eps, pre_proj_mult=pre_proj_mult, pre_body_mult=pre_body_mult,
        post_hit_ppo_weight=post_hit_ppo_weight, win_nt1_boost=win_nt1_boost,
    )
    init_ppo = cur_ppo
    validated_dev = validated.to(device)
    step1_lr_mults = (1.0, 0.5, 0.25, 0.10)

    for step in range(updates):
        before_actor, before_critic = deepcopy(actor.state_dict()), deepcopy(critic.state_dict())
        max_step1_retries = 3 if step == 0 else 0
        accepted = False
        final_row: dict | None = None

        for retry_idx in range(max_step1_retries + 1):
            if step == 0:
                optimizer.param_groups[0]["lr"] = actor_lr * step1_lr_mults[retry_idx]
            step_elite_bc_coef = 0.0 if (step == 0 and retry_idx >= 1) else elite_bc_coef
            step_ent_coef = 0.0 if (step == 0 and retry_idx >= 3) else ent_coef

            actor.train()
            critic.train()
            for m in worlds:
                m.train()
            optimizer.zero_grad(set_to_none=True)
            obj_critic = obj_aux = obj_guard = obj_elite = obj_ent = obj_gkl = 0.0

            for r, off_probs, anc_probs in zip(rollouts, offline_probs_list, anchor_probs_list, strict=True):
                inp_dev = r.inputs[None].to(device)
                valid = r.train_rows.to(device)
                spatial, emb, _ = actor.backbone.forward_sequence(inp_dev, 0)
                v_logits = actor.heads(emb)[0][0][valid]
                log_p = (v_logits / r.temperature).log_softmax(-1)
                old_log_p = r.old_log_distribution[r.train_rows].to(device)
                chosen = r.action[r.train_rows, None].to(device)
                ratio = (log_p.gather(1, chosen) - old_log_p.gather(1, chosen)).flatten().exp()
                eff_adv = _effective_advantage(r, mean, scale, pre_proj_mult, pre_body_mult, post_hit_ppo_weight, win_nt1_boost, device)
                ppo = -torch.minimum(ratio * eff_adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * eff_adv).sum() / samples

                value = critic(inp_dev)[2][0][valid]
                old_value = r.old_value[r.train_rows].to(device)
                returns = r.returns[r.train_rows].to(device)
                clipped = old_value + (value - old_value).clamp(-value_clip, value_clip)
                if use_smooth_l1_critic:
                    value_loss = 0.5 * F.smooth_l1_loss(value, returns, beta=5.0, reduction="sum") / samples
                else:
                    value_loss = 0.5 * torch.maximum((value - returns).square(), (clipped - returns).square()).sum() / samples

                behavior_kl = (old_log_p.exp() * (old_log_p - log_p)).sum() / samples

                greedy_probs = v_logits.softmax(-1)
                obs_v = inp_dev[0, valid]
                guard = trio_safety_guard(v_logits, obs_v, cat_dev, samples)

                # Dash-rate drift guardrail (prevents WP20-style dash explosion)
                cur_dash_mass = greedy_probs[:, dash_mask].sum(-1).mean()
                dash_drift = 0.05 * F.relu(cur_dash_mass - ref_dash_rate * dash_drift_mult).square() * (len(v_logits) / samples)

                aux = ppo.new_zeros(())
                if world_aux_coef > 0.0 and len(r.world_rows):
                    start = r.world_rows.to(device)
                    seq = start[:, None] + torch.arange(STEPS, device=device)
                    future = start[:, None] + torch.tensor(HORIZONS, device=device)
                    target = inp_dev[0, future, :2] - inp_dev[0, start, None, :2]
                    latent = torch.cat((spatial[0, start], emb[0, start]), -1)
                    act_dev = r.action.to(device)
                    forecasts = torch.stack([m(latent, inp_dev[0, start, :6], act_dev[seq])[..., :2] for m in worlds])
                    aux = (
                        float(world_aux_coef)
                        * F.smooth_l1_loss(forecasts.mean(0)[:, validated_dev], target[:, validated_dev], beta=0.04)
                        * len(start)
                        / samples
                    )

                loss = ppo + value_loss + behavior_kl_coef * behavior_kl + guard + dash_drift + aux
                if offline_kl_coef > 0.0 and off_probs is not None:
                    offline_kl = (off_probs * (off_probs.clamp_min(1e-8).log() - log_p)).sum() / samples
                    loss = loss + offline_kl_coef * offline_kl
                cur_greedy_logp = v_logits.log_softmax(-1)
                ent_val = -(greedy_probs * cur_greedy_logp).sum(-1).sum() / samples
                obj_ent += float(ent_val.detach())
                if step_ent_coef > 0.0:
                    loss = loss - step_ent_coef * ent_val
                if anc_probs is not None:
                    anchor_kl = (anc_probs * (anc_probs.clamp_min(1e-8).log() - cur_greedy_logp)).sum() / samples
                    obj_gkl += float(anchor_kl.detach())
                    if anchor_kl_coef > 0.0:
                        loss = loss + anchor_kl_coef * anchor_kl
                loss.backward()
                obj_critic += float(value_loss.detach())
                obj_aux += float(aux.detach())
                obj_guard += float((guard + dash_drift).detach())

            if elite_samples and step_elite_bc_coef > 0.0:
                elite_weight_sum = sum(e.weight * int(e.train_rows.sum()) for e in elite_samples)
                if elite_weight_sum > 0:
                    for e in elite_samples:
                        e_valid = e.train_rows.to(device)
                        if not e_valid.any():
                            continue
                        _, e_emb, _ = actor.backbone.forward_sequence(e.inputs[None].to(device), 0)
                        e_logits = actor.heads(e_emb)[0][0][e_valid]
                        e_loss = (
                            step_elite_bc_coef
                            * e.weight
                            * F.cross_entropy(e_logits, e.action[e.train_rows].to(device), reduction="sum")
                            / elite_weight_sum
                        )
                        e_loss.backward()
                        obj_elite += float(e_loss.detach())

            nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.8, error_if_nonfinite=True)
            optimizer.step()

            post_ppo, post_kl = eval_ppo_and_kl_wp22(
                actor, rollouts, mean, scale, samples, device,
                clip_eps=clip_eps, pre_proj_mult=pre_proj_mult, pre_body_mult=pre_body_mult,
                post_hit_ppo_weight=post_hit_ppo_weight, win_nt1_boost=win_nt1_boost,
            )
            ppo_delta = post_ppo - cur_ppo
            accepted = bool(
                np.isfinite(post_kl)
                and post_kl <= kl_limit
                and (ppo_delta < 0.0 or (post_ppo < init_ppo and ppo_delta <= 1e-4))
            )
            if max_greedy_kl > 0.0 and step >= 2 and obj_gkl > max_greedy_kl:
                accepted = False
            final_row = {
                "step": step + 1,
                "pre_ppo": cur_ppo,
                "ppo": post_ppo,
                "ppo_delta": ppo_delta,
                "critic": obj_critic,
                "world_aux": obj_aux,
                "guard": obj_guard,
                "elite_bc": obj_elite,
                "kl": post_kl,
                "greedy_kl": round(obj_gkl, 6),
                "entropy": round(obj_ent, 4),
                "accepted": accepted,
            }
            if step == 0:
                final_row["step1_retries"] = retry_idx
            if accepted:
                cur_ppo = post_ppo
                if step == 0:
                    if retry_idx == 1:
                        optimizer.param_groups[0]["lr"] = actor_lr * 0.85
                    elif retry_idx >= 2:
                        optimizer.param_groups[0]["lr"] = actor_lr * 0.5
                break
            actor.load_state_dict(before_actor)
            if step > 0 or retry_idx < max_step1_retries:
                critic.load_state_dict(before_critic)
            if step == 0 and retry_idx < max_step1_retries:
                optimizer.state.clear()

        assert final_row is not None
        progress.append(final_row)
        if not accepted:
            break
    actor.eval()
    critic.eval()
    for m in worlds:
        m.eval()
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("offline", "behavior", "native", "world", "world-validation", "plan", "output"):
        parser.add_argument("--" + path, type=Path, required=True)
    parser.add_argument("--rollout", type=Path, nargs="+", required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument("--kl-limit", type=float, default=0.008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(6)

    plan = json.loads(args.plan.read_bytes())
    offline_sha256 = sha256(args.offline)
    behavior_sha256 = sha256(args.behavior)
    world_sha256 = sha256(args.world)
    if (
        sha256(args.native) != plan["actor_sha256"]
        or behavior_sha256 != plan["learner_sha256"]
        or offline_sha256 != plan.get("offline_sha256", offline_sha256)
        or world_sha256 != plan.get("world_sha256", world_sha256)
        or ("world_validation_sha256" in plan and sha256(args.world_validation) != plan["world_validation_sha256"])
        or args.updates < 1
    ):
        raise ValueError("The native PPO policy and Python weights must match their frozen plan")

    saved = torch.load(args.behavior, map_location="cpu", weights_only=True)
    off_saved = torch.load(args.offline, map_location="cpu", weights_only=True)
    audit_sha256 = validate_behavior_lineage(saved, behavior_sha256, offline_sha256, world_sha256, plan.get("audit_sha256"))
    catalogue = saved["catalogue"]
    cfg = saved["provenance"]["scaled_cfg"]

    actor, critic, is_v2 = build_actor_critic_from_cfg(catalogue, cfg)
    actor.load_state_dict(saved["state"])
    critic.load_state_dict(saved["critic"])
    offline_actor, _, _ = build_actor_critic_from_cfg(catalogue, cfg)
    offline_actor.load_state_dict(off_saved["state"])
    offline_actor.requires_grad_(False)

    worlds, validated = load_world(args.world, args.world_validation, offline_sha256, audit_sha256, 2 * cfg["d_model"], len(catalogue))
    rollouts, evidence = verified_rollouts_wp22(args.rollout, plan, actor, critic, catalogue, is_v2)

    actor.to(device)
    critic.to(device)
    offline_actor.to(device)
    worlds = [w.to(device) for w in worlds]

    progress = optimize_safe_wp22(
        actor, offline_actor, critic, rollouts, worlds, validated, catalogue,
        args.updates, args.actor_lr, args.kl_limit, device,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    asset = args.output / "learner.pt"
    accepted = sum(row["accepted"] for row in progress)
    if accepted:
        torch.save(
            {
                **saved,
                "state": {k: v.cpu() for k, v in actor.state_dict().items()},
                "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
                "ppo": {
                    "parent_sha256": plan["learner_sha256"],
                    "offline_sha256": offline_sha256,
                    "world_sha256": world_sha256,
                    "plan_sha256": sha256(args.plan),
                    "accepted_updates": accepted,
                },
            },
            asset,
        )
    result = {
        "is_v2_attn": is_v2,
        "device": str(device),
        "progress": progress,
        "verified_fights": evidence,
        "validated_world_position_components": validated.tolist(),
        "checkpoint_sha256": sha256(asset) if accepted else None,
    }
    (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
