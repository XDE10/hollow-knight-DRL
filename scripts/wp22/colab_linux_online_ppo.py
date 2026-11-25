"""Colab G4 Closed-Loop 16-Lane Online Safe Recurrent No-Hit PPO Trainer.

Runs entirely on a single Colab G4 (96 GB RTX PRO 6000 + multi-core CPU + 170+ GB RAM):
  For each round r = 1..N:
    1. 16-way parallel native Linux Hollow Knight sampled PPO rollout collection
       (two non-greedy temperatures, e.g. 32 fights @ 0.22 + 32 fights @ 0.35 = 64 fights in ~9 min!).
    2. Exact C# RNG uniform & log-probability verification on every single fight.
    3. On-G4 GPU Safe Recurrent Actor-Critic PPO update (~20s on RTX PRO 6000!) with:
       - Direct No-Hit & Anti-Damage Reward Shaping (-1.00 proj, -0.75 body/spikes, +1.50 no-hit win)
       - 2.6x pre-proj (1.0s) & 2.0x pre-body (0.8s) negative GAE advantage amplification
       - Strict monotonic trust-region (kl_limit=0.008) + offline KL anchor + dash_rate drift guard
         + all 6 physical safety guards + anti-Quake-into-disc trap guard!
    4. Export V2 native .bin (with verified V2 139-fight P1 Opener) -> 16-lane 30-episode greedy validation gate (~4 min!).
    5. Promote ONLY if (win_rate improves OR win_rate ties with lower avg_damage / higher no_hit_wins),
       and immediately push every promoted champion to Google Drive (mydrive:HKRL-WP22/cloud-rl/)!
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, Path("/content/hkrl-repo"), Path("/content/hkrl-repo/scripts"), Path("/content/hkrl-repo/python")):
    if str(_p) not in sys.path and _p.is_dir():
        sys.path.insert(0, str(_p))

from colab_linux_multi_runner import compute_no_hit_win, run_parallel_cohort, sha256
from train_sisters_recurrent_ppo import load_world, native_gae, read_native_draws
from train_wp22_gpu_ppo import (
    EliteSample,
    WP22Rollout,
    build_actor_critic_from_cfg,
    eval_ppo_and_kl_wp22,
    optimize_safe_wp22,
)
from wp18_scaled_model import ScaledRecurrentActor, export_two_phase_scaled
from wp22_attn_model import ATTN_MAGIC, AttnRecurrentActor, export_two_phase_attn, motion_history_4lag
from train_sisters_timing_iql import motion_history
from train_wp18_scaled_drl import pre_window
from audit_sisters_mainline_replay import read_episode
from train_hkrl_wm_recurrent_rl import prepare_fight


def export_native_from_pt(
    pt_path: Path,
    out_bin: Path,
    parent_bin: Path,
    opener_pt: Path | None = None,
) -> dict[str, Any]:
    sv = torch.load(pt_path, map_location="cpu", weights_only=True)
    cfg = sv["provenance"]["scaled_cfg"]
    cat = sv["catalogue"]
    is_v2 = cfg.get("magic") == ATTN_MAGIC or ("d_tok" in cfg)
    opening_actor = None
    if opener_pt is not None and opener_pt.is_file():
        op_sv = torch.load(opener_pt, map_location="cpu", weights_only=True)
        ocfg = op_sv["info"]["cfg"]
        if op_sv["info"].get("v2_attn") or ocfg.get("magic") == ATTN_MAGIC or ("d_tok" in ocfg):
            opening_actor = AttnRecurrentActor(
                op_sv["catalogue"],
                d_boss=ocfg["d_boss"], d_disc=ocfg["d_disc"], d_past=ocfg["d_past"],
                d_tok=ocfg["d_tok"], n_heads=ocfg["n_heads"],
                d_model=ocfg["d_model"], d_ffn=ocfg["d_ffn"],
                n_blocks=ocfg["n_blocks"], n_gru_layers=ocfg["n_gru_layers"], dropout=0.0,
            ).eval()
        else:
            opening_actor = ScaledRecurrentActor(
                op_sv["catalogue"],
                d_boss=ocfg["d_boss"], d_disc=ocfg["d_disc"], d_past=ocfg["d_past"],
                d_model=ocfg["d_model"], d_ffn=ocfg["d_ffn"],
                n_blocks=ocfg["n_blocks"], n_gru_layers=ocfg["n_gru_layers"], dropout=0.0,
            ).eval()
        opening_actor.load_state_dict(op_sv["state"])

    if is_v2:
        actor = AttnRecurrentActor(
            cat,
            d_boss=cfg["d_boss"], d_disc=cfg["d_disc"], d_past=cfg["d_past"],
            d_tok=cfg["d_tok"], n_heads=cfg["n_heads"],
            d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
            n_blocks=cfg["n_blocks"], n_gru_layers=cfg["n_gru_layers"], dropout=0.0,
        ).eval()
        actor.load_state_dict(sv["state"])
        if opening_actor is not None:
            return export_two_phase_attn(actor, cat.numpy(), out_bin, opening_actor=opening_actor)
        return export_two_phase_attn(actor, cat.numpy(), out_bin, opening_parent_bin=parent_bin)
    actor = ScaledRecurrentActor(
        cat,
        d_boss=cfg["d_boss"], d_disc=cfg["d_disc"], d_past=cfg["d_past"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        n_blocks=cfg["n_blocks"], n_gru_layers=cfg["n_gru_layers"], dropout=0.0,
    ).eval()
    actor.load_state_dict(sv["state"])
    if opening_actor is not None:
        return export_two_phase_scaled(actor, cat.numpy(), out_bin, opening_actor=opening_actor)
    return export_two_phase_scaled(actor, cat.numpy(), out_bin, opening_parent_bin=parent_bin)


@torch.no_grad()
def build_verified_cloud_rollouts(
    results: list[dict[str, Any]],
    run_dir: Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    catalogue: torch.Tensor,
    is_v2: bool,
    temperatures: dict[int, float],
    draw_seeds: dict[int, int],
    actor_sha: str,
    mod_sha: str,
    no_hit_bonus: float = 2.50,
    proj_pen: float = 1.25,
    body_pen: float = 0.95,
    first_hit_pen: float = 3.00,
    iframe_zero_sec: float = 1.30,
    hitless_dps_bonus: float = 0.006,
    one_hit_bonus: float = 1.00,
    post_hit_reward_scale: float = 1.0,
    radiant_max_trio_hits: int = 0,
    radiant_fail_pen: float = 4.0,
) -> tuple[list[WP22Rollout], list[dict[str, Any]]]:
    hist_fn = motion_history_4lag if is_v2 else motion_history
    lookup = {tuple(a): i for i, a in enumerate(catalogue.tolist())}
    rollouts: list[WP22Rollout] = []
    evidence: list[dict[str, Any]] = []
    actor = actor.cpu().eval()
    critic = critic.cpu().eval()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    try:
        for r in results:
            seed = r["seed"]
            temp = temperatures[seed]
            dseed = draw_seeds[seed]
            folder = run_dir / r["folder"]
            receipt = json.loads((folder / "episode-000.json").read_bytes())
            item = json.loads((folder / "manifest.json").read_bytes())[0]
            source = folder / item["file"]
            if (
                receipt["seed"] != seed
                or receipt["unity_reset_seed"] != seed
                or receipt["behavior_policy_sha256"] != actor_sha
                or receipt["mod_sha256"] != mod_sha
                or receipt["teacher_fraction"] != 0
                or receipt["game_speed"] != 1
                or abs(float(receipt["ppo_temperature"]) - temp) > 1e-9
                or receipt["sampler_draw_seed"] != dseed
            ):
                raise ValueError(f"Cloud PPO rollout provenance mismatch on seed {seed}")

            with np.load(source, allow_pickle=False) as tape:
                arrays = {k: tape[k] for k in tape.files}
            episode = read_episode(
                {k: v for k, v in arrays.items() if not k.startswith("ppo_")},
                {"won": receipt["win"], "truncated": receipt["truncated"]},
            )
            fight = prepare_fight(episode, lookup)
            n = len(episode.clock)
            selected = arrays["ppo_rows"] < n
            rows = arrays["ppo_rows"][selected]
            actions = arrays["ppo_action"][selected]
            if not np.array_equal(rows, np.flatnonzero(episode.observation[:, 23] > 0.5)):
                raise ValueError(f"Sampled rows mismatch on seed {seed}")

            pre_proj_all = pre_window(episode.clock, episode.events[:, 2] > 0, 1.00)
            pre_body_all = pre_window(episode.clock, (episode.events[:, 1] + episode.events[:, 3]) > 0, 0.80)

            trio_phase = episode.observation[:, 23] > 0.5
            dmg_ev_all = episode.events[:, 1:5].sum(axis=1)
            trio_dmg_ev = dmg_ev_all * trio_phase.astype("f4")
            cum_hits_all = np.cumsum(trio_dmg_ev)
            prior_hits_all = cum_hits_all - trio_dmg_ev
            first_hit_all = (trio_dmg_ev > 0) & (prior_hits_all == 0)
            hitless_all = (cum_hits_all == 0)
            hit_times = np.where(trio_dmg_ev > 0, episode.clock, -1e9)
            in_iframe_all = (episode.clock - np.maximum.accumulate(hit_times)) <= iframe_zero_sec
            trio_masks_lost = int(round(float(trio_dmg_ev.sum())))

            is_nohit = bool(receipt["win"]) and trio_masks_lost == 0
            masks_lost = sum(int(v) for v in receipt["damage_taken"].values())
            stats = {
                "seed": seed,
                "temperature": temp,
                "win": bool(receipt["win"]),
                "no_hit_win": is_nohit,
                "trio_masks_lost": trio_masks_lost,
                "trio_no_hit_win": is_nohit,
                "masks_lost": masks_lost,
                "decisions": len(rows),
                "valid_decisions": int(fight.train_rows.sum()),
            }
            evidence.append(stats)
            if not len(rows):
                continue

            observations, logits, values = [], [], []
            for lo, hi in fight.segments:
                x = torch.from_numpy(hist_fn(episode.clock, episode.observation, np.arange(lo, hi)))
                observations.append(x)
                x_cpu = x[None]
                logits.append((actor(x_cpu)[0].double() / temp).log_softmax(-1))
                values.append(critic(x_cpu)[2][0])

            old = torch.cat(logits)
            selected_logp = old[torch.arange(len(rows)), torch.from_numpy(actions).long()].numpy()
            log_error = float(np.max(np.abs(selected_logp - arrays["ppo_log_probability"][selected])))
            if log_error > 3e-4:
                raise ValueError(f"Cloud PPO logp parity error on {seed}: {log_error}")
            stats["max_logp_error"] = round(log_error, 6)

            for seg_idx, (x, log_dist, val, (lo, hi)) in enumerate(zip(observations, logits, values, fight.segments, strict=True)):
                seg_valid = fight.train_rows[lo:hi]
                if not seg_valid.any():
                    continue
                seg_clock = episode.clock[lo:hi]
                seg_rew = episode.reward[lo:hi].copy()
                seg_ev = episode.events[lo:hi]
                # Direct No-Hit & Anti-Damage reward shaping (100% real-game events, zero teacher!)
                seg_rew -= proj_pen * seg_ev[:, 2]
                seg_rew -= body_pen * (seg_ev[:, 1] + seg_ev[:, 3])
                if first_hit_pen > 0:
                    seg_rew -= first_hit_pen * first_hit_all[lo:hi].astype("f4")
                if iframe_zero_sec > 0:
                    seg_rew -= (0.45 / 21.0) * seg_ev[:, 0] * in_iframe_all[lo:hi].astype("f4")
                if post_hit_reward_scale < 1.0:
                    post_hit_non_iframe = ((~hitless_all[lo:hi]) & (~in_iframe_all[lo:hi])).astype("f4")
                    seg_rew -= (1.0 - post_hit_reward_scale) * (0.45 / 21.0) * seg_ev[:, 0] * post_hit_non_iframe
                if hitless_dps_bonus > 0:
                    seg_rew += hitless_dps_bonus * seg_ev[:, 0] * hitless_all[lo:hi].astype("f4")
                seg_term = episode.terminated[lo:hi].copy()
                if seg_idx == len(fight.segments) - 1:
                    if is_nohit:
                        seg_rew[-1] += no_hit_bonus
                    elif bool(receipt["win"]) and trio_masks_lost == 1 and one_hit_bonus > 0:
                        seg_rew[-1] += one_hit_bonus
                    elif bool(r.get("radiant_early_stopped")):
                        last_v = int(np.flatnonzero(seg_valid)[-1])
                        seg_term[last_v] = True
                        if radiant_fail_pen > 0:
                            seg_rew[last_v] -= radiant_fail_pen
                adv, ret = native_gae(seg_rew, val.numpy(), seg_clock, seg_valid, seg_term)

                exact = (np.abs(np.diff(seg_clock) - 0.04) < 1e-4) & (fight.action[lo : hi - 1] >= 0)
                broken = np.r_[0, np.cumsum(~exact)]
                w_starts = np.arange(0, max(0, (hi - lo) - 9))
                if len(w_starts):
                    w_starts = w_starts[broken[w_starts] == broken[w_starts + 9]]
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
                        hitless_mask=torch.from_numpy(np.ascontiguousarray(hitless_all[lo:hi])),
                    )
                )
        return rollouts, evidence
    finally:
        torch.set_num_threads(prev_threads)


def update_elite_pool(
    elite_dir: Path,
    roll_res: list[dict[str, Any]],
    rollouts_dir: Path,
    rd: int,
    max_size: int = 96,
) -> tuple[int, int]:
    elite_dir.mkdir(parents=True, exist_ok=True)
    idx_path = elite_dir / "elite-index.json"
    existing: list[dict[str, Any]] = []
    if idx_path.is_file():
        try:
            existing = [
                e for e in json.loads(idx_path.read_text(encoding="utf-8"))
                if (elite_dir / str(e.get("id", ""))).is_dir()
            ]
        except Exception:
            existing = []
    by_id: dict[str, dict[str, Any]] = {str(e["id"]): e for e in existing}

    for r in roll_res:
        if not r.get("win"):
            continue
        masks_lost = sum(int(v) for v in (r.get("damage_taken") or {}).values())
        p1_dmg = max(0, 9 - int(r.get("entry_hp", 9)))
        trio_lost = int(r.get("trio_masks_lost", max(0, masks_lost - p1_dmg)))
        if trio_lost > 1:
            continue
        seed = int(r["seed"])
        eid = f"r{rd:02d}_s{seed}"
        src = rollouts_dir / r["folder"]
        ep_path = src / "episode-000.json"
        man_path = src / "manifest.json"
        if not (ep_path.is_file() and man_path.is_file()):
            continue
        item = json.loads(man_path.read_bytes())[0]
        tape_path = src / item["file"]
        if not tape_path.is_file():
            continue
        dst = elite_dir / eid
        if not (dst / item["file"]).is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ep_path, dst / "episode-000.json")
            shutil.copy2(man_path, dst / "manifest.json")
            shutil.copy2(tape_path, dst / item["file"])
        by_id[eid] = {"id": eid, "round": int(rd), "seed": seed, "masks_lost": int(masks_lost), "trio_masks_lost": trio_lost}

    def _tl(e: dict[str, Any]) -> int:
        return int(e.get("trio_masks_lost", e.get("masks_lost", 9)))

    kept = sorted(by_id.values(), key=lambda e: (_tl(e), int(e.get("masks_lost", 9)), -int(e["round"]), int(e["seed"])))
    for evicted in kept[max_size:]:
        shutil.rmtree(elite_dir / str(evicted["id"]), ignore_errors=True)
    kept = kept[:max_size]
    idx_path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    return len(kept), sum(1 for e in kept if _tl(e) == 0)


def load_elite_samples(
    elite_dir: Path,
    catalogue: torch.Tensor,
    is_v2: bool,
    max_batch: int = 12,
    rd: int = 1,
    pure_zero_only: bool = False,
) -> list[EliteSample]:
    idx_path = elite_dir / "elite-index.json"
    if not idx_path.is_file() or max_batch <= 0:
        return []
    entries = json.loads(idx_path.read_text(encoding="utf-8"))
    if not entries:
        return []

    def _tl(e: dict[str, Any]) -> int:
        return int(e.get("trio_masks_lost", e.get("masks_lost", 9)))

    zeros = [e for e in entries if _tl(e) == 0]
    ones = [e for e in entries if _tl(e) == 1]
    rng = np.random.default_rng(7700 + rd)
    chosen: list[dict[str, Any]] = []
    if zeros:
        if len(zeros) <= max_batch:
            chosen.extend(zeros)
        else:
            by_new = sorted(zeros, key=lambda e: (-int(e["round"]), int(e["seed"])))
            newest = by_new[: max(1, len(by_new) // 2)]
            n_new = min(len(newest), max(1, (max_batch * 3) // 5))
            picks = rng.choice(len(newest), size=n_new, replace=False)
            fresh = [newest[int(i)] for i in picks]
            fresh_ids = {str(e["id"]) for e in fresh}
            rest = [e for e in zeros if str(e["id"]) not in fresh_ids]
            picks2 = rng.choice(len(rest), size=max_batch - n_new, replace=False)
            chosen.extend(fresh)
            chosen.extend(rest[int(i)] for i in picks2)
    rem_one = 0 if (pure_zero_only or len(zeros) >= 12) else min(len(ones), 6, max(0, max_batch - len(chosen)))
    if rem_one > 0:
        picks = rng.choice(len(ones), size=rem_one, replace=False)
        chosen.extend(ones[int(i)] for i in picks)
    if not chosen:
        return []

    hist_fn = motion_history_4lag if is_v2 else motion_history
    lookup = {tuple(a): i for i, a in enumerate(catalogue.tolist())}
    samples_out: list[EliteSample] = []
    for e in chosen:
        folder = elite_dir / str(e["id"])
        ep_path = folder / "episode-000.json"
        man_path = folder / "manifest.json"
        if not (ep_path.is_file() and man_path.is_file()):
            continue
        receipt = json.loads(ep_path.read_bytes())
        item = json.loads(man_path.read_bytes())[0]
        source = folder / item["file"]
        if not source.is_file():
            continue
        with np.load(source, allow_pickle=False) as tape:
            arrays = {k: tape[k] for k in tape.files}
        episode = read_episode(
            {k: v for k, v in arrays.items() if not k.startswith("ppo_")},
            {"won": receipt["win"], "truncated": receipt["truncated"]},
        )
        fight = prepare_fight(episode, lookup)
        w = 1.0 if int(e.get("trio_masks_lost", e.get("masks_lost", 9))) == 0 else 0.30
        for lo, hi in fight.segments:
            seg_valid = fight.train_rows[lo:hi]
            if not seg_valid.any():
                continue
            x = torch.from_numpy(hist_fn(episode.clock, episode.observation, np.arange(lo, hi)))
            samples_out.append(
                EliteSample(
                    inputs=x,
                    action=torch.from_numpy(fight.action[lo:hi].clip(0)).long(),
                    train_rows=torch.from_numpy(seg_valid),
                    weight=w,
                )
            )
    return samples_out


def trio_stats(res: list[dict[str, Any]]) -> tuple[int, float]:
    """Return (trio_no_hit_wins, trio_avg_masks_lost) ignoring Phase-1 damage."""
    n_nohit = 0
    trio_total = 0
    for r in res:
        tot = sum(int(v) for v in (r.get("damage_taken") or {}).values())
        trio = int(r.get("trio_masks_lost", max(0, tot - max(0, 9 - int(r.get("entry_hp", 9))))))
        trio_total += trio
        n_nohit += int(bool(r.get("win")) and trio == 0)
    return n_nohit, trio_total / max(1, len(res))


def compute_stage_config(stage: int, args: argparse.Namespace) -> dict[str, Any]:
    use_rad = bool(getattr(args, "radiant_curriculum", False))
    cfg: dict[str, Any] = dict(
        temps=args.temps, actor_lr=args.actor_lr, updates=args.updates, kl_limit=args.kl_limit,
        clip_eps=args.clip_eps, proj_pen=args.proj_pen, body_pen=args.body_pen,
        first_hit_pen=args.first_hit_pen, hitless_dps_bonus=args.hitless_dps_bonus,
        post_hit_reward_scale=args.post_hit_reward_scale, post_hit_ppo_weight=args.post_hit_ppo_weight,
        no_hit_bonus=args.no_hit_bonus, one_hit_bonus=args.one_hit_bonus, ent_coef=args.ent_coef,
        elite_bc_coef=min(args.elite_bc_coef, 0.035), elite_batch=min(args.elite_batch, 14),
        pure_zero_elites=False, val_every=args.val_every,
        radiant_max_trio_hits=int(getattr(args, "radiant_max_hits_s1", 0)) if use_rad else 0,
    )
    if stage >= 2:
        cfg.update(
            temps="0.12,0.22,0.32", actor_lr=args.actor_lr * 0.80,
            kl_limit=min(args.kl_limit, 0.040), clip_eps=min(args.clip_eps, 0.16),
            proj_pen=max(args.proj_pen, 2.80), body_pen=max(args.body_pen, 2.40),
            first_hit_pen=max(args.first_hit_pen, 7.50), hitless_dps_bonus=max(args.hitless_dps_bonus, 0.010),
            post_hit_reward_scale=min(args.post_hit_reward_scale, 0.15),
            post_hit_ppo_weight=min(args.post_hit_ppo_weight, 0.50),
            no_hit_bonus=max(args.no_hit_bonus, 10.0), one_hit_bonus=0.0,
            ent_coef=min(args.ent_coef, 0.0006),
            elite_bc_coef=min(args.elite_bc_coef, 0.025),
            elite_batch=min(max(args.elite_batch, 12), 16),
            pure_zero_elites=True,
            val_every=min(args.val_every, 2),
            radiant_max_trio_hits=int(getattr(args, "radiant_max_hits_s2", 2)) if use_rad else 0,
        )
    if stage >= 3:
        cfg.update(
            temps="0.08,0.15,0.24", actor_lr=args.actor_lr * 0.55, updates=min(args.updates, 18),
            kl_limit=min(args.kl_limit, 0.025), clip_eps=min(args.clip_eps, 0.14),
            proj_pen=max(args.proj_pen, 3.20), body_pen=max(args.body_pen, 2.80),
            first_hit_pen=max(args.first_hit_pen, 11.0), post_hit_reward_scale=0.05,
            post_hit_ppo_weight=min(args.post_hit_ppo_weight, 0.35),
            no_hit_bonus=max(args.no_hit_bonus, 14.0), ent_coef=0.0,
            elite_bc_coef=min(args.elite_bc_coef, 0.020),
            elite_batch=min(max(args.elite_batch, 12), 16),
            val_every=1,
            radiant_max_trio_hits=int(getattr(args, "radiant_max_hits_s3", 1)) if use_rad else 0,
        )
    return cfg


def next_curriculum_stage(
    cur_stage: int, *, v_wins: int, val_episodes: int, v_trio_nohit: int, v_trio_dmg: float, collapsed: bool
) -> int:
    if collapsed:
        return max(1, cur_stage - 1)
    if v_wins < val_episodes - 1:
        return cur_stage
    if cur_stage <= 2 and (v_trio_dmg < 0.45 or v_trio_nohit >= round(0.75 * val_episodes)):
        return 3
    if cur_stage == 1 and (v_trio_dmg < 1.25 or v_trio_nohit >= round(0.30 * val_episodes)):
        return 2
    return cur_stage


def sync_to_drive(conf: Path | None, dest: str, paths: list[Path]) -> None:
    if conf is None or not conf.is_file() or not dest:
        return
    for p in paths:
        if p.is_file():
            subprocess.run(["rclone", "--config", str(conf), "copy", str(p), dest], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-learner", type=Path, required=True)
    ap.add_argument("--init-native", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--world-val", type=Path, required=True)
    ap.add_argument("--opener-pt", type=Path, default=None)
    ap.add_argument("--game-root", type=Path, default=Path("/content/hk-linux-base"))
    ap.add_argument("--repo-root", type=Path, default=Path("/content/hkrl-repo"))
    ap.add_argument("--observe-script", type=Path, default=Path("/content/hkrl-repo/observe-collect.py"))
    ap.add_argument("--work-dir", type=Path, default=Path("/content/cloud-online-ppo"))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--rollouts-per-round", type=int, default=64)
    ap.add_argument("--val-episodes", type=int, default=30)
    ap.add_argument("--first-train-seed", type=int, default=11000001)
    ap.add_argument("--first-val-seed", type=int, default=11100001)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--actor-lr", type=float, default=1.5e-5)
    ap.add_argument("--kl-limit", type=float, default=0.012)
    ap.add_argument("--proj-pen", type=float, default=1.25)
    ap.add_argument("--body-pen", type=float, default=0.95)
    ap.add_argument("--no-hit-bonus", type=float, default=2.50)
    ap.add_argument("--offline-kl-coef", type=float, default=0.04)
    ap.add_argument("--temps", type=str, default="0.20,0.35,0.50")
    ap.add_argument("--clip-eps", type=float, default=0.16)
    ap.add_argument("--pre-proj-mult", type=float, default=2.5)
    ap.add_argument("--pre-body-mult", type=float, default=1.9)
    ap.add_argument("--first-hit-pen", type=float, default=3.00)
    ap.add_argument("--iframe-zero-sec", type=float, default=1.30)
    ap.add_argument("--hitless-dps-bonus", type=float, default=0.006)
    ap.add_argument("--one-hit-bonus", type=float, default=1.00)
    ap.add_argument("--offline-kl-decay-rounds", type=int, default=6)
    ap.add_argument("--anchor-kl-coef", type=float, default=0.015)
    ap.add_argument("--dash-drift-mult", type=float, default=1.45)
    ap.add_argument("--elite-bc-coef", type=float, default=0.035)
    ap.add_argument("--elite-batch", type=int, default=12)
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--holdout-episodes", type=int, default=32)
    ap.add_argument("--first-holdout-seed", type=int, default=11400001)
    ap.add_argument("--behavior-kl-coef", type=float, default=0.025)
    ap.add_argument("--critic-warmup", type=int, default=8)
    ap.add_argument("--value-clip", type=float, default=0.2)
    ap.add_argument("--post-hit-reward-scale", type=float, default=1.0)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--collapse-min-wins", type=int, default=27)
    ap.add_argument("--collapse-dmg-margin", type=float, default=1.50)
    ap.add_argument("--auto-curriculum", action="store_true")
    ap.add_argument("--auto-formal100", action="store_true")
    ap.add_argument("--radiant-curriculum", action="store_true")
    ap.add_argument("--radiant-max-hits-s1", type=int, default=0)
    ap.add_argument("--radiant-max-hits-s2", type=int, default=2)
    ap.add_argument("--radiant-max-hits-s3", type=int, default=1)
    ap.add_argument("--radiant-fail-pen", type=float, default=4.0)
    ap.add_argument("--post-hit-ppo-weight", type=float, default=0.85)
    ap.add_argument("--ent-coef", type=float, default=0.0015)
    ap.add_argument("--formal100-first-seed", type=int, default=12000001)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nographics", action="store_true")
    ap.add_argument("--no-batchmode", action="store_true")
    ap.add_argument("--soft-gl", action="store_true")
    ap.add_argument("--rclone-conf", type=Path, default=None)
    ap.add_argument("--drive-dest", type=str, default="")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpus = os.cpu_count() or 8
    workers = args.workers if args.workers > 0 else min(32, max(4, cpus - 4))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    mod_sha = sha256(args.game_root / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll")

    off_saved = torch.load(args.init_learner, map_location="cpu", weights_only=True)
    catalogue = off_saved["catalogue"]
    cfg = off_saved["provenance"]["scaled_cfg"]
    offline_sha = sha256(args.init_learner)
    audit_sha = off_saved["provenance"]["audit_sha256"]
    worlds, validated = load_world(args.world, args.world_val, offline_sha, audit_sha, 2 * cfg["d_model"], len(catalogue))
    worlds = [w.to(device) for w in worlds]

    offline_actor, _, is_v2 = build_actor_critic_from_cfg(catalogue, cfg)
    offline_actor.load_state_dict(off_saved["state"])
    offline_actor.to(device).eval().requires_grad_(False)

    cur_pt = args.work_dir / "champion.pt"
    cur_bin = args.work_dir / "champion.bin"
    if args.resume and cur_pt.is_file() and cur_bin.is_file():
        print(f"RESUMING existing champion sha256={sha256(cur_bin)}", flush=True)
    else:
        shutil.copy2(args.init_learner, cur_pt)
        export_native_from_pt(cur_pt, cur_bin, args.init_native, opener_pt=args.opener_pt)

    work_pt = (args.work_dir / "latest.pt") if args.continuous else cur_pt
    work_bin = (args.work_dir / "latest.bin") if args.continuous else cur_bin
    if args.continuous:
        if args.resume and work_pt.is_file() and work_bin.is_file():
            print(f"RESUMING continuous working policy latest.bin sha256={sha256(work_bin)}", flush=True)
        else:
            shutil.copy2(cur_pt, work_pt)
            shutil.copy2(cur_bin, work_bin)
            print(f"INITIALIZED continuous working policy from champion sha256={sha256(work_bin)}", flush=True)

    val_seeds = list(range(args.first_val_seed, args.first_val_seed + args.val_episodes))
    hist_path = args.work_dir / "rl-history.json"
    if args.resume and hist_path.is_file():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        promoted_entries = [h for h in history if h.get("promoted") and "wins" in h]
        best_entry = promoted_entries[-1] if promoted_entries else history[0]
        best_wins = int(best_entry["wins"])
        best_nohit = int(best_entry["nohit"])
        best_dmg = float(best_entry["avg_dmg"])
        n_eps = max(1, args.val_episodes)
        best_trio_nohit = int(best_entry.get("trio_nohit", best_nohit))
        best_trio_dmg = float(
            best_entry.get("trio_avg_dmg", (best_entry["trio_hits"] / n_eps) if "trio_hits" in best_entry else best_dmg)
        )
        peak_full_nohit = max(
            [int(h["nohit"]) for h in history if "nohit" in h and int(h.get("wins", 0)) >= args.val_episodes - 1] or [0]
        )
        cur_stage = int(history[-1].get("next_stage", 1)) if args.auto_curriculum else 1
        start_round = max(int(h["round"]) for h in history) + 1
        print(
            f"RESUMED history from {hist_path}: start_round={start_round} stage={cur_stage} "
            f"(best: wins={best_wins}/{args.val_episodes}, nohit={best_nohit}, dmg={best_dmg:.3f}, "
            f"trio_nohit={best_trio_nohit}, trio_dmg={best_trio_dmg:.3f})",
            flush=True,
        )
    else:
        print(f"=== [ROUND 0] Baseline {args.val_episodes}-ep validation (workers={workers}) ===", flush=True)
        base_res = run_parallel_cohort(
            seeds=val_seeds, workers=min(workers, len(val_seeds)),
            game_root=args.game_root, repo_root=args.repo_root, observe_script=args.observe_script,
            actor_bin=cur_bin, run_dir=args.work_dir / "val-r00",
            mode="eval", temperatures={}, draw_seeds={}, use_nographics=args.nographics, no_batchmode=args.no_batchmode, soft_gl=args.soft_gl,
        )
        best_wins = sum(int(r["win"]) for r in base_res)
        best_nohit = sum(int(r["no_hit_win"]) for r in base_res)
        best_dmg = sum(sum(int(v) for v in r["damage_taken"].values()) for r in base_res) / max(1, len(base_res))
        best_trio_nohit, best_trio_dmg = trio_stats(base_res)
        peak_full_nohit = best_nohit if best_wins >= len(base_res) - 1 else 0
        cur_stage = 1
        print(f"BASELINE: wins={best_wins}/{len(base_res)} nohit={best_nohit} avg_dmg={best_dmg:.3f}", flush=True)

        history = [{
            "round": 0, "promoted": True, "wins": best_wins, "nohit": best_nohit, "avg_dmg": round(best_dmg, 3),
            "trio_nohit": best_trio_nohit, "trio_avg_dmg": round(best_trio_dmg, 3), "next_stage": 1,
        }]
        hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        sync_to_drive(args.rclone_conf, args.drive_dest, [hist_path, cur_pt, cur_bin])
        start_round = 1

    elite_dir = args.work_dir / "elite-pool"
    if args.resume and not (elite_dir / "elite-index.json").is_file():
        for cs in sorted(args.work_dir.glob("round-*/rollouts/cloud-summary.json")):
            try:
                prev_rd = int(cs.parent.parent.name.split("-")[1])
                rows = json.loads(cs.read_text(encoding="utf-8")).get("rows", [])
                e_tot, e_z = update_elite_pool(elite_dir, rows, cs.parent, prev_rd)
                print(f"BACKFILLED_ELITE_POOL round={prev_rd} total={e_tot} zero_hit={e_z}", flush=True)
            except Exception as exc:
                print(f"BACKFILL_ELITE_WARN {cs}: {exc}", flush=True)

    for rd in range(start_round, args.rounds + 1):
        rdir = args.work_dir / f"round-{rd:02d}"
        sc = compute_stage_config(cur_stage if args.auto_curriculum else 1, args)
        print(
            f"STAGE_CONFIG round={rd} stage={cur_stage} "
            + " ".join(f"{k}={v}" for k, v in sc.items()),
            flush=True,
        )
        t_first = args.first_train_seed + (rd - 1) * args.rollouts_per_round
        t_seeds = list(range(t_first, t_first + args.rollouts_per_round))
        temp_list = [float(x.strip()) for x in str(sc["temps"]).split(",") if x.strip()]
        if not temp_list:
            temp_list = [0.20, 0.35, 0.50]
        temps = {s: temp_list[(i * len(temp_list)) // len(t_seeds)] for i, s in enumerate(t_seeds)}
        dseeds = {s: 9900000 + rd * 10000 + i * 53 for i, s in enumerate(t_seeds)}
        if args.resume:
            bin_sha_cur = sha256(work_bin)
            rollouts_dir = rdir / "rollouts"
            for s in t_seeds:
                rf = rollouts_dir / f"games/{s}-policy/result.json"
                if rf.is_file():
                    try:
                        rj = json.loads(rf.read_bytes())
                        epf = rollouts_dir / rj["folder"] / "episode-000.json"
                        if rj.get("policy_sha256") == bin_sha_cur and epf.is_file():
                            epj = json.loads(epf.read_bytes())
                            if float(epj.get("ppo_temperature", 0.0)) > 0:
                                temps[s] = float(epj["ppo_temperature"])
                                dseeds[s] = int(epj["sampler_draw_seed"])
                    except Exception:
                        pass

        rad_hits = int(sc.get("radiant_max_trio_hits", 0))
        print(f"\n=== [ROUND {rd}/{args.rounds}] Collecting {len(t_seeds)} sampled PPO fights (workers={workers}, radiant_max_trio_hits={rad_hits}) ===", flush=True)
        roll_res = run_parallel_cohort(
            seeds=t_seeds, workers=min(workers, len(t_seeds)),
            game_root=args.game_root, repo_root=args.repo_root, observe_script=args.observe_script,
            actor_bin=work_bin, run_dir=rdir / "rollouts",
            mode="ppo", temperatures=temps, draw_seeds=dseeds, use_nographics=args.nographics, no_batchmode=args.no_batchmode, soft_gl=args.soft_gl,
            radiant_max_trio_hits=rad_hits,
        )

        elite_dir = args.work_dir / "elite-pool"
        elite_total, elite_zeros = update_elite_pool(elite_dir, roll_res, rdir / "rollouts", rd)
        print(f"ROUND {rd} ELITE POOL: total={elite_total} zero_hit={elite_zeros}", flush=True)

        sv = torch.load(work_pt, map_location="cpu", weights_only=True)
        actor, critic, _ = build_actor_critic_from_cfg(catalogue, cfg)
        actor.load_state_dict(sv["state"])
        critic.load_state_dict(sv["critic"])
        actor.eval()
        critic.eval()

        rollouts, evidence = build_verified_cloud_rollouts(
            roll_res, rdir / "rollouts", actor, critic, catalogue, is_v2,
            temps, dseeds, sha256(work_bin), mod_sha,
            no_hit_bonus=sc["no_hit_bonus"], proj_pen=sc["proj_pen"], body_pen=sc["body_pen"],
            first_hit_pen=sc["first_hit_pen"], iframe_zero_sec=args.iframe_zero_sec,
            hitless_dps_bonus=sc["hitless_dps_bonus"], one_hit_bonus=sc["one_hit_bonus"],
            post_hit_reward_scale=sc["post_hit_reward_scale"],
            radiant_fail_pen=args.radiant_fail_pen,
        )
        actor.to(device).eval()
        critic.to(device).eval()
        anchor_actor = deepcopy(actor).eval().requires_grad_(False)

        eff_offline_kl = (
            args.offline_kl_coef * max(0.0, 1.0 - (rd - 1) / args.offline_kl_decay_rounds)
            if args.offline_kl_decay_rounds > 0
            else args.offline_kl_coef
        )
        if args.continuous:
            rollback_count = sum(1 for h in history if h.get("collapse_rollback"))
            eff_lr = sc["actor_lr"] * (0.75 ** min(rollback_count, 3))
        else:
            stall_rounds = 0
            for h in reversed(history):
                if int(h.get("round", 0)) <= 0 or h.get("holdout"):
                    continue
                if h.get("promoted"):
                    break
                stall_rounds += 1
            eff_lr = sc["actor_lr"] * 0.75 if stall_rounds >= 3 else sc["actor_lr"]
        elite_samples = (
            load_elite_samples(
                elite_dir, catalogue, is_v2, max_batch=sc["elite_batch"], rd=rd,
                pure_zero_only=bool(sc["pure_zero_elites"]),
            )
            if sc["elite_bc_coef"] > 0.0
            else []
        )

        progress = optimize_safe_wp22(
            actor, offline_actor, critic, rollouts, worlds, validated, catalogue,
            sc["updates"], eff_lr, sc["kl_limit"], device,
            critic_warmup=args.critic_warmup,
            offline_kl_coef=eff_offline_kl,
            clip_eps=sc["clip_eps"],
            pre_proj_mult=args.pre_proj_mult,
            pre_body_mult=args.pre_body_mult,
            anchor_actor=anchor_actor,
            anchor_kl_coef=args.anchor_kl_coef,
            dash_drift_mult=args.dash_drift_mult,
            elite_samples=elite_samples,
            elite_bc_coef=sc["elite_bc_coef"],
            behavior_kl_coef=args.behavior_kl_coef,
            value_clip=args.value_clip,
            critic_lr=1.0e-4 if args.continuous else 8.0e-5,
            critic_warmup_lr=1.5e-4 if args.continuous else 1.2e-4,
            post_hit_ppo_weight=sc["post_hit_ppo_weight"],
            ent_coef=sc["ent_coef"],
        )
        accepted = sum(int(bool(p["accepted"])) for p in progress)
        print(
            f"ROUND {rd} GPU PPO: accepted_updates={accepted} off_kl={eff_offline_kl:.4f} "
            f"lr={eff_lr:.2e} elite_segs={len(elite_samples)} progress={json.dumps(progress)}",
            flush=True,
        )
        if not accepted:
            history.append({"round": rd, "promoted": False, "reason": "no_ppo_step_accepted", "next_stage": cur_stage})
            hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            sync_to_drive(args.rclone_conf, args.drive_dest, [hist_path, args.work_dir / "live.log"])
            continue

        cand_pt = rdir / "candidate.pt"
        cand_bin = rdir / "candidate.bin"
        torch.save(
            {
                **sv,
                "state": {k: v.cpu() for k, v in actor.state_dict().items()},
                "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
                "cloud_ppo_round": rd,
            },
            cand_pt,
        )
        export_native_from_pt(cand_pt, cand_bin, args.init_native, opener_pt=args.opener_pt)

        do_val = (not args.continuous) or (sc["val_every"] <= 1) or (rd % sc["val_every"] == 0) or (rd == args.rounds)
        if not do_val:
            shutil.copy2(cand_pt, work_pt)
            shutil.copy2(cand_bin, work_bin)
            r_wins = sum(int(r["win"]) for r in roll_res)
            r_nohit = sum(int(r["no_hit_win"]) for r in roll_res)
            r_dmg = sum(sum(int(v) for v in r["damage_taken"].values()) for r in roll_res) / max(1, len(roll_res))
            last_kl = float(progress[-1]["kl"])
            print(
                f"ROUND {rd} CONTINUOUS_ADVANCE (val skipped): rollout={r_wins}/{len(roll_res)} "
                f"nohit={r_nohit} avg_dmg={r_dmg:.3f} kl={last_kl:.5f} elite={elite_total}(0hit={elite_zeros})",
                flush=True,
            )
            history.append({
                "round": rd,
                "promoted": False,
                "continuous_step": True,
                "val_skipped": True,
                "rollout_wins": r_wins,
                "rollout_nohit": r_nohit,
                "rollout_avg_dmg": round(r_dmg, 3),
                "accepted_updates": accepted,
                "final_kl": round(last_kl, 5),
                "offline_kl_coef": round(eff_offline_kl, 4),
                "elite_total": elite_total,
                "elite_zeros": elite_zeros,
                "stage": cur_stage,
                "next_stage": cur_stage,
            })
            hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            sync_to_drive(args.rclone_conf, args.drive_dest, [hist_path, args.work_dir / "live.log", work_pt, work_bin])
            print(f"ROUND {rd} SYNCED TO {args.drive_dest} (continuous_step=True)", flush=True)
            continue

        print(f"=== [ROUND {rd}] Validating candidate on {args.val_episodes} episodes ===", flush=True)
        v_res = run_parallel_cohort(
            seeds=val_seeds, workers=min(workers, len(val_seeds)),
            game_root=args.game_root, repo_root=args.repo_root, observe_script=args.observe_script,
            actor_bin=cand_bin, run_dir=rdir / "val",
            mode="eval", temperatures={}, draw_seeds={}, use_nographics=args.nographics, no_batchmode=args.no_batchmode, soft_gl=args.soft_gl,
        )
        v_wins = sum(int(r["win"]) for r in v_res)
        v_nohit = sum(int(r["no_hit_win"]) for r in v_res)
        v_dmg_total = sum(sum(int(v) for v in r["damage_taken"].values()) for r in v_res)
        v_dmg = v_dmg_total / max(1, len(v_res))
        v_p1_hits = sum(max(0, 9 - int(r.get("entry_hp", 9))) for r in v_res)
        v_trio_hits = max(0, v_dmg_total - v_p1_hits)
        v_trio_nohit, v_trio_dmg = trio_stats(v_res)

        v_wr = v_wins / max(1, len(v_res))
        best_wr = best_wins / max(1, args.val_episodes)
        improved = (
            (v_wr > best_wr + 1e-6)
            or (abs(v_wr - best_wr) <= 1e-6 and v_trio_nohit > best_trio_nohit and v_trio_dmg <= best_trio_dmg + 0.15)
            or (abs(v_wr - best_wr) <= 1e-6 and v_trio_dmg < best_trio_dmg - 0.05 and v_trio_nohit >= best_trio_nohit - 1)
            or (abs(v_wr - best_wr) <= 1e-6 and v_trio_nohit == best_trio_nohit and v_trio_dmg < best_trio_dmg - 0.02)
        )
        nohit_saved = False
        if v_wins >= len(v_res) - 1 and v_nohit > peak_full_nohit:
            peak_full_nohit = v_nohit
            shutil.copy2(cand_pt, args.work_dir / "best-nohit.pt")
            shutil.copy2(cand_bin, args.work_dir / "best-nohit.bin")
            nohit_saved = True
        if improved:
            best_wins, best_nohit, best_dmg = v_wins, v_nohit, v_dmg
            best_trio_nohit, best_trio_dmg = v_trio_nohit, v_trio_dmg
            shutil.copy2(cand_pt, cur_pt)
            shutil.copy2(cand_bin, cur_bin)
        collapsed = False
        if args.continuous:
            collapsed = (v_wins < args.collapse_min_wins) or (v_trio_dmg > best_trio_dmg + args.collapse_dmg_margin)
            if collapsed:
                shutil.copy2(cur_pt, work_pt)
                shutil.copy2(cur_bin, work_bin)
                print(
                    f"ROUND {rd} COLLAPSE_ROLLBACK: wins={v_wins}/{len(v_res)} avg_dmg={v_dmg:.3f} -> reverted latest to champion",
                    flush=True,
                )
            else:
                shutil.copy2(cand_pt, work_pt)
                shutil.copy2(cand_bin, work_bin)
                print(f"ROUND {rd} CONTINUOUS_KEPT: latest advanced (promoted={improved})", flush=True)
        prev_stage = cur_stage
        if args.auto_curriculum:
            cur_stage = next_curriculum_stage(
                cur_stage, v_wins=v_wins, val_episodes=len(v_res), v_trio_nohit=v_trio_nohit,
                v_trio_dmg=v_trio_dmg, collapsed=collapsed,
            )
            if cur_stage != prev_stage:
                print(f"ROUND {rd} STAGE_TRANSITION {prev_stage} -> {cur_stage}", flush=True)
        print(
            f"ROUND {rd} VAL: wins={v_wins}/{len(v_res)} nohit={v_nohit} avg_dmg={v_dmg:.3f} "
            f"(p1={v_p1_hits}, trio={v_trio_hits}, trio_nohit={v_trio_nohit}) | "
            f"PROMOTED={improved} (best: {best_wins}/{len(v_res)}, nohit={best_nohit}, dmg={best_dmg:.3f}, "
            f"trio_nohit={best_trio_nohit}, trio_dmg={best_trio_dmg:.3f})",
            flush=True,
        )
        rec = {
            "round": rd,
            "promoted": improved,
            "collapse_rollback": collapsed,
            "continuous_mode": bool(args.continuous),
            "wins": v_wins,
            "nohit": v_nohit,
            "avg_dmg": round(v_dmg, 3),
            "p1_hits": v_p1_hits,
            "trio_hits": v_trio_hits,
            "trio_nohit": v_trio_nohit,
            "trio_avg_dmg": round(v_trio_dmg, 3),
            "stage": prev_stage,
            "next_stage": cur_stage,
            "accepted_updates": accepted,
            "final_kl": round(float(progress[-1]["kl"]), 5),
            "final_greedy_kl": progress[-1].get("greedy_kl"),
            "final_entropy": progress[-1].get("entropy"),
            "offline_kl_coef": round(eff_offline_kl, 4),
            "elite_total": elite_total,
            "elite_zeros": elite_zeros,
        }
        history.append(rec)
        sync_paths = [hist_path, args.work_dir / "live.log"]
        if improved:
            sync_paths.extend([cur_pt, cur_bin])
        if nohit_saved:
            sync_paths.extend([args.work_dir / "best-nohit.pt", args.work_dir / "best-nohit.bin"])
        if args.continuous and not collapsed:
            sync_paths.extend([work_pt, work_bin])
        if rd % 2 == 0:
            try:
                tar_path = args.work_dir / "elite-pool.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tf:
                    tf.add(elite_dir, arcname="elite-pool")
                sync_paths.append(tar_path)
            except Exception as exc:
                print(f"ELITE_TAR_WARN {exc}", flush=True)

        run_holdout = (
            (args.holdout_every > 0 and rd % args.holdout_every == 0)
            or (improved and v_nohit >= 10 and v_nohit >= best_nohit)
        )
        if run_holdout and args.holdout_episodes > 0:
            holdout_idx = sum(1 for h in history if h.get("holdout"))
            h_first = args.first_holdout_seed + holdout_idx * args.holdout_episodes
            h_seeds = list(range(h_first, h_first + args.holdout_episodes))
            audit_bin = work_bin if (args.continuous and not collapsed) else cur_bin
            print(f"=== [ROUND {rd}] Running fresh {args.holdout_episodes}-ep holdout audit ({h_first}..{h_seeds[-1]}) ===", flush=True)
            h_res = run_parallel_cohort(
                seeds=h_seeds, workers=min(workers, len(h_seeds)),
                game_root=args.game_root, repo_root=args.repo_root, observe_script=args.observe_script,
                actor_bin=audit_bin, run_dir=rdir / "holdout",
                mode="eval", temperatures={}, draw_seeds={}, use_nographics=args.nographics, no_batchmode=args.no_batchmode, soft_gl=args.soft_gl,
            )
            h_wins = sum(int(r["win"]) for r in h_res)
            h_nohit = sum(int(r["no_hit_win"]) for r in h_res)
            h_dmg_tot = sum(sum(int(v) for v in r["damage_taken"].values()) for r in h_res)
            h_dmg = h_dmg_tot / max(1, len(h_res))
            h_p1 = sum(max(0, 9 - int(r.get("entry_hp", 9))) for r in h_res)
            h_trio = max(0, h_dmg_tot - h_p1)
            print(
                f"HOLDOUT_AUDIT r={rd}: wins={h_wins}/{len(h_res)} nohit={h_nohit} "
                f"avg_dmg={h_dmg:.3f} (p1={h_p1}, trio={h_trio})",
                flush=True,
            )
            rec["holdout"] = {
                "first_seed": h_first,
                "episodes": len(h_res),
                "wins": h_wins,
                "nohit": h_nohit,
                "avg_dmg": round(h_dmg, 3),
                "p1_hits": h_p1,
                "trio_hits": h_trio,
                "audited_policy": "latest" if (args.continuous and not collapsed) else "champion",
            }

        if (
            args.auto_formal100
            and not collapsed
            and v_wins == len(v_res)
            and v_nohit >= int(round(0.90 * len(v_res)))
            and not any(h.get("formal100") and rd - int(h["round"]) < 3 for h in history[:-1])
        ):
            f_idx = sum(1 for h in history if h.get("formal100"))
            f_first = args.formal100_first_seed + f_idx * 100
            f_seeds = list(range(f_first, f_first + 100))
            f_dir = args.work_dir / f"formal100-r{rd:02d}"
            print(f"=== [ROUND {rd}] AUTO FORMAL100 on fresh seeds {f_first}..{f_seeds[-1]} ===", flush=True)
            f_res = run_parallel_cohort(
                seeds=f_seeds, workers=min(workers, len(f_seeds)),
                game_root=args.game_root, repo_root=args.repo_root, observe_script=args.observe_script,
                actor_bin=cand_bin, run_dir=f_dir,
                mode="eval", temperatures={}, draw_seeds={}, use_nographics=args.nographics, no_batchmode=args.no_batchmode, soft_gl=args.soft_gl,
            )
            f_wins = sum(int(r["win"]) for r in f_res)
            f_nohit = sum(int(r["no_hit_win"]) for r in f_res)
            f_trio_nohit, f_trio_dmg = trio_stats(f_res)
            f_tot = sum(sum(int(v) for v in r["damage_taken"].values()) for r in f_res)
            f_p1 = sum(max(0, 9 - int(r.get("entry_hp", 9))) for r in f_res)
            print(
                f"FORMAL100 r={rd}: wins={f_wins}/{len(f_res)} nohit={f_nohit} trio_nohit={f_trio_nohit} "
                f"avg_dmg={f_tot / max(1, len(f_res)):.3f} (p1={f_p1})",
                flush=True,
            )
            rec["formal100"] = {
                "first_seed": f_first, "episodes": len(f_res), "wins": f_wins, "nohit": f_nohit,
                "trio_nohit": f_trio_nohit, "avg_dmg": round(f_tot / max(1, len(f_res)), 3),
                "p1_hits": f_p1, "trio_hits": max(0, f_tot - f_p1),
            }
            frozen_pt = args.work_dir / f"formal100-r{rd:02d}-candidate.pt"
            frozen_bin = args.work_dir / f"formal100-r{rd:02d}-candidate.bin"
            shutil.copy2(cand_pt, frozen_pt)
            shutil.copy2(cand_bin, frozen_bin)
            receipts = args.work_dir / f"formal100-r{rd:02d}-receipts.tar.gz"
            try:
                with tarfile.open(receipts, "w:gz") as tf:
                    tf.add(f_dir / "cloud-summary.json", arcname="cloud-summary.json")
                    for p in sorted(f_dir.glob("games/*-policy/attempt-*/game/episode-000.json")):
                        tf.add(p, arcname=str(p.relative_to(f_dir)))
            except Exception as exc:
                print(f"RECEIPT_TAR_WARN {exc}", flush=True)
            sync_paths.extend([frozen_pt, frozen_bin, receipts])
            if f_wins >= 98:
                shutil.copy2(cand_pt, args.work_dir / "GOAL1_CHAMPION.pt")
                shutil.copy2(cand_bin, args.work_dir / "GOAL1_CHAMPION.bin")
                sync_paths.extend([args.work_dir / "GOAL1_CHAMPION.pt", args.work_dir / "GOAL1_CHAMPION.bin"])
                print("GOAL1_ACCEPTED_100EP", flush=True)
            if f_wins >= 98 and f_nohit >= 95:
                shutil.copy2(cand_pt, args.work_dir / "GOAL2_CHAMPION.pt")
                shutil.copy2(cand_bin, args.work_dir / "GOAL2_CHAMPION.bin")
                sync_paths.extend([args.work_dir / "GOAL2_CHAMPION.pt", args.work_dir / "GOAL2_CHAMPION.bin"])
                hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
                sync_to_drive(args.rclone_conf, args.drive_dest, sync_paths)
                print("GOAL2_ACCEPTED_100EP", flush=True)
                print("CLOUD_ONLINE_PPO_ALL_DONE", json.dumps(history[-1], indent=2), flush=True)
                return

        hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        sync_to_drive(args.rclone_conf, args.drive_dest, sync_paths)
        print(f"ROUND {rd} SYNCED TO {args.drive_dest} (promoted={improved})", flush=True)

    print("CLOUD_ONLINE_PPO_ALL_DONE", json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
