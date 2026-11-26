"""WP22-Gold+ Scaled-Down Balanced Recurrent IQL + AWR + Future-Occupancy + Maneuver
+ Action-Conditioned World(512, 75) Trainer (Zero C# DLL changes, Magic 0x3E4C5149).

Key WP22-Gold+ fixes over WP18-WP21:
  1. Architecture slimming & rebalancing (exact 1.41M Actor, 1.47M Critic, zero DLL rebuild):
     d_boss=96, d_disc=80, d_past=64, d_model=256, d_ffn=512, n_blocks=3, n_gru_layers=1.
     GRU params cut 1.77M -> 0.39M (cures trajectory memorization); disc MLP raised 3K -> 7.7K;
     World model dynamically follows 2 * d_model = World(512, 75).
  2. Clean de-duplicated dataset & de-noised holdouts:
     - Never double-imports training-100.zip (100% subset of teacher-144.zip).
     - Splits teacher-144 by training-100 provenance: 79 new 99% teacher train wins + 20 clean
       new-teacher holdout; 34 legacy 77% teacher train fights demoted to <=8% background pool.
     - Integrates Track A (+60 new 99% pure-teacher fights, wins only -> ~138 clean teacher wins)
       and Track B (+60 strong-model DAgger fights merged with wp19/wp20 18 fights -> 62 train
       + 16 high-SNR strong-model holdout, keeping all strong-model losses for corrective labels).
  3. Anti-overfitting without breaking physical safety constraints:
     - Trained from scratch (no hot-start patch), Actor dropout=0.12 / wd=1.5e-3,
       Critic dropout=0.10 / wd=1e-3, NO global label_smoothing, preserves all 6 physical
       safety penalties, caps corrective multiplier <= 3.5x, EMA(0.996) saved to ckpt["state"]
       (+ raw_state preserved for continuation).
     - Balanced batch=96 sampler: 50% clean new teacher + 31.25% strong DAgger + 10.4% pre-hit
       1.0s danger anchors (6% proj + 4% body) + 8.3% legacy background.
  4. Verified scaled_opener binding (WP19-PPO / WP18 opener sha256=7540e98eb3ee7a05...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

_SISTERS_DIR = Path(__file__).resolve().parent
for _p in (_SISTERS_DIR, _SISTERS_DIR / '_lib', _SISTERS_DIR.parents[1] / 'python'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

_HERE = Path(__file__).resolve().parent
WP18_DIR = Path("D:/CodexArtifacts/HKRL-Handoff/WP18-ScaledDRL")
for _cand in (_HERE, WP18_DIR, WP18_DIR / "HKRL-wp18", Path("C:/Users/Mr.Z/.codex/worktrees/sisters-ci-repair/HKRL")):
    if str(_cand) not in sys.path and _cand.is_dir():
        sys.path.insert(0, str(_cand))
    if (_cand / "scripts").is_dir() and str(_cand / "scripts") not in sys.path:
        sys.path.insert(0, str(_cand / "scripts"))
    if (_cand / "python").is_dir() and str(_cand / "python") not in sys.path:
        sys.path.insert(0, str(_cand / "python"))

from audit_sisters_mainline_replay import Episode
from sisters_future_occupancy import future_occupancy
from train_hkrl_wm_recurrent_rl import (
    BURN_IN,
    UNROLL,
    Fight,
    native_catalogue,
    prepare_fight,
    read_fights,
    sha256,
)
from train_sisters_recurrent_world import (
    HORIZONS,
    STEPS,
    World,
    assess as assess_world,
    fit as fit_world,
)
from train_sisters_timing_iql import CONTROL, DISCS, FOES, future_movements, motion_history
from train_wp18_scaled_drl import (
    WP18Fight,
    build_hazard_mask,
    build_scaled_world_game,
    evaluate_wp18,
    greedy_maneuver_loss,
    load_packed_cohort,
    maneuver_groups,
    mirror_observation_192,
    pre_window,
)
from wp18_scaled_model import (
    ScaledRecurrentActor,
    ScaledRecurrentValues,
    export_two_phase_scaled,
    extract_opening_bytes,
)
from wp22_attn_model import (
    AttnRecurrentActor,
    AttnRecurrentValues,
    export_two_phase_attn,
    motion_history_4lag,
)
from hkrl.spaces import BUTTON_BITS


def compute_wp22_weights(
    obs: np.ndarray,
    clock: np.ndarray,
    events: np.ndarray,
    student_actions: np.ndarray,
    labels: np.ndarray,
    cat_np: np.ndarray,
    rare_np: np.ndarray,
    is_corrective: bool,
    is_strong_dagger: bool = False,
    anti_disc_v2: bool = False,
    disc_w_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated WP22 weights (Gen-1 & Gen-2 Anti-Disc / Near-No-Hit).

    Returns (weights, body_anchor_mask, proj_anchor_mask, teacher_disc_anchor_mask).
    """
    valid = labels >= 0
    trio = obs[:, 23] > 0.5
    w = np.zeros(len(labels), dtype="f4")

    x_world = obs[:, 0] * 10.0
    low_y = (obs[:, 1] * 8.0 + 7.41) < 8.8
    near_edge = trio & (np.abs(x_world) > 6.85) & low_y
    danger_edge = trio & (np.abs(x_world) > 8.35) & low_y

    # 1.0s pre-hit windows for danger anchor sampling; focused pre-hit for loss upweighting
    before_body_1s = trio & pre_window(clock, events[:, 1] > 0, 0.80)
    before_proj_1s = trio & pre_window(clock, events[:, 2] > 0, 1.00)
    before_spike_1s = trio & pre_window(clock, events[:, 3] > 0, 0.64)

    before_body = trio & pre_window(clock, events[:, 1] > 0, 0.44)
    before_proj = trio & pre_window(clock, events[:, 2] > 0, 0.72 if anti_disc_v2 else 0.56)
    before_spike = trio & pre_window(clock, events[:, 3] > 0, 0.40)

    disc_present = (obs[:, 132] > 0.5) | (obs[:, 146] > 0.5)
    d0_close = (obs[:, 132] > 0.5) & (np.abs(obs[:, 133]) < 0.65) & (((obs[:, 133] * obs[:, 135]) < 0.03) | (np.abs(obs[:, 133]) < 0.32))
    d1_close = (obs[:, 146] > 0.5) & (np.abs(obs[:, 147]) < 0.65) & (((obs[:, 147] * obs[:, 149]) < 0.03) | (np.abs(obs[:, 147]) < 0.32))
    disc_close = trio & (d0_close | d1_close)

    body_anchor = valid & (before_body_1s | before_spike_1s | danger_edge)
    proj_anchor = valid & before_proj_1s
    teacher_disc_anchor = valid & (disc_close | (trio & disc_present))

    if not valid.any():
        return w, body_anchor, proj_anchor, teacher_disc_anchor

    lbl = labels.clip(0)
    w[valid] = rare_np[lbl[valid]]

    prior_move = np.rint(obs[:, CONTROL]).astype("i8")
    prior_aim = np.rint(obs[:, CONTROL + 1]).astype("i8")
    prior_bits_mat = (obs[:, CONTROL + 4 : CONTROL + 13] > 0.5).astype("i8")
    prior_btns = (prior_bits_mat << np.arange(9, dtype="i8")).sum(1)
    prior_cmd = np.column_stack((prior_move, prior_aim, prior_btns))

    tgt_cmd = cat_np[lbl]
    switched = (tgt_cmd != prior_cmd).any(1)
    w[valid & switched] *= 1.30

    bosses = obs[:, FOES:DISCS].reshape(-1, 3, 36)
    near_boss_hazard = (
        (bosses[:, :, 0] > 0.5)
        & (np.abs(bosses[:, :, 1]) <= 0.40)
        & (bosses[:, :, 20:26].max(axis=2) > 0.5)
    ).any(1)
    wall_prep = (
        (bosses[:, :, 0] > 0.5)
        & ((bosses[:, :, 27:30].max(axis=2) > 0.5) | (bosses[:, :, 32:36].max(axis=2) > 0.5))
    ).any(1)
    airborne = obs[:, 7] < 0.5
    in_threat = before_body | before_proj | before_spike | danger_edge | near_boss_hazard | disc_present

    dash_onset = ((prior_btns & 4) == 0) & ((tgt_cmd[:, 2] & 4) != 0)
    cast_onset = ((prior_btns & 16) == 0) & ((tgt_cmd[:, 2] & 16) != 0)
    jtap_onset = ((prior_btns & 1) == 0) & ((tgt_cmd[:, 2] & 1) != 0)
    shadow_ready = obs[:, 175] >= 1.45

    # Moderate calibrated onset boosts to keep dash_ratio cleanly inside [1.15, 1.50]
    w[valid & dash_onset & in_threat & shadow_ready] *= 1.30
    w[valid & dash_onset & in_threat & (~shadow_ready)] *= 1.10
    w[valid & cast_onset & in_threat] *= 1.22
    w[valid & jtap_onset & (disc_present | before_proj)] *= 1.25
    w[valid & jtap_onset & ~(dash_onset | cast_onset) & (~disc_present) & (~wall_prep)] *= 1.05

    if anti_disc_v2:
        # Gen-2 Anti-Disc: explicitly sharpen disc-dodge decisions on BOTH clean teacher & DAgger
        w[valid & trio & disc_present] *= (1.0 + 0.22 * disc_w_scale)
        w[valid & disc_close] *= (1.0 + 0.48 * disc_w_scale)
        w[valid & disc_close & dash_onset & shadow_ready] *= 1.32
        w[valid & disc_close & jtap_onset] *= 1.20
        # Penalize/sharpen moments when hero is airborne near disc (prevents 10845201 jump-into-disc!)
        w[valid & disc_close & airborne] *= 1.18

    if is_corrective:
        proj_bonus = (2.15 * disc_w_scale) if anti_disc_v2 else 1.60
        corr_mult = (
            1.0
            + 0.45 * near_edge.astype("f4")
            + 0.90 * danger_edge.astype("f4")
            + 1.40 * before_body.astype("f4")
            + proj_bonus * before_proj.astype("f4")
            + 1.20 * before_spike.astype("f4")
        )
        disagree = (tgt_cmd != student_actions).any(1)
        corr_mult = np.where(disagree & in_threat, corr_mult * 1.18, corr_mult)

        airborne_disc_err = (disc_present | before_proj) & airborne & disagree
        missed_disc_jump = (disc_present | before_proj) & (((tgt_cmd[:, 2] & 3) != 0) & ((student_actions[:, 2] & 3) == 0))
        jump_into_disc = (disc_present | wall_prep | before_proj) & ((student_actions[:, 2] & 3) != 0) & ((tgt_cmd[:, 2] & 3) == 0)
        late_bad_cast_or_atk = in_threat & (((student_actions[:, 2] & 24) != 0) & ((tgt_cmd[:, 2] & 24) == 0))
        bad_focus_in_threat = ((student_actions[:, 2] & 32) != 0) & ((tgt_cmd[:, 2] & 32) == 0)
        missed_dash = in_threat & (((tgt_cmd[:, 2] & 4) != 0) & ((student_actions[:, 2] & 4) == 0))
        missed_cast = in_threat & (((tgt_cmd[:, 2] & 16) != 0) & ((student_actions[:, 2] & 16) == 0))

        corr_mult = np.where(missed_dash | missed_cast, corr_mult * 1.18, corr_mult)
        corr_mult = np.where(late_bad_cast_or_atk | bad_focus_in_threat, corr_mult * 1.15, corr_mult)
        corr_mult = np.where(jump_into_disc | missed_disc_jump | airborne_disc_err, corr_mult * (1.30 if anti_disc_v2 else 1.22), corr_mult)
        if is_strong_dagger:
            corr_mult *= 1.08
        if anti_disc_v2:
            cap = np.where(before_proj | disc_close, 4.4, 3.5).astype("f4")
            corr_mult = np.minimum(corr_mult, cap)
        else:
            corr_mult = np.minimum(corr_mult, 3.5)
        w[valid] *= corr_mult[valid]

        body_anchor = body_anchor | (valid & (missed_dash | missed_cast | late_bad_cast_or_atk | bad_focus_in_threat))
        proj_anchor = proj_anchor | (valid & (jump_into_disc | missed_disc_jump | airborne_disc_err | disc_close))
    else:
        # Pure teacher trajectory: never reinforce frames immediately preceding a teacher hit
        teacher_mistake = valid & pre_window(clock, events[:, 1:5].sum(1) > 0, 0.72)
        w[teacher_mistake] = 0.0
        body_anchor = body_anchor & (~teacher_mistake)
        proj_anchor = proj_anchor & (~teacher_mistake)

    return w.astype("f4"), body_anchor, proj_anchor, teacher_disc_anchor


@dataclass
class WP22FightEntry:
    rf: WP18Fight
    body_anchor: np.ndarray
    proj_anchor: np.ndarray
    teacher_disc_anchor: np.ndarray


def build_wp22_fight_pair(
    ep: Episode,
    lookup: dict[tuple[int, ...], int],
    sup_lbl: np.ndarray | None,
    mirror_action_map: np.ndarray,
    cat_np: np.ndarray,
    rare_np: np.ndarray,
    is_corrective: bool,
    is_strong_dagger: bool,
    kind_tag: str,
    include_mirror: bool,
    hist_fn=motion_history,
    anti_disc_v2: bool = False,
    disc_w_scale: float = 1.0,
) -> list[WP22FightEntry]:
    f_orig = prepare_fight(ep, lookup)
    lbl_orig = np.full(len(ep.clock), -1, dtype="i8") if sup_lbl is None else sup_lbl.copy()

    if sup_lbl is not None:
        # Mask redundant high-HP focus onset labels (>6.5 masks)
        focus_lbl = (lbl_orig >= 0) & ((cat_np[lbl_orig.clip(0), 2] & 32) != 0)
        high_hp_onset = (ep.observation[:, 169] < 0.5) & (ep.observation[:, 4] > (6.5 / 9.0))
        lbl_orig = np.where(focus_lbl & high_hp_onset, -1, lbl_orig).astype("i8")

    # Near-No-Hit quality multiplier on clean 99% teacher fights (pushes toward zero-hit mastery!)
    fight_quality_mult = 1.0
    if anti_disc_v2 and kind_tag == "teacher_new" and f_orig.won:
        total_hits = int(ep.events[:, 1:5].sum())
        if total_hits == 0:
            fight_quality_mult = 1.38
        elif total_hits <= 2:
            fight_quality_mult = 1.18
        elif total_hits >= 5:
            fight_quality_mult = 0.85

    occ_raw = future_occupancy(ep.clock, ep.observation)
    occ_val = ~np.isnan(occ_raw[:, 0, 0, 0, 0])
    occ_clean = np.nan_to_num(occ_raw, nan=0.0)
    occ_flat_orig = occ_clean.reshape(len(ep.clock), 54).astype("f2")

    if sup_lbl is not None and (lbl_orig >= 0).any():
        t_cmds = np.where((lbl_orig >= 0)[:, None], cat_np[lbl_orig.clip(0)], ep.action)
        fmov_orig = future_movements(ep.clock, ep.action, t_cmds, ep.observation[:, 23] > 0.5)
        fmov_orig[lbl_orig < 0] = -1
    else:
        fmov_orig = np.full((len(ep.clock), 3), -1, dtype="i1")

    hx_orig = hist_fn(ep.clock, ep.observation)
    if sup_lbl is not None:
        sw_orig, ba_orig, pa_orig, tda_orig = compute_wp22_weights(
            ep.observation, ep.clock, ep.events, ep.action, lbl_orig, cat_np, rare_np,
            is_corrective, is_strong_dagger, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
        )
        sw_orig = (sw_orig * fight_quality_mult).astype("f4")
    else:
        sw_orig = np.zeros(len(ep.clock), dtype="f4")
        ba_orig = np.zeros(len(ep.clock), dtype=bool)
        pa_orig = np.zeros(len(ep.clock), dtype=bool)
        tda_orig = np.zeros(len(ep.clock), dtype=bool)
    hz_orig = build_hazard_mask(ep.observation, ep.clock, ep.events)
    rf_orig = WP18Fight(f_orig, hx_orig, lbl_orig, sw_orig, ba_orig | pa_orig, hz_orig, occ_flat_orig, occ_val, fmov_orig, kind_tag)
    out = [WP22FightEntry(rf_orig, ba_orig, pa_orig, tda_orig)]

    if include_mirror:
        m_obs = mirror_observation_192(ep.observation)
        m_act = ep.action.copy()
        m_act[:, 0] *= -1
        m_ep = Episode(m_obs, m_act, ep.reward, ep.events, ep.clock, ep.next_row, ep.terminated, ep.truncated, ep.controller)
        m_fight = prepare_fight(m_ep, lookup)
        m_lbl = (
            np.full(len(ep.clock), -1, dtype="i8")
            if sup_lbl is None
            else np.where(lbl_orig >= 0, mirror_action_map[lbl_orig.clip(0)], -1).astype("i8")
        )
        m_hx = hist_fn(m_ep.clock, m_obs)
        if sup_lbl is not None:
            m_sw, m_ba, m_pa, m_tda = compute_wp22_weights(
                m_obs, m_ep.clock, m_ep.events, m_act, m_lbl, cat_np, rare_np,
                is_corrective, is_strong_dagger, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
            )
            m_sw = (m_sw * fight_quality_mult).astype("f4")
        else:
            m_sw = np.zeros(len(ep.clock), dtype="f4")
            m_ba = np.zeros(len(ep.clock), dtype=bool)
            m_pa = np.zeros(len(ep.clock), dtype=bool)
            m_tda = np.zeros(len(ep.clock), dtype=bool)
        m_hz = build_hazard_mask(m_obs, m_ep.clock, m_ep.events)
        m_occ_flat = occ_clean[..., ::-1].reshape(len(ep.clock), 54).astype("f2")
        m_fmov = np.where(fmov_orig >= 0, (2 - fmov_orig // 4) * 4 + (fmov_orig % 4), -1).astype("i1")
        rf_m = WP18Fight(m_fight, m_hx, m_lbl, m_sw, m_ba | m_pa, m_hz, m_occ_flat, occ_val, m_fmov, kind_tag)
        out.append(WP22FightEntry(rf_m, m_ba, m_pa, m_tda))
    return out


class WP22Sampler:
    """Balanced WP22-Gold+ / Gen-2 Anti-Disc sampler."""

    def __init__(
        self,
        entries: list[WP22FightEntry],
        rng: np.random.Generator,
        proj_focus: bool = False,
        anti_disc_v2: bool = False,
        pure_clean: bool = False,
    ) -> None:
        self.rng = rng
        self.rl_pools: dict[tuple[str, int, bool], list[tuple[WP22FightEntry, int, int]]] = {}
        self.sup_pools: dict[tuple[str, int], list[tuple[WP22FightEntry, int, int]]] = {}

        for burn in (0, BURN_IN):
            for rl_group in ("teacher_new", "strong_dagger", "background"):
                for hazard_only in (False, True):
                    pool = []
                    for ent in entries:
                        rf = ent.rf
                        match = (
                            (rl_group == "teacher_new" and rf.kind == "teacher_new")
                            or (rl_group == "strong_dagger" and rf.kind == "strong_dagger")
                            or (rl_group == "background" and rf.kind in ("legacy_bg", "student_rl"))
                        )
                        if not match:
                            continue
                        elig = rf.fight.train_rows & (rf.hazard_anchor if hazard_only else True)
                        for lo, hi in rf.fight.segments:
                            sl = elig[lo + burn : hi] if burn else elig[lo : min(hi, lo + UNROLL)]
                            if sl.any():
                                pool.append((ent, lo, hi))
                    if pool:
                        self.rl_pools[rl_group, burn, hazard_only] = pool

            for sup_group in ("teacher_new", "teacher_disc_anchor", "strong_dagger", "proj_anchor", "body_anchor", "legacy_bg"):
                pool = []
                for ent in entries:
                    rf = ent.rf
                    if sup_group == "teacher_new":
                        ok = rf.kind == "teacher_new" and rf.fight.won
                        elig = rf.sup_label >= 0
                    elif sup_group == "teacher_disc_anchor":
                        ok = rf.kind == "teacher_new" and rf.fight.won
                        elig = ent.teacher_disc_anchor
                    elif sup_group == "strong_dagger":
                        ok = rf.kind == "strong_dagger"
                        elig = rf.sup_label >= 0
                    elif sup_group == "proj_anchor":
                        ok = rf.kind == "strong_dagger"
                        elig = ent.proj_anchor
                    elif sup_group == "body_anchor":
                        ok = rf.kind == "strong_dagger"
                        elig = ent.body_anchor
                    else:
                        ok = rf.kind == "legacy_bg"
                        elig = rf.sup_label >= 0
                    if not ok:
                        continue
                    for lo, hi in rf.fight.segments:
                        sl = elig[lo + burn : hi] if burn else elig[lo : min(hi, lo + UNROLL)]
                        if sl.any():
                            pool.append((ent, lo, hi))
                if pool:
                    self.sup_pools[sup_group, burn] = pool

        if anti_disc_v2 and pure_clean:
            # 0% Legacy 77% contamination! 100% pure 99% teacher + strong DAgger + 24% dedicated disc anchors
            self.sup_schedule = (
                ["teacher_new"] * 21
                + ["teacher_disc_anchor"] * 7
                + ["strong_dagger"] * 15
                + ["proj_anchor"] * 5
                + ["body_anchor"] * 2
            )
        elif anti_disc_v2:
            # Gen-2 Anti-Disc: 22% dedicated disc anchors (12% clean teacher + 10% strong DAgger)
            self.sup_schedule = (
                ["teacher_new"] * 20
                + ["teacher_disc_anchor"] * 6
                + ["strong_dagger"] * 14
                + ["proj_anchor"] * 5
                + ["body_anchor"] * 2
                + ["legacy_bg"] * 3
            )
        elif proj_focus:
            self.sup_schedule = (
                ["teacher_new"] * 24
                + ["strong_dagger"] * 16
                + ["proj_anchor"] * 4
                + ["body_anchor"] * 2
                + ["legacy_bg"] * 4
            )
        else:
            self.sup_schedule = (
                ["teacher_new"] * 25
                + ["strong_dagger"] * 16
                + ["proj_anchor"] * 3
                + ["body_anchor"] * 2
                + ["legacy_bg"] * 4
            )
        assert len(self.sup_schedule) == 50

    def _slice(self, rf: WP18Fight, lo: int, hi: int, burn: int, anchor_mask: np.ndarray) -> tuple[np.ndarray, ...]:
        f = rf.fight
        if burn:
            valid = anchor_mask[lo + burn : hi]
            anchor = lo + burn + int(self.rng.choice(np.flatnonzero(valid)))
            start = max(lo, anchor - burn - int(self.rng.integers(UNROLL)))
        else:
            start = lo
        indices = np.arange(start, start + burn + UNROLL + 1)
        clipped = np.minimum(indices, hi - 1)
        scored = clipped[burn:-1]
        in_bounds = indices[burn:-1] < hi
        rl_mask = in_bounds & f.train_rows[scored]
        sup_mask = in_bounds & (rf.sup_label[scored] >= 0)
        occ_mask = rl_mask & rf.occ_valid[scored]
        return (
            rf.history_x[clipped],
            f.action[scored].clip(0).astype("i8"),
            f.episode.reward[scored].astype("f4"),
            f.discount[scored].astype("f4"),
            rl_mask,
            rf.occ_target[scored].astype("f4"),
            occ_mask,
            rf.sup_label[scored].clip(0).astype("i8"),
            (rf.sup_weight[scored] * sup_mask.astype("f4")).astype("f4"),
            sup_mask,
            rf.fmov_target[scored].astype("i8"),
            f.episode.events[scored, 0].astype("f4"),
            (f.episode.events[scored, 1:5].sum(1) > 0).astype("f4"),
        )

    def sample_rl(self, batch_size: int, burn: int) -> tuple[Tensor, ...]:
        rows = []
        rl_sched = ["teacher_new"] * 3 + ["strong_dagger"] * 3 + ["background"] * 2
        for i in range(batch_size):
            grp = rl_sched[i % len(rl_sched)]
            if (grp, burn, False) not in self.rl_pools:
                grp = "teacher_new"
            hazard = (i % 2 == 1) and ((grp, burn, True) in self.rl_pools)
            pool = self.rl_pools[grp, burn, hazard]
            ent, lo, hi = pool[int(self.rng.integers(len(pool)))]
            rf = ent.rf
            anchor = rf.fight.train_rows & (rf.hazard_anchor if hazard else True)
            w = self._slice(rf, lo, hi, burn, anchor)
            rows.append((w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[11], w[12]))
        return tuple(torch.from_numpy(np.stack(col)) for col in zip(*rows, strict=True))

    def sample_sup(self, batch_size: int, burn: int) -> tuple[Tensor, ...]:
        rows = []
        offset = int(self.rng.integers(len(self.sup_schedule)))
        for i in range(batch_size):
            grp = self.sup_schedule[(offset + i) % len(self.sup_schedule)]
            if (grp, burn) not in self.sup_pools:
                grp = "strong_dagger" if ("strong_dagger", burn) in self.sup_pools else "teacher_new"
            pool = self.sup_pools[grp, burn]
            ent, lo, hi = pool[int(self.rng.integers(len(pool)))]
            if grp == "proj_anchor":
                anchor = ent.proj_anchor
            elif grp == "body_anchor":
                anchor = ent.body_anchor
            elif grp == "teacher_disc_anchor":
                anchor = ent.teacher_disc_anchor
            else:
                anchor = ent.rf.sup_label >= 0
            w = self._slice(ent.rf, lo, hi, burn, anchor)
            rows.append((w[0], w[7], w[8], w[9], w[5], w[6], w[10]))
        return tuple(torch.from_numpy(np.stack(col)) for col in zip(*rows, strict=True))


@torch.inference_mode()
def evaluate_wp22(
    actor: nn.Module,
    occ_head: nn.Linear,
    teacher_holdout: dict[str, Fight],
    student_holdout: list[WP18Fight],
    cat_np: np.ndarray,
    device: torch.device,
    hist_fn=motion_history,
) -> dict[str, Any]:
    actor.eval()
    dash_bit = 1 << BUTTON_BITS["dash"]
    cast_bit = 1 << BUTTON_BITS["cast"]
    jtap_bit = 1 << BUTTON_BITS["jump_tap"]
    focus_bit = 1 << BUTTON_BITS["focus_hold"]
    key_specs = [("dash", dash_bit, 166), ("cast", cast_bit, 168), ("jump_tap", jtap_bit, 164), ("focus_hold", focus_bit, 169)]
    onset_stats = {k: {"labeled": 0, "pred": 0, "match": 0} for k, _, _ in key_specs}
    t_tot = t_match = 0
    t_nll = 0.0

    for _, fight in teacher_holdout.items():
        if not (fight.pure_teacher and fight.won):
            continue
        ep = fight.episode
        for lo, hi in fight.segments:
            rows = np.arange(lo, hi)
            valid = fight.train_rows[rows] & fight.teacher[rows]
            if not valid.any():
                continue
            logits = actor(torch.from_numpy(hist_fn(ep.clock, ep.observation, rows)).to(device)[None])[0].cpu()
            lp = logits.log_softmax(-1)
            tgt = fight.action[rows[valid]]
            pred = logits[valid].argmax(-1).numpy()
            t_tot += len(tgt)
            t_match += int((pred == tgt).sum())
            t_nll += float(-lp[valid, tgt].sum())
            obs_v = ep.observation[rows[valid]]
            for kname, bit, pcol in key_specs:
                p0 = obs_v[:, pcol] < 0.5
                t_on = ((cat_np[tgt, 2] & bit) != 0) & p0
                p_on = ((cat_np[pred, 2] & bit) != 0) & p0
                onset_stats[kname]["labeled"] += int(t_on.sum())
                onset_stats[kname]["pred"] += int(p_on.sum())
                onset_stats[kname]["match"] += int((t_on & p_on).sum())

    for s in onset_stats.values():
        rec = s["match"] / max(1, s["labeled"])
        prec = s["match"] / max(1, s["pred"])
        s.update(recall=rec, precision=prec, f1=2 * rec * prec / max(1e-8, rec + prec))

    buckets = {
        k: {"rows": 0, "correct": 0, "nll_sum": 0.0, "outward": 0, "invalid_onsets": 0}
        for k in ("all", "danger", "before_body", "before_proj")
    }
    occ_mae_sum = 0.0
    occ_cnt = 0
    for rf in student_holdout:
        ep = rf.fight.episode
        clock, obs = ep.clock, ep.observation
        x_w = obs[:, 0] * 10.0
        low = (obs[:, 1] * 8.0 + 7.41) < 8.6
        danger = (np.abs(x_w) > 8.35) & low
        b_body = pre_window(clock, ep.events[:, 1] > 0, 0.40)
        b_proj = pre_window(clock, ep.events[:, 2] > 0, 0.44)
        for lo, hi in rf.fight.segments:
            tgt = rf.sup_label[lo:hi]
            val = tgt >= 0
            if not val.any():
                continue
            x_seg = torch.from_numpy(rf.history_x[lo:hi]).to(device)[None]
            logits, _, emb, _ = actor.forward_full(x_seg, 0)
            logits = logits[0].cpu()
            ov = rf.occ_valid[lo:hi]
            if ov.any():
                opred = torch.sigmoid(occ_head(emb[0, ov])).cpu().numpy()
                occ_mae_sum += float(np.abs(opred - rf.occ_target[lo:hi][ov].astype("f4")).mean() * ov.sum())
                occ_cnt += int(ov.sum())
            pred = logits.argmax(-1).numpy()
            ce = F.cross_entropy(logits, torch.from_numpy(tgt).long(), ignore_index=-1, reduction="none").numpy()
            ob_s = obs[lo:hi]
            side = np.sign(ob_s[:, 0])
            p_out = (side != 0) & (cat_np[pred, 0] == side)
            p_btns = cat_np[pred, 2]
            inv_c = ((p_btns & 16) != 0) & (ob_s[:, 168] < 0.5) & ((ob_s[:, 11] < 0.5) | (ob_s[:, 5] < 33.0 / 99.0 - 1e-3))
            inv_d = ((p_btns & 4) != 0) & (ob_s[:, 166] < 0.5) & (ob_s[:, 13] < 0.5)
            inv_j = ((p_btns & 1) != 0) & (ob_s[:, 164] < 0.5) & (ob_s[:, 7] < 0.5) & (ob_s[:, 8] < 0.5)
            inv_f = ((p_btns & 32) != 0) & (ob_s[:, 169] < 0.5) & ((ob_s[:, 12] < 0.5) | (ob_s[:, 5] < 33.0 / 99.0 - 1e-3) | (ob_s[:, 7] < 0.5))
            inv_any = inv_c | inv_d | inv_j | inv_f
            for bname, m in (
                ("all", val),
                ("danger", val & danger[lo:hi]),
                ("before_body", val & b_body[lo:hi]),
                ("before_proj", val & b_proj[lo:hi]),
            ):
                b = buckets[bname]
                b["rows"] += int(m.sum())
                b["correct"] += int(((pred == tgt) & m).sum())
                b["nll_sum"] += float(ce[m].sum())
                b["outward"] += int((p_out & m).sum())
                b["invalid_onsets"] += int((inv_any & m).sum())
    for b in buckets.values():
        if b["rows"]:
            b["top1"] = b["correct"] / b["rows"]
            b["nll"] = b["nll_sum"] / b["rows"]
            b["outward_rate"] = b["outward"] / b["rows"]

    t_nll_mean = t_nll / max(1, t_tot)
    s_all = buckets["all"].get("nll", 2.0)
    s_proj = buckets["before_proj"].get("nll", s_all)
    s_body = buckets["before_body"].get("nll", s_all)
    s_danger = buckets["danger"].get("nll", 1.0)
    crit_f1 = 0.45 * onset_stats["dash"]["f1"] + 0.35 * onset_stats["cast"]["f1"] + 0.20 * onset_stats["jump_tap"]["f1"]
    dash_ratio = max(1, onset_stats["dash"]["pred"]) / max(1, onset_stats["dash"]["labeled"])
    dash_calib_pen = 0.16 * max(0.0, abs(math.log(dash_ratio / 1.40)) - 0.25)

    composite = (
        0.45 * t_nll_mean
        + 0.45 * s_all
        + 0.40 * s_proj
        + 0.40 * s_body
        + 0.20 * s_danger
        - 0.65 * crit_f1
        + dash_calib_pen
    )
    return {
        "teacher_win_top1": t_match / max(1, t_tot),
        "teacher_win_nll": t_nll_mean,
        "dash_ratio": dash_ratio,
        "occ_mae": occ_mae_sum / max(1, occ_cnt),
        "onset_keys": onset_stats,
        "student_holdout": buckets,
        "critical_onset_f1": crit_f1,
        "composite_loss": composite,
    }


@torch.inference_mode()
def build_wp22_world_game(actor: nn.Module, fight: Fight, stride: int, device: torch.device, hist_fn=motion_history) -> tuple[Tensor, ...] | None:
    tape = fight.episode
    blocks = []
    indices = np.asarray(HORIZONS)
    for lo, hi in fight.segments:
        if hi - lo <= STEPS:
            continue
        rows = np.arange(lo, hi)
        exact = (np.abs(np.diff(tape.clock[rows]) - 0.04) < 1e-4) & (fight.action[rows[:-1]] >= 0)
        broken = np.r_[0, np.cumsum(~exact)]
        starts = np.arange(0, len(rows) - STEPS)
        starts = starts[broken[starts] == broken[starts + STEPS]][::stride]
        if not len(starts):
            continue
        window = starts[:, None] + np.arange(STEPS)
        future = starts[:, None] + indices
        x = torch.from_numpy(hist_fn(tape.clock, tape.observation, rows)).to(device)[None]
        spatial, emb, _ = actor.backbone.forward_sequence(x, 0)
        latent = torch.cat((spatial[0, starts], emb[0, starts]), dim=-1)
        hero = tape.observation[rows, :6]
        events = np.concatenate((tape.events[rows, :1], tape.events[rows, 1:5].sum(1, keepdims=True)), axis=1)
        prefix = np.concatenate((np.zeros((1, 2), dtype="f4"), np.cumsum(events, axis=0)))
        scores = prefix[future] - prefix[starts, None]
        physical = hero[future, :4] - hero[starts, None, :4]
        blocks.append((
            latent,
            torch.from_numpy(hero[starts]).to(device),
            torch.from_numpy(fight.action[rows[window]]).to(device),
            torch.from_numpy(physical).to(device),
            torch.from_numpy(scores).to(device),
        ))
    if not blocks:
        return None
    return tuple(torch.cat(x) for x in zip(*blocks, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opener-pt", type=Path, default=None)
    ap.add_argument("--opening-parent", type=Path, default=None)
    ap.add_argument("--teacher-archive", type=Path, required=True)
    ap.add_argument("--teacher-audit", type=Path, required=True)
    ap.add_argument("--ppo-audit", type=Path, required=True)
    ap.add_argument("--wp22-teacher-npz", type=Path, default=None)
    ap.add_argument("--wp22-teacher-meta", type=Path, default=None)
    ap.add_argument("--wp22-dagger-npz", type=Path, default=None)
    ap.add_argument("--wp22-dagger-meta", type=Path, default=None)
    ap.add_argument("--wp19-corr-npz", type=Path, required=True)
    ap.add_argument("--wp19-corr-meta", type=Path, required=True)
    ap.add_argument("--wp20-corr-npz", type=Path, required=True)
    ap.add_argument("--wp20-corr-meta", type=Path, required=True)
    ap.add_argument("--wp6-npz", type=Path, required=True)
    ap.add_argument("--wp6-meta", type=Path, required=True)
    ap.add_argument("--wp10-npz", type=Path, required=True)
    ap.add_argument("--wp10-meta", type=Path, required=True)
    ap.add_argument("--wp11-npz", type=Path, required=True)
    ap.add_argument("--wp11-meta", type=Path, required=True)
    ap.add_argument("--wp15-npz", type=Path, required=True)
    ap.add_argument("--wp15-meta", type=Path, required=True)
    ap.add_argument("--wp17-npz", type=Path, required=True)
    ap.add_argument("--wp17-meta", type=Path, required=True)
    ap.add_argument("--rl-npz", type=Path, required=True)
    ap.add_argument("--rl-meta", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--critic-warmup", type=int, default=1500)
    ap.add_argument("--joint-steps", type=int, default=10000)
    ap.add_argument("--wm-updates", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--val-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20262201)
    ap.add_argument(
        "--variant",
        default="a_141m",
        choices=(
            "a_141m", "b_215m", "c_290m", "d_215m_proj",
            "e_attn_256", "f_attn_320", "g_attn_320_2lgru",
            "h_disc_320_2lgru", "i_disc_384_2lgru", "j_disc_320_1lgru_hi",
            "k_disc_384_8h_1lgru", "l_disc_320_2lgru_pure", "m_disc_384_2lgru_pure",
        ),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    catalogue = native_catalogue(args.parent)
    cat_np = catalogue.numpy()
    lookup = {tuple(a): i for i, a in enumerate(cat_np.tolist())}
    mirror_action_map = np.array([lookup.get((-int(m), int(a), int(b)), -1) for m, a, b in cat_np], dtype="i8")

    # WP22-Gold+ 6 parallel candidate architectures (4 V1 0x3E4C5149 + 2 V2 0x3F4C5149 Entity-Attention+4-Lag)
    VARIANT_SPECS = {
        "a_141m": dict(
            v2=False, proj_focus=False, awr_w=0.24, disc_jump_w=0.32,
            dims=dict(d_boss=96, d_disc=80, d_past=64, d_model=256, d_ffn=512, n_blocks=3, n_gru_layers=1),
            actor_drop=0.12, critic_drop=0.10, actor_wd=1.5e-3, critic_wd=1.0e-3,
        ),
        "b_215m": dict(
            v2=False, proj_focus=False, awr_w=0.24, disc_jump_w=0.32,
            dims=dict(d_boss=96, d_disc=96, d_past=64, d_model=320, d_ffn=640, n_blocks=3, n_gru_layers=1),
            actor_drop=0.14, critic_drop=0.10, actor_wd=1.5e-3, critic_wd=1.0e-3,
        ),
        "c_290m": dict(
            v2=False, proj_focus=False, awr_w=0.24, disc_jump_w=0.32,
            dims=dict(d_boss=96, d_disc=80, d_past=64, d_model=384, d_ffn=512, n_blocks=2, n_gru_layers=2),
            actor_drop=0.15, critic_drop=0.12, actor_wd=2.0e-3, critic_wd=1.2e-3,
        ),
        "d_215m_proj": dict(
            v2=False, proj_focus=True, awr_w=0.30, disc_jump_w=0.42,
            dims=dict(d_boss=96, d_disc=112, d_past=64, d_model=320, d_ffn=640, n_blocks=3, n_gru_layers=1),
            actor_drop=0.14, critic_drop=0.10, actor_wd=1.5e-3, critic_wd=1.0e-3,
        ),
        "e_attn_256": dict(
            v2=True, proj_focus=True, awr_w=0.26, disc_jump_w=0.36,
            dims=dict(d_boss=96, d_disc=80, d_past=48, d_tok=96, n_heads=4, d_model=256, d_ffn=512, n_blocks=3, n_gru_layers=1),
            actor_drop=0.12, critic_drop=0.10, actor_wd=1.5e-3, critic_wd=1.0e-3,
        ),
        "f_attn_320": dict(
            v2=True, proj_focus=True, awr_w=0.26, disc_jump_w=0.36,
            dims=dict(d_boss=96, d_disc=96, d_past=64, d_tok=128, n_heads=4, d_model=320, d_ffn=640, n_blocks=3, n_gru_layers=1),
            actor_drop=0.14, critic_drop=0.10, actor_wd=1.5e-3, critic_wd=1.0e-3,
        ),
        "g_attn_320_2lgru": dict(
            v2=True, proj_focus=True, awr_w=0.28, disc_jump_w=0.38,
            dims=dict(d_boss=96, d_disc=96, d_past=64, d_tok=128, n_heads=4, d_model=320, d_ffn=640, n_blocks=3, n_gru_layers=2),
            actor_drop=0.15, critic_drop=0.12, actor_wd=2.0e-3, critic_wd=1.2e-3,
        ),
        # === WP22-Gen2 Anti-Disc & Near-No-Hit V2 Attention Fleet ===
        "h_disc_320_2lgru": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=False, disc_w_scale=1.15,
            awr_w=0.28, disc_jump_w=0.44, disc_quake_w=0.45, air_disc_jump_w=0.36,
            dims=dict(d_boss=96, d_disc=112, d_past=64, d_tok=128, n_heads=4, d_model=320, d_ffn=640, n_blocks=3, n_gru_layers=2),
            actor_drop=0.10, critic_drop=0.10, actor_wd=1.2e-3, critic_wd=1.0e-3,
        ),
        "i_disc_384_2lgru": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=False, disc_w_scale=1.15,
            awr_w=0.28, disc_jump_w=0.44, disc_quake_w=0.45, air_disc_jump_w=0.36,
            dims=dict(d_boss=112, d_disc=112, d_past=64, d_tok=160, n_heads=4, d_model=384, d_ffn=768, n_blocks=4, n_gru_layers=2),
            actor_drop=0.12, critic_drop=0.10, actor_wd=1.4e-3, critic_wd=1.0e-3,
        ),
        "j_disc_320_1lgru_hi": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=False, disc_w_scale=1.35,
            awr_w=0.26, disc_jump_w=0.50, disc_quake_w=0.52, air_disc_jump_w=0.42,
            dims=dict(d_boss=96, d_disc=128, d_past=64, d_tok=128, n_heads=4, d_model=320, d_ffn=640, n_blocks=4, n_gru_layers=1),
            actor_drop=0.10, critic_drop=0.10, actor_wd=1.2e-3, critic_wd=1.0e-3,
        ),
        "k_disc_384_8h_1lgru": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=False, disc_w_scale=1.25,
            awr_w=0.26, disc_jump_w=0.46, disc_quake_w=0.48, air_disc_jump_w=0.38,
            dims=dict(d_boss=112, d_disc=128, d_past=64, d_tok=160, n_heads=8, d_model=384, d_ffn=768, n_blocks=4, n_gru_layers=1),
            actor_drop=0.11, critic_drop=0.10, actor_wd=1.3e-3, critic_wd=1.0e-3,
        ),
        "l_disc_320_2lgru_pure": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=True, disc_w_scale=1.25,
            awr_w=0.30, disc_jump_w=0.48, disc_quake_w=0.50, air_disc_jump_w=0.40,
            dims=dict(d_boss=96, d_disc=128, d_past=64, d_tok=128, n_heads=4, d_model=320, d_ffn=640, n_blocks=4, n_gru_layers=2),
            actor_drop=0.10, critic_drop=0.10, actor_wd=1.2e-3, critic_wd=1.0e-3,
        ),
        "m_disc_384_2lgru_pure": dict(
            v2=True, proj_focus=True, anti_disc_v2=True, pure_clean=True, disc_w_scale=1.25,
            awr_w=0.30, disc_jump_w=0.48, disc_quake_w=0.50, air_disc_jump_w=0.40,
            dims=dict(d_boss=112, d_disc=128, d_past=64, d_tok=160, n_heads=8, d_model=384, d_ffn=768, n_blocks=4, n_gru_layers=2),
            actor_drop=0.12, critic_drop=0.10, actor_wd=1.4e-3, critic_wd=1.0e-3,
        ),
    }
    vspec = VARIANT_SPECS[args.variant]
    is_v2 = bool(vspec["v2"])
    anti_disc_v2 = bool(vspec.get("anti_disc_v2", False))
    pure_clean = bool(vspec.get("pure_clean", False))
    disc_w_scale = float(vspec.get("disc_w_scale", 1.0))
    hist_fn = motion_history_4lag if is_v2 else motion_history
    base_dims = vspec["dims"]
    actor_kwargs = {**base_dims, "dropout": vspec["actor_drop"]}
    critic_kwargs = {**base_dims, "dropout": vspec["critic_drop"]}
    ActorCls = AttnRecurrentActor if is_v2 else ScaledRecurrentActor
    CriticCls = AttnRecurrentValues if is_v2 else ScaledRecurrentValues
    actor = ActorCls(catalogue, **actor_kwargs).to(device)
    ema_actor = deepcopy(actor).eval().requires_grad_(False)
    values = CriticCls(catalogue, **critic_kwargs).to(device)
    target_values = deepcopy(values).eval().requires_grad_(False)

    d_model = actor.cfg["d_model"]
    occ_head = nn.Linear(d_model, 54).to(device)
    fmov_head = nn.Linear(d_model, 3 * 12).to(device)

    # Dynamic World(2 * d_model, 75) = World(512, 75)
    worlds = [World(2 * d_model, len(cat_np)).to(device) for _ in range(3)]
    world_opt = torch.optim.AdamW([p for wm in worlds for p in wm.parameters()], lr=1.0e-3, weight_decay=1e-4)

    n_actor_params = sum(p.numel() for p in actor.parameters())
    n_critic_params = sum(p.numel() for p in values.parameters())
    print(
        f"WP22-Gold+ Actor={n_actor_params:,} Critic={n_critic_params:,} WorldIn={2 * d_model} "
        f"cfg={actor.cfg} bf16={use_bf16}",
        flush=True,
    )

    # Clean de-duplicated teacher split using training-100 provenance
    t100_audit = json.loads(args.ppo_audit.read_bytes())
    t100_sources = {e["provenance"]["source"]: e["provenance"]["split"] for e in t100_audit["episodes"]}
    teacher_groups = read_fights(args.teacher_archive, args.teacher_audit, catalogue)

    clean_t_train: dict[str, Fight] = {}
    legacy_t_train: dict[str, Fight] = {}
    clean_t_holdout: dict[str, Fight] = {}
    for src, f in teacher_groups["train"].items():
        if src in t100_sources:
            assert t100_sources[src] == "train", f"Split leak on {src}"
            clean_t_train[src] = f
        else:
            legacy_t_train[src] = f
    for src, f in teacher_groups["holdout"].items():
        if src in t100_sources:
            assert t100_sources[src] == "holdout", f"Split leak on {src}"
            clean_t_holdout[src] = f

    sup_acts = [f.action[f.train_rows & f.teacher] for f in clean_t_train.values() if f.pure_teacher and f.won]
    all_sup_act = np.concatenate(sup_acts)
    counts_act = np.bincount(all_sup_act, minlength=len(cat_np)).astype("f4").clip(1.0)
    rare_np = np.minimum((counts_act.max() / counts_act) ** 0.08, 1.35).astype("f4")

    train_entries: list[WP22FightEntry] = []
    strong_holdout: list[WP18Fight] = []
    stats_counts = {
        "clean_t100_train_wins": 0,
        "clean_t100_holdout": len(clean_t_holdout),
        "track_a_teacher_wins": 0,
        "legacy_teacher_train": 0,
        "strong_dagger_train": 0,
        "strong_dagger_train_losses": 0,
        "strong_dagger_holdout": 0,
        "legacy_weak_corr_train": 0,
        "student_rl": 0,
    }

    # 1. Clean 99% teacher from training-100 (via teacher-144, zero duplication!)
    t_items = list(clean_t_train.items())[: (3 if args.smoke else len(clean_t_train))]
    for _, f in t_items:
        sup_lbl = np.where(f.train_rows & f.teacher & f.won, f.action, -1).astype("i8")
        if f.won:
            stats_counts["clean_t100_train_wins"] += 1
        train_entries.extend(
            build_wp22_fight_pair(
                f.episode, lookup, sup_lbl, mirror_action_map, cat_np, rare_np, False, False, "teacher_new", True,
                hist_fn=hist_fn, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
            )
        )

    # 2. Track A new 99% pure-teacher fights (wins ONLY in supervision pool!)
    if args.wp22_teacher_npz is not None and args.wp22_teacher_meta is not None:
        ta_eps = load_packed_cohort(args.wp22_teacher_npz, args.wp22_teacher_meta, "ta{idx:03d}_", has_teacher=True)
        if args.smoke:
            ta_eps = ta_eps[:2]
        for meta_item, ep, t_idx in ta_eps:
            assert t_idx is not None
            won = bool(meta_item["won"])
            f_tmp = prepare_fight(ep, lookup)
            lbl = np.where(f_tmp.train_rows & (t_idx >= 0) & won, t_idx, -1).astype("i8")
            if won:
                stats_counts["track_a_teacher_wins"] += 1
            train_entries.extend(
                build_wp22_fight_pair(
                    ep, lookup, lbl, mirror_action_map, cat_np, rare_np, False, False, "teacher_new", True,
                    hist_fn=hist_fn, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
                )
            )

    # 3. Legacy 44 old teacher (train split only -> background pool <= 8%)
    leg_items = list(legacy_t_train.items())[: (2 if args.smoke else len(legacy_t_train))]
    for _, f in leg_items:
        sup_lbl = np.where(f.train_rows & f.teacher & f.won, f.action, -1).astype("i8")
        stats_counts["legacy_teacher_train"] += 1
        train_entries.extend(
            build_wp22_fight_pair(
                f.episode, lookup, sup_lbl, mirror_action_map, cat_np, rare_np, False, False, "legacy_bg", True,
                hist_fn=hist_fn, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
            )
        )

    # 4. Strong-model on-policy DAgger (wp19 + wp20 + wp22 Track B; keep ALL losses!)
    strong_sources: list[tuple[Path, Path, str]] = [
        (args.wp19_corr_npz, args.wp19_corr_meta, "ep{idx:02d}_"),
        (args.wp20_corr_npz, args.wp20_corr_meta, "ep{idx:02d}_"),
    ]
    if args.wp22_dagger_npz is not None and args.wp22_dagger_meta is not None:
        strong_sources.append((args.wp22_dagger_npz, args.wp22_dagger_meta, "tb{idx:03d}_"))

    for npz_p, meta_p, pfx in strong_sources:
        eps = load_packed_cohort(npz_p, meta_p, pfx, has_teacher=True)
        if args.smoke:
            eps = [e for e in eps if e[0]["split"] == "train"][:2] + [e for e in eps if e[0]["split"] == "holdout"][:1]
        for meta_item, ep, t_idx in eps:
            assert t_idx is not None
            f_tmp = prepare_fight(ep, lookup)
            lbl = np.where(f_tmp.train_rows & (t_idx >= 0), t_idx, -1).astype("i8")
            if meta_item["split"] == "holdout":
                stats_counts["strong_dagger_holdout"] += 1
                strong_holdout.extend(
                    e.rf for e in build_wp22_fight_pair(
                        ep, lookup, lbl, mirror_action_map, cat_np, rare_np, True, True, "strong_dagger", False,
                        hist_fn=hist_fn, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
                    )
                )
            else:
                stats_counts["strong_dagger_train"] += 1
                if not bool(meta_item.get("won", True)):
                    stats_counts["strong_dagger_train_losses"] += 1
                train_entries.extend(
                    build_wp22_fight_pair(
                        ep, lookup, lbl, mirror_action_map, cat_np, rare_np, True, True, "strong_dagger", True,
                        hist_fn=hist_fn, anti_disc_v2=anti_disc_v2, disc_w_scale=disc_w_scale,
                    )
                )

    # 5. Legacy weak corrective cohorts (train split ONLY -> background pool <= 8%)
    for npz_p, meta_p, pfx in (
        (args.wp6_npz, args.wp6_meta, "ep{idx:02d}_"),
        (args.wp10_npz, args.wp10_meta, "ep{idx:02d}_"),
        (args.wp11_npz, args.wp11_meta, "ep{idx:02d}_"),
        (args.wp15_npz, args.wp15_meta, "ep{idx:02d}_"),
        (args.wp17_npz, args.wp17_meta, "ex{idx:03d}_"),
    ):
        eps = load_packed_cohort(npz_p, meta_p, pfx, has_teacher=True)
        eps_tr = [e for e in eps if e[0]["split"] == "train"]
        if args.smoke:
            eps_tr = eps_tr[:1]
        for _, ep, t_idx in eps_tr:
            assert t_idx is not None
            f_tmp = prepare_fight(ep, lookup)
            lbl = np.where(f_tmp.train_rows & (t_idx >= 0), t_idx, -1).astype("i8")
            stats_counts["legacy_weak_corr_train"] += 1
            train_entries.extend(
                build_wp22_fight_pair(ep, lookup, lbl, mirror_action_map, cat_np, rare_np, True, False, "legacy_bg", True, hist_fn=hist_fn)
            )

    # 6. Unlabeled student RL trajectories (Critic/AWR only)
    rl_eps = load_packed_cohort(args.rl_npz, args.rl_meta, "rl{idx:03d}_", has_teacher=False)
    for _, ep, _ in rl_eps[: (2 if args.smoke else len(rl_eps))]:
        stats_counts["student_rl"] += 1
        train_entries.extend(
            build_wp22_fight_pair(ep, lookup, None, mirror_action_map, cat_np, rare_np, False, False, "student_rl", True, hist_fn=hist_fn)
        )

    print(
        "WP22_DATA_READY",
        json.dumps({"counts": stats_counts, "train_mirrored_trajs": len(train_entries), "load_sec": round(time.monotonic() - t0, 2)}),
        flush=True,
    )

    sampler = WP22Sampler(
        train_entries,
        rng,
        proj_focus=bool(vspec["proj_focus"]),
        anti_disc_v2=anti_disc_v2,
        pure_clean=pure_clean,
    )
    cat_dev = catalogue.to(device)
    groups_dev = maneuver_groups(cat_dev)
    rare_dev = torch.from_numpy(rare_np).to(device)
    bits_dev = actor.basis[:, 6:].to(device)
    bit_pos = bits_dev[torch.from_numpy(all_sup_act).to(device)].mean(0)
    bit_weight = ((1.0 - bit_pos) / bit_pos.clamp_min(1e-4)).pow(0.20).clamp(1.0, 2.4)

    jtap_mask = (cat_dev[:, 2] & 1) != 0
    jump_any_mask = (cat_dev[:, 2] & 3) != 0
    dash_mask = (cat_dev[:, 2] & 4) != 0
    cast_mask = (cat_dev[:, 2] & 16) != 0
    cast_down_mask = cast_mask & (cat_dev[:, 1] == -1)
    focus_mask = (cat_dev[:, 2] & 32) != 0
    dash_left = dash_mask & (cat_dev[:, 0] == -1)
    dash_right = dash_mask & (cat_dev[:, 0] == 1)

    def factor_supervision(factor_logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
        c_move = F.cross_entropy(factor_logits[:, :3], cat_dev[labels, 0] + 1, reduction="none")
        c_aim = F.cross_entropy(factor_logits[:, 3:6], cat_dev[labels, 1] + 1, reduction="none")
        c_bits = F.binary_cross_entropy_with_logits(
            factor_logits[:, 6:], bits_dev[labels], pos_weight=bit_weight, reduction="none"
        ).mean(-1)
        tot = c_move + c_aim + 1.4 * c_bits
        return (tot * weights).sum() / weights.sum().clamp_min(1e-6)

    c_warm = 2 if args.smoke else args.critic_warmup
    j_steps = 4 if args.smoke else args.joint_steps
    total_steps = c_warm + j_steps
    bs = 4 if args.smoke else args.batch_size
    lr_warm = 1 if args.smoke else 500
    world_online_warm = 1 if args.smoke else 400

    actor_opt = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": 3.0e-4, "weight_decay": vspec["actor_wd"]},
            {"params": [*occ_head.parameters(), *fmov_head.parameters()], "lr": 4.5e-4, "weight_decay": 2.0e-4},
        ]
    )
    critic_opt = torch.optim.AdamW(values.parameters(), lr=3.0e-4, weight_decay=vspec["critic_wd"])

    def lr_lambda_actor(step_idx: int) -> float:
        if step_idx < lr_warm:
            return float(step_idx + 1) / float(max(1, lr_warm))
        prog = float(step_idx - lr_warm) / float(max(1, j_steps - lr_warm))
        # Decay 3e-4 -> 2e-5 (eta ratio = 2/30 = 0.0667)
        return (2.0 / 30.0) + (1.0 - 2.0 / 30.0) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    actor_sched = torch.optim.lr_scheduler.LambdaLR(actor_opt, lr_lambda_actor)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=max(1, total_steps), eta_min=2.0e-5)

    t_hold_subset = dict(list(clean_t_holdout.items())[: (2 if args.smoke else len(clean_t_holdout))])

    provenance = {
        "method": "scaled-recurrent-iql-wp22-gold-plus",
        "scaled_cfg": actor.cfg,
        "critic_cfg": values.backbone.cfg,
        "world_latent_dim": 2 * d_model,
        "actor_params": n_actor_params,
        "critic_params": n_critic_params,
        "wp_variant": f"wp22-gold-{args.variant}-clean139t-62dagger-scratch",
        "audit_sha256": sha256(args.ppo_audit),
        "opening_parent_sha256": sha256(args.parent),
        "critic_warmup": c_warm,
        "joint_steps": j_steps,
        "data_counts": stats_counts,
    }

    best_comp = float("inf")
    best_comp_step = 0
    best_comp_path: Path | None = None
    best_comp_eval: dict[str, Any] | None = None

    best_skill = -float("inf")
    best_skill_step = 0
    best_skill_path: Path | None = None
    best_skill_eval: dict[str, Any] | None = None

    val_history = []
    logs = []

    def fetch_step_batches(step_i: int) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]:
        b_len = 0 if step_i % 4 == 1 else BURN_IN
        rl_b = sampler.sample_rl(bs, b_len)
        sup_b = sampler.sample_sup(bs, b_len) if step_i > c_warm else None
        if device.type == "cuda":
            rl_b = tuple(t.pin_memory() for t in rl_b)
            if sup_b is not None:
                sup_b = tuple(t.pin_memory() for t in sup_b)
        return rl_b, sup_b

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    next_fut = prefetch_pool.submit(fetch_step_batches, 1)

    try:
      for step in range(1, total_steps + 1):
        burn = 0 if step % 4 == 1 else BURN_IN
        rl_cpu, sup_cpu = next_fut.result()
        if step < total_steps:
            next_fut = prefetch_pool.submit(fetch_step_batches, step + 1)
        rx, r_act, r_rew, r_disc, r_mask, r_occ_tgt, r_occ_m, r_atk_ev, r_hit_ev = (
            t.to(device, non_blocking=True) for t in rl_cpu
        )
        r_act = r_act.long()

        # 1. IQL Critic update (Expectile V + Double Q)
        values.train()
        critic_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            q1, q2, val = values(rx, burn)
            with torch.no_grad():
                tq1, tq2, _ = target_values(rx[:, :-1], burn)
                low_q = torch.minimum(tq1, tq2).gather(-1, r_act.unsqueeze(-1)).squeeze(-1)
                td_target = r_rew + r_disc * val[:, 1:]
            gap = (low_q - val[:, :-1]).float()
            v_loss = (torch.where(gap >= 0, 0.7, 0.3) * gap.square())[r_mask].mean()
            q_loss = sum(
                F.smooth_l1_loss(q[:, :-1].float().gather(-1, r_act.unsqueeze(-1)).squeeze(-1)[r_mask], td_target.float()[r_mask])
                for q in (q1, q2)
            )
        (q_loss + v_loss).backward()
        nn.utils.clip_grad_norm_(values.parameters(), 3.0, error_if_nonfinite=True)
        critic_opt.step()
        critic_sched.step()
        with torch.no_grad():
            for p_old, p_new in zip(target_values.parameters(), values.parameters(), strict=True):
                p_old.lerp_(p_new, 0.01)

        if step <= c_warm:
            if step in (1, c_warm) or step % 500 == 0:
                print(json.dumps({"warmup_step": step, "q": round(float(q_loss.detach()), 4), "v": round(float(v_loss.detach()), 4), "sec": round(time.monotonic() - t0, 1)}), flush=True)
            continue

        j_cur = step - c_warm
        actor.train()
        actor_opt.zero_grad(set_to_none=True)

        assert sup_cpu is not None
        sx, s_lbl, s_w, s_mask, s_occ_tgt, s_occ_m, s_fmov = (t.to(device, non_blocking=True) for t in sup_cpu)
        s_lbl = s_lbl.long()
        s_fmov = s_fmov.long()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            s_logits_amp, s_factors_amp, s_emb_amp, _ = actor.forward_full(sx[:, :-1], burn)
            r_spatial_amp, r_emb_amp, _ = actor.backbone.forward_sequence(rx[:, :-1], burn)
            r_logits_amp = actor.heads(r_emb_amp)[0]

        s_logits = s_logits_amp.float()
        s_factors = s_factors_amp.float()
        s_emb = s_emb_amp.float()
        r_spatial = r_spatial_amp.float()
        r_emb = r_emb_amp.float()
        r_logits = r_logits_amp.float()

        flat_logits = s_logits[s_mask]
        flat_factors = s_factors[s_mask]
        flat_lbl = s_lbl[s_mask]
        flat_w = s_w[s_mask]

        # STRICTLY NO global label_smoothing (avoids fighting the 6 hard physical safety penalties!)
        sup_ce = (F.cross_entropy(flat_logits, flat_lbl, reduction="none") * flat_w).sum() / flat_w.sum().clamp_min(1e-6)
        fac_loss = factor_supervision(flat_factors, flat_lbl, flat_w)
        man_loss = (greedy_maneuver_loss(flat_logits, groups_dev, flat_lbl) * flat_w).sum() / flat_w.sum().clamp_min(1e-6)

        occ_loss = flat_logits.new_zeros(())
        if s_occ_m.any():
            occ_loss = occ_loss + F.smooth_l1_loss(torch.sigmoid(occ_head(s_emb[s_occ_m])), s_occ_tgt[s_occ_m], beta=0.05)

        fmov_preds = fmov_head(s_emb[s_mask]).reshape(-1, 3, 12)
        fmov_tgts = s_fmov[s_mask]
        fmov_loss = flat_logits.new_zeros(())
        fmov_cnt = 0
        for kh in range(3):
            vm = fmov_tgts[:, kh] >= 0
            if vm.any():
                fmov_loss = fmov_loss + F.cross_entropy(fmov_preds[vm, kh], fmov_tgts[vm, kh])
                fmov_cnt += 1
        if fmov_cnt > 0:
            fmov_loss = fmov_loss / fmov_cnt

        # 6 Verified Hard Physical Constraints
        probs = flat_logits.softmax(-1)
        flat_obs = sx[:, burn:-1, :][s_mask]
        flat_xw = flat_obs[:, 0] * 10.0
        flat_bosses = flat_obs[:, FOES:DISCS].reshape(-1, 3, 36)

        disc_active = (flat_obs[:, 132] > 0.5) | (flat_obs[:, 146] > 0.5)
        wall_throw_prep = (
            (flat_bosses[:, :, 0] > 0.5)
            & ((flat_bosses[:, :, 27:30].max(dim=2).values > 0.5) | (flat_bosses[:, :, 32:36].max(dim=2).values > 0.5))
        ).any(dim=1)
        any_boss_attack_or_wall = ((flat_bosses[:, :, 0] > 0.5) & (flat_bosses[:, :, 20:36].max(dim=2).values > 0.5)).any(dim=1)
        boss_lunge = ((flat_bosses[:, :, 0] > 0.5) & (flat_bosses[:, :, 20:26].max(dim=2).values > 0.5)).any(dim=1)
        fast_threat = (
            (flat_bosses[:, :, 0] > 0.5)
            & (
                ((flat_bosses[:, :, 21] > 0.5) & (flat_bosses[:, :, 1].abs() <= 0.32))
                | ((flat_bosses[:, :, 22] > 0.5) & (flat_bosses[:, :, 1].abs() <= 0.24))
                | ((flat_bosses[:, :, 25] > 0.5) & (flat_bosses[:, :, 1].abs() <= 0.52))
            )
        ).any(dim=1)

        no_cast = (flat_obs[:, 168] < 0.5) & ((flat_obs[:, 11] < 0.5) | (flat_obs[:, 5] < (33.0 / 99.0 - 1e-3))) & (~cast_mask[flat_lbl])
        no_focus = (
            (flat_obs[:, 169] < 0.5)
            & ((flat_obs[:, 12] < 0.5) | (flat_obs[:, 5] < (33.0 / 99.0 - 1e-3)) | (flat_obs[:, 7] < 0.5) | (flat_obs[:, 4] > (6.5 / 9.0)))
            & (~focus_mask[flat_lbl])
        )
        no_dash = (flat_obs[:, 166] < 0.5) & (flat_obs[:, 13] < 0.5) & (~dash_mask[flat_lbl])
        no_jtap = (flat_obs[:, 164] < 0.5) & (flat_obs[:, 7] < 0.5) & (flat_obs[:, 8] < 0.5) & (~jtap_mask[flat_lbl])

        inv_loss = flat_logits.new_zeros(())
        for cond_m, act_m in ((no_cast, cast_mask), (no_focus, focus_mask), (no_dash, dash_mask), (no_jtap, jtap_mask)):
            if cond_m.any():
                inv_loss = inv_loss + 0.72 * probs[cond_m][:, act_m].sum(-1).mean()

        truly_calm = (~any_boss_attack_or_wall) & (~disc_active) & (flat_xw.abs() < 6.5) & (~dash_mask[flat_lbl])
        dash_cons_loss = (
            0.12 * probs[truly_calm][:, dash_mask].sum(-1).mean()
            if truly_calm.any()
            else flat_logits.new_zeros(())
        )

        bad_disc_jump = (disc_active | wall_throw_prep) & (flat_obs[:, 7] > 0.5) & (~jump_any_mask[flat_lbl])
        disc_jump_loss = (
            float(vspec["disc_jump_w"]) * probs[bad_disc_jump][:, jump_any_mask].sum(-1).mean()
            if bad_disc_jump.any()
            else flat_logits.new_zeros(())
        )

        bad_fast_quake = fast_threat & (~cast_down_mask[flat_lbl])
        quake_antic_loss = (
            0.20 * probs[bad_fast_quake][:, cast_down_mask].sum(-1).mean()
            if bad_fast_quake.any()
            else flat_logits.new_zeros(())
        )

        # Gen-2 Forensic Anti-Disc & Anti-Dive Penalties (from 10845201 & 10845208 autopsies)
        disc_quake_loss = flat_logits.new_zeros(())
        air_disc_jump_loss = flat_logits.new_zeros(())
        dive_jump_loss = flat_logits.new_zeros(())
        if anti_disc_v2:
            bad_disc_quake = (disc_active | wall_throw_prep) & (~cast_down_mask[flat_lbl])
            if bad_disc_quake.any():
                disc_quake_loss = float(vspec.get("disc_quake_w", 0.45)) * probs[bad_disc_quake][:, cast_down_mask].sum(-1).mean()
            bad_air_disc_jump = (disc_active | wall_throw_prep) & (flat_obs[:, 7] < 0.5) & (~jtap_mask[flat_lbl])
            if bad_air_disc_jump.any():
                air_disc_jump_loss = float(vspec.get("air_disc_jump_w", 0.36)) * probs[bad_air_disc_jump][:, jtap_mask].sum(-1).mean()
            overhead_dive = (
                (flat_bosses[:, :, 0] > 0.5)
                & (flat_bosses[:, :, 1].abs() <= 0.30)
                & (flat_bosses[:, :, 2] > 0.12)
                & (flat_bosses[:, :, 20:24].max(dim=2).values > 0.5)
            ).any(dim=1)
            bad_dive_jump = overhead_dive & (~jump_any_mask[flat_lbl])
            if bad_dive_jump.any():
                dive_jump_loss = 0.28 * probs[bad_dive_jump][:, jump_any_mask].sum(-1).mean()

        unsafe_focus = (~focus_mask[flat_lbl]) & (disc_active | boss_lunge | wall_throw_prep | (flat_xw.abs() > 4.0) | (flat_obs[:, 7] < 0.5))
        ufoc_loss = (
            0.32 * probs[unsafe_focus][:, focus_mask].sum(-1).mean()
            if unsafe_focus.any()
            else flat_logits.new_zeros(())
        )

        cliff = (flat_xw.abs() > 5.2) & (~dash_mask[flat_lbl])
        cliff_loss = (
            0.32 * torch.where(flat_xw > 5.2, probs[:, dash_right].sum(-1), probs[:, dash_left].sum(-1))[cliff].mean()
            if cliff.any()
            else flat_logits.new_zeros(())
        )

        # Offline RL AWR + Occupancy + RL-batch safety
        r_log_dist = r_logits[r_mask].log_softmax(-1)
        r_logp = r_log_dist.gather(-1, r_act[r_mask].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            adv_w = (3.0 * gap.detach()[r_mask]).clamp(-10.0, float(np.log(12.0))).exp() * rare_dev[r_act[r_mask]]
        awr_loss = -(adv_w * r_logp).sum() / adv_w.sum().clamp_min(1e-6)

        if r_occ_m.any():
            occ_loss = occ_loss + F.smooth_l1_loss(torch.sigmoid(occ_head(r_emb[r_occ_m])), r_occ_tgt[r_occ_m], beta=0.05)

        r_probs = r_logits[r_mask].softmax(-1)
        r_obs = rx[:, burn:-1, :][r_mask]
        r_bosses = r_obs[:, FOES:DISCS].reshape(-1, 3, 36)
        r_lunge = ((r_bosses[:, :, 0] > 0.5) & (r_bosses[:, :, 20:26].max(dim=2).values > 0.5)).any(dim=1)
        r_wall_prep = (
            (r_bosses[:, :, 0] > 0.5)
            & ((r_bosses[:, :, 27:30].max(dim=2).values > 0.5) | (r_bosses[:, :, 32:36].max(dim=2).values > 0.5))
        ).any(dim=1)
        r_disc_act = (r_obs[:, 132] > 0.5) | (r_obs[:, 146] > 0.5)
        r_xw = r_obs[:, 0] * 10.0

        for cond_m, act_m in (
            ((r_obs[:, 168] < 0.5) & ((r_obs[:, 11] < 0.5) | (r_obs[:, 5] < (33.0 / 99.0 - 1e-3))), cast_mask),
            (
                (r_obs[:, 169] < 0.5)
                & ((r_obs[:, 12] < 0.5) | (r_obs[:, 5] < (33.0 / 99.0 - 1e-3)) | (r_obs[:, 7] < 0.5) | (r_obs[:, 4] > (6.5 / 9.0))),
                focus_mask,
            ),
            ((r_obs[:, 166] < 0.5) & (r_obs[:, 13] < 0.5), dash_mask),
            ((r_obs[:, 164] < 0.5) & (r_obs[:, 7] < 0.5) & (r_obs[:, 8] < 0.5), jtap_mask),
        ):
            if cond_m.any():
                inv_loss = inv_loss + 0.65 * r_probs[cond_m][:, act_m].sum(-1).mean()
        r_ufoc = r_disc_act | r_lunge | r_wall_prep | (r_xw.abs() > 4.0) | (r_obs[:, 7] < 0.5)
        if r_ufoc.any():
            ufoc_loss = ufoc_loss + 0.28 * r_probs[r_ufoc][:, focus_mask].sum(-1).mean()
        r_cliff = r_xw.abs() > 5.4
        if r_cliff.any():
            cliff_loss = cliff_loss + 0.22 * torch.where(
                r_xw > 5.4, r_probs[:, dash_right].sum(-1), r_probs[:, dash_left].sum(-1)
            )[r_cliff].mean()

        # Online Action-Conditioned World(512, 75) auxiliary rollout
        wm_aux = sup_ce.new_zeros(())
        valid_seq = r_mask[:, :-STEPS]
        for k_step in range(1, STEPS):
            valid_seq = valid_seq & r_mask[:, k_step : -STEPS + k_step]
        b_idx, t_idx = torch.nonzero(valid_seq, as_tuple=True)
        wm_batch = None
        if len(b_idx) > 0:
            take = torch.arange(0, len(b_idx), max(1, len(b_idx) // 128), device=device)
            bi, ti = b_idx[take], t_idx[take]
            latent_live = torch.cat((r_spatial[bi, ti], r_emb[bi, ti]), dim=-1)
            hero0 = rx[bi, burn + ti, :6]
            act_seq = torch.stack([r_act[bi, ti + k] for k in range(STEPS)], dim=1)
            fut_phys = torch.stack([rx[bi, burn + ti + h, :4] - hero0[:, :4] for h in HORIZONS], dim=1)
            fut_atk = torch.stack([torch.stack([r_atk_ev[bi, ti + k] for k in range(h)], dim=1).sum(1) for h in HORIZONS], dim=1)
            fut_hit = torch.stack([(torch.stack([r_hit_ev[bi, ti + k] for k in range(h)], dim=1).sum(1) > 0).float() for h in HORIZONS], dim=1)
            wm_batch = (latent_live.detach(), hero0, act_seq, fut_phys, fut_atk, fut_hit)
            if j_cur > world_online_warm:
                for wm in worlds:
                    wm.train().requires_grad_(False)
                preds = torch.stack([wm(latent_live, hero0, act_seq)[..., :2] for wm in worlds]).mean(0)
                wm_aux = F.smooth_l1_loss(preds, fut_phys[..., :2], beta=0.04)

        actor_loss = (
            sup_ce
            + 0.20 * fac_loss
            + 0.25 * man_loss
            + float(vspec["awr_w"]) * awr_loss
            + 0.50 * occ_loss
            + 0.15 * fmov_loss
            + 0.06 * wm_aux
            + inv_loss
            + dash_cons_loss
            + disc_jump_loss
            + quake_antic_loss
            + disc_quake_loss
            + air_disc_jump_loss
            + dive_jump_loss
            + ufoc_loss
            + cliff_loss
        )
        actor_loss.backward()
        nn.utils.clip_grad_norm_([*actor.parameters(), *occ_head.parameters(), *fmov_head.parameters()], 1.5, error_if_nonfinite=True)
        actor_opt.step()
        actor_sched.step()

        if wm_batch is not None:
            lat_d, hero0, act_seq, fut_phys, fut_atk, fut_hit = wm_batch
            for wm in worlds:
                wm.train().requires_grad_(True)
            world_opt.zero_grad(set_to_none=True)
            wl_tot = lat_d.new_zeros(())
            for wm in worlds:
                pr = wm(lat_d, hero0, act_seq)
                wl_tot = wl_tot + (
                    F.smooth_l1_loss(pr[..., :2], fut_phys[..., :2], beta=0.04)
                    + 0.20 * F.smooth_l1_loss(pr[..., 2:4], fut_phys[..., 2:4], beta=0.1)
                    + 0.10 * F.smooth_l1_loss(pr[..., 4], torch.log1p(fut_atk), beta=0.3)
                    + 0.10 * F.binary_cross_entropy_with_logits(pr[..., 5], fut_hit)
                )
            wl_tot.backward()
            nn.utils.clip_grad_norm_([p for wm in worlds for p in wm.parameters()], 5.0, error_if_nonfinite=True)
            world_opt.step()

        with torch.no_grad():
            ema_decay = 0.996 if not args.smoke else 0.5
            for p_ema, p_cur in zip(ema_actor.parameters(), actor.parameters(), strict=True):
                p_ema.lerp_(p_cur, 1.0 - ema_decay)

        if j_cur % 250 == 0 or j_cur in (1, j_steps):
            row = {
                "step": step,
                "j_cur": j_cur,
                "q": round(float(q_loss.detach()), 4),
                "sup_ce": round(float(sup_ce.detach()), 4),
                "awr": round(float(awr_loss.detach()), 4),
                "occ": round(float(occ_loss.detach()), 5),
                "wm": round(float(wm_aux.detach()), 5),
                "sec": round(time.monotonic() - t0, 1),
            }
            logs.append(row)
            print(json.dumps(row), flush=True)

        if j_cur % args.val_interval == 0 or j_cur == j_steps:
            ev = evaluate_wp22(ema_actor, occ_head, t_hold_subset, strong_holdout, cat_np, device, hist_fn=hist_fn)
            val_history.append({"step": step, "j_cur": j_cur, **ev})
            ckpt_path = ckpt_dir / f"step-{step:05d}.pt"
            torch.save(
                {
                    "state": {k: v.cpu() for k, v in ema_actor.state_dict().items()},
                    "raw_state": {k: v.cpu() for k, v in actor.state_dict().items()},
                    "critic": {k: v.cpu() for k, v in values.state_dict().items()},
                    "target": {k: v.cpu() for k, v in target_values.state_dict().items()},
                    "catalogue": catalogue.cpu(),
                    "step": step,
                    "j_cur": j_cur,
                    "provenance": provenance,
                },
                ckpt_path,
            )
            p_top1 = ev["student_holdout"]["before_proj"].get("top1", 0.0)
            b_top1 = ev["student_holdout"]["before_body"].get("top1", 0.0)
            s_top1 = ev["student_holdout"]["all"].get("top1", 0.0)
            s_nll = ev["student_holdout"]["all"].get("nll", 2.0)
            t_top1 = ev["teacher_win_top1"]
            dash_f1 = ev["onset_keys"]["dash"]["f1"]
            skill_score = 0.35 * p_top1 + 0.35 * b_top1 + 0.20 * t_top1 + 0.15 * s_top1 + 0.15 * dash_f1 - 0.15 * s_nll
            print(
                "VAL",
                json.dumps({
                    "step": step,
                    "j_cur": j_cur,
                    "t_top1": round(t_top1, 4),
                    "dash_r": round(ev["dash_ratio"], 3),
                    "dash_f1": round(dash_f1, 4),
                    "focus_p": ev["onset_keys"]["focus_hold"]["pred"],
                    "occ_mae": round(ev["occ_mae"], 4),
                    "stu_nll": round(s_nll, 4),
                    "proj_top1": round(p_top1, 4),
                    "body_top1": round(b_top1, 4),
                    "proj_rows": ev["student_holdout"]["before_proj"]["rows"],
                    "body_rows": ev["student_holdout"]["before_body"]["rows"],
                    "inv_onsets": ev["student_holdout"]["all"].get("invalid_onsets", 0),
                    "skill": round(skill_score, 4),
                    "composite": round(ev["composite_loss"], 4),
                }),
                flush=True,
            )
            # Preferred selection gate: dash_ratio in [1.15, 1.50] & focus_p <= 20
            gated = (1.15 <= ev["dash_ratio"] <= 1.50) and (ev["onset_keys"]["focus_hold"]["pred"] <= 20)
            if gated:
                if ev["composite_loss"] < best_comp:
                    best_comp = ev["composite_loss"]
                    best_comp_step = step
                    best_comp_path = ckpt_path
                    best_comp_eval = ev
                if skill_score > best_skill:
                    best_skill = skill_score
                    best_skill_step = step
                    best_skill_path = ckpt_path
                    best_skill_eval = ev
            elif best_comp_path is None:
                best_comp_step = step
                best_comp_path = ckpt_path
                best_comp_eval = ev
                best_skill_step = step
                best_skill_path = ckpt_path
                best_skill_eval = ev
    finally:
        prefetch_pool.shutdown(wait=True)

    assert best_skill_path is not None and best_comp_path is not None
    assert best_skill_eval is not None and best_comp_eval is not None

    # Resolve verified WP19-PPO / WP18 scaled_opener
    opening_actor = None
    opening_bytes = None
    if args.opener_pt is not None:
        op_saved = torch.load(args.opener_pt, map_location="cpu", weights_only=True)
        ocfg = op_saved["info"]["cfg"]
        opening_actor = ScaledRecurrentActor(
            op_saved["catalogue"],
            d_boss=ocfg["d_boss"],
            d_disc=ocfg["d_disc"],
            d_past=ocfg["d_past"],
            d_model=ocfg["d_model"],
            d_ffn=ocfg["d_ffn"],
            n_blocks=ocfg["n_blocks"],
            n_gru_layers=ocfg["n_gru_layers"],
        ).eval()
        opening_actor.load_state_dict(op_saved["state"])
    elif args.opening_parent is not None:
        opening_bytes = extract_opening_bytes(args.opening_parent)
    else:
        raise ValueError("Provide --opener-pt or --opening-parent to bind verified scaled_opener!")

    sel_ckpt = args.output / "learner-wp22-skill.pt"
    sel_ckpt.write_bytes(best_skill_path.read_bytes())
    comp_ckpt = args.output / "learner-wp22-comp.pt"
    comp_ckpt.write_bytes(best_comp_path.read_bytes())

    export_kwargs = {k: v for k, v in actor_kwargs.items() if k != "dropout"}
    export_fn = export_two_phase_attn if is_v2 else export_two_phase_scaled
    exports = {}
    for tag, pt_p in (("skill", sel_ckpt), ("comp", comp_ckpt)):
        sv = torch.load(pt_p, map_location="cpu", weights_only=True)
        ac = ActorCls(catalogue, **export_kwargs).eval()
        ac.load_state_dict(sv["state"])
        exports[tag] = export_fn(
            ac,
            cat_np,
            args.output / f"world_wp22_{tag}_scaledopener.bin",
            opening_actor=opening_actor,
            opening_bytes=opening_bytes,
        )
        print(f"WP22_EXPORT_{tag.upper()}", json.dumps({"step": sv["step"], **exports[tag]}), flush=True)

    # Fit & validate final 3-member Action-Conditioned World(2*d_model, 75) on training-100.zip
    actor_hash = sha256(sel_ckpt)
    sv_skill = torch.load(sel_ckpt, map_location="cpu", weights_only=True)
    actor_best_dev = ActorCls(catalogue, **export_kwargs).to(device).eval()
    actor_best_dev.load_state_dict(sv_skill["state"])

    ppo_archive = args.ppo_audit.with_name("training-100.zip")
    ppo_groups = read_fights(ppo_archive, args.ppo_audit, catalogue)
    prepared_wm: dict[str, dict[str, tuple[Tensor, ...]]] = {"train": {}, "holdout": {}}
    for split, fights in ppo_groups.items():
        items = list(fights.items())[: (4 if args.smoke else len(fights))]
        for src, f_w in items:
            bg = build_wp22_world_game(actor_best_dev, f_w, stride=3 if split == "train" else 1, device=device, hist_fn=hist_fn)
            if bg is not None:
                prepared_wm[split][src] = bg
    del actor_best_dev

    wm_sizes = {k: {"fights": len(v), "windows": sum(len(e[0]) for e in v.values())} for k, v in prepared_wm.items()}
    hit_prior = torch.stack([(v[-1][..., 1] > 0).float().mean(0) for v in prepared_wm["train"].values()]).mean(0)
    final_worlds = [World(2 * d_model, len(cat_np)).to(device) for _ in range(3)]
    wm_progress = fit_world(final_worlds, prepared_wm["train"], updates=4 if args.smoke else args.wm_updates, seed=10576022)
    final_worlds_cpu = [m.cpu().eval() for m in final_worlds]
    holdout_cpu = {k: tuple(t.cpu() for t in v) for k, v in prepared_wm["holdout"].items()}
    wm_report = assess_world(final_worlds_cpu, holdout_cpu, hit_prior.cpu())

    world_dir = args.output / "world"
    world_dir.mkdir(exist_ok=True)
    world_pt = world_dir / "world.pt"
    torch.save(
        {
            "world": [m.state_dict() for m in final_worlds_cpu],
            "actor_sha256": actor_hash,
            "horizons": HORIZONS,
            "latent_dim": 2 * d_model,
        },
        world_pt,
    )
    wm_validation = {
        "actor_sha256": actor_hash,
        "audit_sha256": sha256(args.ppo_audit),
        "world_sha256": sha256(world_pt),
        "latent_dim": 2 * d_model,
        "position_units": "game units",
        "horizons_seconds": [0.04 * h for h in HORIZONS],
        "splits": wm_sizes,
        "measures": wm_report,
        "progress": wm_progress,
    }
    (world_dir / "validation.json").write_text(json.dumps(wm_validation, indent=2) + "\n", encoding="utf-8")

    m_eq = wm_report["equal_fight_means"]
    w_err = np.asarray(m_eq["world_mae_xy"])
    beat_all = (w_err < np.minimum(m_eq["persistence_mae_xy"], m_eq["constant_velocity_mae_xy"])) & (
        w_err < np.asarray(m_eq["shuffled_action_mae_xy"])
    )
    print("WP22_WORLD_VALIDATED", beat_all.tolist(), f"({int(beat_all.sum())}/6)", flush=True)

    summary = {
        "provenance": provenance,
        "selected_skill_step": best_skill_step,
        "selected_comp_step": best_comp_step,
        "skill_sha256": actor_hash,
        "comp_sha256": sha256(comp_ckpt),
        "exports": exports,
        "world_sha256": wm_validation["world_sha256"],
        "world_validated_components": beat_all.tolist(),
        "skill_evaluation": best_skill_eval,
        "comp_evaluation": best_comp_eval,
        "validation_history": val_history,
        "training_log": logs,
        "total_seconds": round(time.monotonic() - t0, 2),
    }
    (args.output / "wp22-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WP22_TRAIN_COMPLETE", json.dumps({"skill_step": best_skill_step, "comp_step": best_comp_step, "sha256": actor_hash}), flush=True)


if __name__ == "__main__":
    main()
