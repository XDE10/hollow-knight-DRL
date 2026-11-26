"""WP18 Scaled Deep Residual + 2-Layer Continuous GRU Recurrent IQL + Action-Conditioned World(768, 75)
+ 54-dim Future Occupancy + 36-dim Multi-Horizon Maneuver Planner + Dedicated Scaled Phase-1 Opener.

Key upgrades over WP16:
  1. ~2.98M-param ScaledRecurrentActor (d_model=384, d_ffn=512, 2 Pre-LN ResFFN, 2-Layer GRU(384),
     permutation-invariant Max+Sum 2-layer Boss/Disc encoders) -> 0x3E4C5149 native C# binary.
  2. Dismantled the old 128-GRU KL distillation chains (no longer tethered to capped WP12/13/14).
  3. All 407 zero-overlap fights (~2.46M mirrored frames including +70 WP17 extra DAgger fights).
  4. Verified C# FSM indices (WallSet/WallWind/WallThrow = 27:30, TwinWall = 32:36, Fall = 21, Charge = 25):
     - Shade-Cloak Conservation Loss (prevents wasting 1.5s shadow dash CD in safe states)
     - Wall-Throw Prep Anti-Jump Constraint (fixes 22/22 airborne disc hits)
     - Fast-Threat Quake Antic Interruption Penalty (prefers instant shadow dash over slow Quake startup)
     - Clean checkpoint selection (no premature truncation by over-strict focus_over_pen)
  5. Trains both a frozen-opener binary and a newly trained Scaled Phase-1 Opener binary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parents[2]):
    if str(_cand) not in sys.path and _cand.is_dir():
        sys.path.insert(0, str(_cand))
    if (_cand / "scripts").is_dir() and str(_cand / "scripts") not in sys.path:
        sys.path.insert(0, str(_cand / "scripts"))
    if (_cand / "python").is_dir() and str(_cand / "python") not in sys.path:
        sys.path.insert(0, str(_cand / "python"))

from audit_sisters_mainline_replay import Episode
from hkrl.spaces import BUTTON_BITS
from sisters_future_occupancy import future_occupancy
from train_hkrl_wm_recurrent_rl import (
    BURN_IN,
    GAP,
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
from train_sisters_timing_iql import CONTROL, DISCS, FOES, MIRROR, future_movements, motion_history
from wp18_scaled_model import (
    ScaledRecurrentActor,
    ScaledRecurrentValues,
    export_two_phase_scaled,
)


def maneuver_groups(catalogue: Tensor) -> Tensor:
    move = catalogue[:, 0] + 1
    aim = catalogue[:, 1] + 1
    btns = catalogue[:, 2]
    jump_any = ((btns & 3) != 0).long()
    dash = ((btns & 4) != 0).long()
    cast = (btns & 16) != 0
    focus = (btns & 32) != 0
    base = move * 4 + 2 * jump_any + dash
    cast_grp = 12 + aim
    return torch.where(focus, torch.full_like(base, 15), torch.where(cast, cast_grp, base))


def greedy_maneuver_loss(logits: Tensor, groups: Tensor, target_action: Tensor) -> Tensor:
    issued = groups[target_action]
    same = groups[None, :] == issued[:, None]
    best_in = logits.masked_fill(~same, -1e9).amax(dim=1)
    best_out = logits.masked_fill(same, -1e9).amax(dim=1)
    return F.softplus(best_out - best_in + 0.05)


def pre_window(clock: np.ndarray, hit_mask: np.ndarray, horizon: float) -> np.ndarray:
    if not hit_mask.any():
        return np.zeros(len(clock), dtype=bool)
    ht = clock[hit_mask]
    fut = np.r_[ht, np.inf][np.searchsorted(ht, clock, side="right")]
    return (fut - clock) <= horizon


def build_hazard_mask(obs: np.ndarray, clock: np.ndarray, events: np.ndarray) -> np.ndarray:
    trio = obs[:, 23] > 0.5
    bosses = obs[:, FOES:DISCS].reshape(-1, 3, 36)
    # Active attack states: Drop(20), Fall(21), Land(22), Setup(23), Wind(24), Charge(25)
    descending_or_charge = bosses[:, :, 20:26].max(axis=2) > 0.5
    near_boss = (bosses[:, :, 0] > 0.5) & (np.abs(bosses[:, :, 1]) <= 0.28)
    vulnerable = obs[:, 9] < 0.5
    boss_hazard = trio & vulnerable & (near_boss & descending_or_charge).any(1)
    pre_hit = trio & pre_window(clock, events[:, 1:5].sum(1) > 0, 0.44)
    return boss_hazard | pre_hit


def mirror_observation_192(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[:, MIRROR] *= -1.0
    return out


def load_packed_cohort(
    npz_path: Path, meta_path: Path, prefix_fmt: str, has_teacher: bool
) -> list[tuple[dict, Episode, np.ndarray | None]]:
    meta = json.loads(meta_path.read_bytes())
    if sha256(npz_path) != meta["npz_sha256"]:
        raise ValueError(f"Packed NPZ hash mismatch: {npz_path}")
    out = []
    with np.load(npz_path, allow_pickle=False) as z:
        for item in meta["episodes"]:
            idx = int(item["index"])
            p = prefix_fmt.format(idx=idx)
            clock = z[p + "clock"]
            ep = Episode(
                observation=z[p + "obs"],
                action=z[p + "act"],
                reward=z[p + "reward"],
                events=z[p + "events"],
                clock=clock,
                next_row=z[p + "next_row"],
                terminated=z[p + "terminated"],
                truncated=z[p + "truncated"],
                controller=np.full(len(clock), "student", dtype=object),
            )
            t_idx = z[p + "teacher_idx"].astype("i8") if has_teacher else None
            out.append((item, ep, t_idx))
    return out


def compute_wp18_weights(
    obs: np.ndarray,
    clock: np.ndarray,
    events: np.ndarray,
    student_actions: np.ndarray,
    labels: np.ndarray,
    cat_np: np.ndarray,
    rare_np: np.ndarray,
    is_corrective: bool,
    is_fresh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    valid = labels >= 0
    trio = obs[:, 23] > 0.5
    w = np.zeros(len(labels), dtype="f4")

    x_world = obs[:, 0] * 10.0
    low_y = (obs[:, 1] * 8.0 + 7.41) < 8.8
    near_edge = trio & (np.abs(x_world) > 6.85) & low_y
    danger_edge = trio & (np.abs(x_world) > 8.35) & low_y
    before_body = trio & pre_window(clock, events[:, 1] > 0, 0.40)
    before_proj = trio & pre_window(clock, events[:, 2] > 0, 0.44)
    before_spike = trio & pre_window(clock, events[:, 3] > 0, 0.40)
    pre_anchor = valid & (before_body | before_proj | before_spike | danger_edge)

    if not valid.any():
        return w, pre_anchor

    lbl = labels.clip(0)
    w[valid] = rare_np[lbl[valid]]

    prior_move = np.rint(obs[:, CONTROL]).astype("i8")
    prior_aim = np.rint(obs[:, CONTROL + 1]).astype("i8")
    prior_bits_mat = (obs[:, CONTROL + 4 : CONTROL + 13] > 0.5).astype("i8")
    prior_btns = (prior_bits_mat << np.arange(9, dtype="i8")).sum(1)
    prior_cmd = np.column_stack((prior_move, prior_aim, prior_btns))

    tgt_cmd = cat_np[lbl]
    switched = (tgt_cmd != prior_cmd).any(1)
    w[valid & switched] *= 1.35

    bosses = obs[:, FOES:DISCS].reshape(-1, 3, 36)
    near_boss_hazard = (
        (bosses[:, :, 0] > 0.5)
        & (np.abs(bosses[:, :, 1]) <= 0.28)
        & (bosses[:, :, 20:26].max(axis=2) > 0.5)
    ).any(1)
    disc_present = (obs[:, 132] > 0.5) | (obs[:, 146] > 0.5)
    wall_prep = (
        (bosses[:, :, 0] > 0.5)
        & ((bosses[:, :, 27:30].max(axis=2) > 0.5) | (bosses[:, :, 32:36].max(axis=2) > 0.5))
    ).any(1)
    in_threat = before_body | before_proj | before_spike | danger_edge | near_boss_hazard | disc_present

    dash_onset = ((prior_btns & 4) == 0) & ((tgt_cmd[:, 2] & 4) != 0)
    cast_onset = ((prior_btns & 16) == 0) & ((tgt_cmd[:, 2] & 16) != 0)
    jtap_onset = ((prior_btns & 1) == 0) & ((tgt_cmd[:, 2] & 1) != 0)
    shadow_ready = obs[:, 175] >= 1.50

    # Calibrated onset weighting: ONLY boost shadow dash when actually in threat & ready!
    w[valid & dash_onset & in_threat & shadow_ready] *= 1.22
    w[valid & cast_onset] *= 1.18
    w[valid & jtap_onset & ~(dash_onset | cast_onset)] *= 1.08

    if is_corrective:
        corr_mult = (
            1.0
            + 0.8 * near_edge.astype("f4")
            + 1.6 * danger_edge.astype("f4")
            + 2.4 * before_body.astype("f4")
            + 2.6 * before_proj.astype("f4")
            + 2.0 * before_spike.astype("f4")
        )
        disagree = (tgt_cmd != student_actions).any(1)
        corr_mult = np.where(disagree & in_threat, corr_mult * 1.30, corr_mult)

        late_bad_cast_or_atk = in_threat & (((student_actions[:, 2] & 24) != 0) & ((tgt_cmd[:, 2] & 24) == 0))
        bad_focus_in_threat = ((student_actions[:, 2] & 32) != 0) & ((tgt_cmd[:, 2] & 32) == 0)
        missed_evasive = in_threat & (
            (((tgt_cmd[:, 2] & 4) != 0) & ((student_actions[:, 2] & 4) == 0))
            | (((tgt_cmd[:, 2] & 16) != 0) & ((student_actions[:, 2] & 16) == 0))
        )
        corr_mult = np.where(missed_evasive | late_bad_cast_or_atk | bad_focus_in_threat, corr_mult * 1.25, corr_mult)

        jump_into_disc = (disc_present | wall_prep | before_proj) & ((student_actions[:, 2] & 3) != 0) & ((tgt_cmd[:, 2] & 3) == 0)
        corr_mult = np.where(jump_into_disc, corr_mult * 1.30, corr_mult)
        if is_fresh:
            corr_mult *= 1.12
        corr_mult = np.minimum(corr_mult, 5.0)
        w[valid] *= corr_mult[valid]

        pre_anchor = pre_anchor | (valid & (missed_evasive | late_bad_cast_or_atk | bad_focus_in_threat | jump_into_disc))

    return w.astype("f4"), pre_anchor


@dataclass
class WP18Fight:
    fight: Fight
    history_x: np.ndarray
    sup_label: np.ndarray
    sup_weight: np.ndarray
    pre_hit_anchor: np.ndarray
    hazard_anchor: np.ndarray
    occ_target: np.ndarray
    occ_valid: np.ndarray
    fmov_target: np.ndarray
    kind: str


def build_fight_pair(
    ep: Episode,
    lookup: dict[tuple[int, ...], int],
    sup_lbl: np.ndarray | None,
    mirror_action_map: np.ndarray,
    cat_np: np.ndarray,
    rare_np: np.ndarray,
    is_corrective: bool,
    is_fresh: bool,
    kind_tag: str,
    include_mirror: bool,
) -> list[WP18Fight]:
    f_orig = prepare_fight(ep, lookup)
    lbl_orig = np.full(len(ep.clock), -1, dtype="i8") if sup_lbl is None else sup_lbl

    occ_raw = future_occupancy(ep.clock, ep.observation)
    occ_val = ~np.isnan(occ_raw[:, 0, 0, 0, 0])
    occ_clean = np.nan_to_num(occ_raw, nan=0.0)
    occ_flat_orig = occ_clean.reshape(len(ep.clock), 54).astype("f2")

    if sup_lbl is not None and (sup_lbl >= 0).any():
        t_cmds = np.where((sup_lbl >= 0)[:, None], cat_np[sup_lbl.clip(0)], ep.action)
        fmov_orig = future_movements(ep.clock, ep.action, t_cmds, ep.observation[:, 23] > 0.5)
        fmov_orig[sup_lbl < 0] = -1
    else:
        fmov_orig = np.full((len(ep.clock), 3), -1, dtype="i1")

    hx_orig = motion_history(ep.clock, ep.observation)
    if sup_lbl is not None:
        sw_orig, pa_orig = compute_wp18_weights(
            ep.observation, ep.clock, ep.events, ep.action, lbl_orig, cat_np, rare_np, is_corrective, is_fresh
        )
    else:
        sw_orig = np.zeros(len(ep.clock), dtype="f4")
        pa_orig = np.zeros(len(ep.clock), dtype=bool)
    hz_orig = build_hazard_mask(ep.observation, ep.clock, ep.events)
    out = [WP18Fight(f_orig, hx_orig, lbl_orig, sw_orig, pa_orig, hz_orig, occ_flat_orig, occ_val, fmov_orig, kind_tag)]

    if include_mirror:
        m_obs = mirror_observation_192(ep.observation)
        m_act = ep.action.copy()
        m_act[:, 0] *= -1
        m_ep = Episode(m_obs, m_act, ep.reward, ep.events, ep.clock, ep.next_row, ep.terminated, ep.truncated, ep.controller)
        m_fight = prepare_fight(m_ep, lookup)
        m_lbl = np.full(len(ep.clock), -1, dtype="i8") if sup_lbl is None else np.where(sup_lbl >= 0, mirror_action_map[sup_lbl.clip(0)], -1).astype("i8")
        m_hx = motion_history(m_ep.clock, m_obs)
        if sup_lbl is not None:
            m_sw, m_pa = compute_wp18_weights(
                m_obs, m_ep.clock, m_ep.events, m_act, m_lbl, cat_np, rare_np, is_corrective, is_fresh
            )
        else:
            m_sw = np.zeros(len(ep.clock), dtype="f4")
            m_pa = np.zeros(len(ep.clock), dtype=bool)
        m_hz = build_hazard_mask(m_obs, m_ep.clock, m_ep.events)
        m_occ_flat = occ_clean[..., ::-1].reshape(len(ep.clock), 54).astype("f2")
        m_fmov = np.where(fmov_orig >= 0, (2 - fmov_orig // 4) * 4 + (fmov_orig % 4), -1).astype("i1")
        out.append(WP18Fight(m_fight, m_hx, m_lbl, m_sw, m_pa, m_hz, m_occ_flat, occ_val, m_fmov, kind_tag))
    return out


class WP18Sampler:
    def __init__(self, fights: list[WP18Fight], rng: np.random.Generator) -> None:
        self.rng = rng
        self.rl_pools: dict[tuple[str, int, bool], list[tuple[WP18Fight, int, int]]] = {}
        self.sup_pools: dict[tuple[str, int], list[tuple[WP18Fight, int, int]]] = {}

        for burn in (0, BURN_IN):
            for rl_group in ("teacher", "recent_student", "prior_student"):
                for hazard_only in (False, True):
                    pool = []
                    for rf in fights:
                        match = (
                            (rl_group == "teacher" and rf.kind == "teacher")
                            or (rl_group == "recent_student" and rf.kind in ("wp15_corrective", "recent_corrective"))
                            or (rl_group == "prior_student" and rf.kind in ("wp6_corrective", "wp17_extra", "student_rl"))
                        )
                        if not match:
                            continue
                        elig = rf.fight.train_rows & (rf.hazard_anchor if hazard_only else True)
                        for lo, hi in rf.fight.segments:
                            sl = elig[lo + burn : hi] if burn else elig[lo : min(hi, lo + UNROLL)]
                            if sl.any():
                                pool.append((rf, lo, hi))
                    if pool:
                        self.rl_pools[rl_group, burn, hazard_only] = pool

            for sup_group in ("teacher_win", "wp15_corrective", "pre_hit_anchor", "recent_corrective", "wp17_extra", "wp6_corrective"):
                pool = []
                for rf in fights:
                    if sup_group == "teacher_win":
                        ok = rf.kind == "teacher" and rf.fight.won
                        elig = rf.sup_label >= 0
                    elif sup_group == "wp15_corrective":
                        ok = rf.kind == "wp15_corrective"
                        elig = rf.sup_label >= 0
                    elif sup_group == "pre_hit_anchor":
                        ok = rf.kind in ("wp15_corrective", "recent_corrective", "wp17_extra", "wp6_corrective")
                        elig = rf.pre_hit_anchor
                    elif sup_group == "recent_corrective":
                        ok = rf.kind == "recent_corrective"
                        elig = rf.sup_label >= 0
                    elif sup_group == "wp17_extra":
                        ok = rf.kind == "wp17_extra"
                        elig = rf.sup_label >= 0
                    else:
                        ok = rf.kind == "wp6_corrective"
                        elig = rf.sup_label >= 0
                    if not ok:
                        continue
                    for lo, hi in rf.fight.segments:
                        sl = elig[lo + burn : hi] if burn else elig[lo : min(hi, lo + UNROLL)]
                        if sl.any():
                            pool.append((rf, lo, hi))
                if pool:
                    self.sup_pools[sup_group, burn] = pool

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
        for i in range(batch_size):
            if i < batch_size * 2 // 8:
                grp = "teacher"
            elif i < batch_size * 5 // 8 and ("recent_student", burn, False) in self.rl_pools:
                grp = "recent_student"
            else:
                grp = "prior_student"
            hazard = (i % 2 == 1) and ((grp, burn, True) in self.rl_pools)
            pool = self.rl_pools[grp, burn, hazard]
            rf, lo, hi = pool[int(self.rng.integers(len(pool)))]
            anchor = rf.fight.train_rows & (rf.hazard_anchor if hazard else True)
            w = self._slice(rf, lo, hi, burn, anchor)
            rows.append((w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[11], w[12]))
        return tuple(torch.from_numpy(np.stack(col)) for col in zip(*rows, strict=True))

    def sample_sup(self, batch_size: int, burn: int) -> tuple[Tensor, ...]:
        rows = []
        schedule = (
            ["teacher_win"] * 6
            + ["pre_hit_anchor"] * 4
            + ["wp15_corrective"] * 2
            + ["recent_corrective"] * 2
            + ["wp17_extra"] * 2
        )
        for i in range(batch_size):
            grp = schedule[i % len(schedule)]
            if (grp, burn) not in self.sup_pools:
                grp = "teacher_win"
            pool = self.sup_pools[grp, burn]
            rf, lo, hi = pool[int(self.rng.integers(len(pool)))]
            anchor = rf.pre_hit_anchor if grp == "pre_hit_anchor" else (rf.sup_label >= 0)
            w = self._slice(rf, lo, hi, burn, anchor)
            rows.append((w[0], w[7], w[8], w[9], w[5], w[6], w[10]))
        return tuple(torch.from_numpy(np.stack(col)) for col in zip(*rows, strict=True))


@torch.inference_mode()
def evaluate_wp18(
    actor: ScaledRecurrentActor,
    occ_head: nn.Linear,
    teacher_holdout: dict[str, Fight],
    student_holdout: list[WP18Fight],
    cat_np: np.ndarray,
    device: torch.device,
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
            logits = actor(torch.from_numpy(motion_history(ep.clock, ep.observation, rows)).to(device)[None])[0].cpu()
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
    # Mild calibration penalty only outside healthy [1.1, 2.3] range; NO premature focus truncation!
    dash_calib_pen = 0.16 * max(0.0, abs(math.log(dash_ratio / 1.65)) - 0.35)

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
def build_scaled_world_game(actor: ScaledRecurrentActor, fight: Fight, stride: int, device: torch.device) -> tuple[Tensor, ...] | None:
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
        x = torch.from_numpy(motion_history(tape.clock, tape.observation, rows)).to(device)[None]
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


def train_scaled_opener(
    teacher_groups: dict[str, dict[str, Fight]],
    catalogue: Tensor,
    mirror_action_map: np.ndarray,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[ScaledRecurrentActor, dict[str, Any]]:
    """Train a clean ScaledRecurrentActor for Phase 1 Opening (obs[:, 23] < 0.5) on pure-teacher wins."""
    rng = np.random.default_rng(seed + 777)
    cat_np = catalogue.numpy()
    lookup = {tuple(a): i for i, a in enumerate(cat_np.tolist())}

    @dataclass
    class OpenSeg:
        hx: np.ndarray
        lbl: np.ndarray
        w: np.ndarray
        segs: list[tuple[int, int]]

    def extract_open(ep: Episode, is_train: bool) -> list[OpenSeg]:
        out = []
        variants = [(ep.observation, ep.action, np.array([lookup.get(tuple(k), -1) for k in ep.action], dtype="i8"))]
        if is_train:
            m_obs = mirror_observation_192(ep.observation)
            m_act = ep.action.copy()
            m_act[:, 0] *= -1
            m_lbl = np.where(variants[0][2] >= 0, mirror_action_map[variants[0][2].clip(0)], -1).astype("i8")
            variants.append((m_obs, m_act, m_lbl))
        for obs, act, lbl in variants:
            op = obs[:, 23] < 0.5
            delta = np.diff(ep.clock)
            cuts = np.r_[0, np.flatnonzero((delta > GAP) | (op[1:] != op[:-1])) + 1, len(op)]
            segs = [(int(lo), int(hi)) for lo, hi in pairwise(cuts) if op[lo] and (hi - lo) >= 4]
            if not segs:
                continue
            hx = motion_history(ep.clock, obs)
            w = np.ones(len(obs), dtype="f4")
            before_hit = op & pre_window(ep.clock, ep.events[:, 1:5].sum(1) > 0, 0.44)
            w[before_hit] *= 2.5
            valid_l = op & (lbl >= 0)
            if valid_l.any():
                pm = np.rint(obs[:, CONTROL]).astype("i8")
                pa = np.rint(obs[:, CONTROL + 1]).astype("i8")
                pb = ((obs[:, CONTROL + 4 : CONTROL + 13] > 0.5).astype("i8") << np.arange(9, dtype="i8")).sum(1)
                pcmd = np.column_stack((pm, pa, pb))
                tcmd = cat_np[lbl.clip(0)]
                w[valid_l & (tcmd != pcmd).any(1)] *= 1.30
                w[valid_l & ((pb & 4) == 0) & ((tcmd[:, 2] & 4) != 0)] *= 1.25
                w[valid_l & ((pb & 16) == 0) & ((tcmd[:, 2] & 16) != 0)] *= 1.20
            out.append(OpenSeg(hx, np.where(valid_l, lbl, -1).astype("i8"), w, segs))
        return out

    train_segs: list[OpenSeg] = []
    for f in teacher_groups["train"].values():
        if f.pure_teacher and f.won:
            train_segs.extend(extract_open(f.episode, is_train=True))
    val_segs: list[OpenSeg] = []
    for f in teacher_groups["holdout"].values():
        if f.pure_teacher and f.won:
            val_segs.extend(extract_open(f.episode, is_train=False))

    pools: dict[int, list[tuple[OpenSeg, int, int]]] = {}
    for burn in (0, BURN_IN):
        p = []
        for s in train_segs:
            for lo, hi in s.segs:
                valid = s.lbl[lo + burn : hi] >= 0 if burn else s.lbl[lo : min(hi, lo + UNROLL)] >= 0
                if valid.any():
                    p.append((s, lo, hi))
        pools[burn] = p

    opener = ScaledRecurrentActor(
        catalogue, d_boss=64, d_disc=32, d_past=48, d_model=256, d_ffn=384, n_blocks=2, n_gru_layers=1, dropout=0.08
    ).to(device)
    ema_opener = deepcopy(opener).eval().requires_grad_(False)
    opt = torch.optim.AdamW(opener.parameters(), lr=3.5e-4, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1.5e-5)

    cat_dev = catalogue.to(device)
    bits_dev = opener.basis[:, 6:].to(device)
    jtap_mask = (cat_dev[:, 2] & 1) != 0
    dash_mask = (cat_dev[:, 2] & 4) != 0
    cast_mask = (cat_dev[:, 2] & 16) != 0
    focus_mask = (cat_dev[:, 2] & 32) != 0

    def eval_opener_holdout() -> tuple[float, float, int]:
        ema_opener.eval()
        tot = match = 0
        nll_s = 0.0
        with torch.inference_mode():
            for s in val_segs:
                for lo, hi in s.segs:
                    v = s.lbl[lo:hi] >= 0
                    if not v.any():
                        continue
                    x = torch.from_numpy(s.hx[lo:hi]).to(device)[None]
                    lg = ema_opener(x, 0)[0].cpu()
                    lp = lg.log_softmax(-1)
                    tgt = s.lbl[lo:hi][v]
                    pr = lg[v].argmax(-1).numpy()
                    tot += len(tgt)
                    match += int((pr == tgt).sum())
                    nll_s += float(-lp[v, tgt].sum())
        return match / max(1, tot), nll_s / max(1, tot), tot

    best_op_nll = float("inf")
    best_op_top1 = 0.0
    best_op_step = 0
    best_op_state = {k: v.cpu().clone() for k, v in ema_opener.state_dict().items()}
    val_tot = 0

    for step in range(1, steps + 1):
        opener.train()
        burn = 0 if step % 4 == 1 else BURN_IN
        pool = pools[burn]
        xs, lbls, ws, ms = [], [], [], []
        for _ in range(batch_size):
            s, lo, hi = pool[int(rng.integers(len(pool)))]
            if burn:
                v = s.lbl[lo + burn : hi] >= 0
                anc = lo + burn + int(rng.choice(np.flatnonzero(v)))
                st = max(lo, anc - burn - int(rng.integers(UNROLL)))
            else:
                st = lo
            idx = np.arange(st, st + burn + UNROLL)
            cl = np.minimum(idx, hi - 1)
            sc = cl[burn:]
            m = (idx[burn:] < hi) & (s.lbl[sc] >= 0)
            xs.append(s.hx[cl])
            lbls.append(s.lbl[sc].clip(0))
            ws.append(s.w[sc] * m.astype("f4"))
            ms.append(m)
        bx = torch.from_numpy(np.stack(xs)).to(device)
        bl = torch.from_numpy(np.stack(lbls)).long().to(device)
        bw = torch.from_numpy(np.stack(ws)).to(device)
        bm = torch.from_numpy(np.stack(ms)).to(device)

        opt.zero_grad(set_to_none=True)
        lg, fac, _, _ = opener.forward_full(bx, burn)
        fl, ff, flbl, fw = lg[bm], fac[bm], bl[bm], bw[bm]
        ce = (F.cross_entropy(fl, flbl, reduction="none") * fw).sum() / fw.sum().clamp_min(1e-6)
        c_m = F.cross_entropy(ff[:, :3], cat_dev[flbl, 0] + 1, reduction="none")
        c_a = F.cross_entropy(ff[:, 3:6], cat_dev[flbl, 1] + 1, reduction="none")
        c_b = F.binary_cross_entropy_with_logits(ff[:, 6:], bits_dev[flbl], reduction="none").mean(-1)
        floss = ((c_m + c_a + 1.5 * c_b) * fw).sum() / fw.sum().clamp_min(1e-6)

        probs = fl.softmax(-1)
        fobs = bx[:, burn:, :][bm]
        inv = fl.new_zeros(())
        for cm, am in (
            ((fobs[:, 168] < 0.5) & ((fobs[:, 11] < 0.5) | (fobs[:, 5] < 33.0 / 99.0 - 1e-3)) & (~cast_mask[flbl]), cast_mask),
            ((fobs[:, 169] < 0.5) & ((fobs[:, 12] < 0.5) | (fobs[:, 5] < 33.0 / 99.0 - 1e-3) | (fobs[:, 7] < 0.5)) & (~focus_mask[flbl]), focus_mask),
            ((fobs[:, 166] < 0.5) & (fobs[:, 13] < 0.5) & (~dash_mask[flbl]), dash_mask),
            ((fobs[:, 164] < 0.5) & (fobs[:, 7] < 0.5) & (fobs[:, 8] < 0.5) & (~jtap_mask[flbl]), jtap_mask),
        ):
            if cm.any():
                inv = inv + 0.60 * probs[cm][:, am].sum(-1).mean()
        (ce + 0.20 * floss + inv).backward()
        nn.utils.clip_grad_norm_(opener.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        sched.step()
        with torch.no_grad():
            d_ema = 0.995 if steps > 20 else 0.5
            for pe, pc in zip(ema_opener.parameters(), opener.parameters(), strict=True):
                pe.lerp_(pc, 1.0 - d_ema)
        if step % 500 == 0 or step == steps:
            top1_v, nll_v, val_tot = eval_opener_holdout()
            print("OPENER_VAL", json.dumps({"step": step, "top1": round(top1_v, 4), "nll": round(nll_v, 4)}), flush=True)
            if nll_v < best_op_nll:
                best_op_nll = nll_v
                best_op_top1 = top1_v
                best_op_step = step
                best_op_state = {k: v.cpu().clone() for k, v in ema_opener.state_dict().items()}

    ema_opener.cpu().load_state_dict(best_op_state)
    info = {
        "steps": steps,
        "selected_step": best_op_step,
        "train_segments": len(pools[0]),
        "holdout_rows": val_tot,
        "holdout_top1": best_op_top1,
        "holdout_nll": best_op_nll,
        "cfg": ema_opener.cfg,
    }
    print("WP18_OPENER_DONE", json.dumps(info), flush=True)
    return ema_opener.cpu(), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-archive", type=Path, required=True)
    ap.add_argument("--teacher-audit", type=Path, required=True)
    ap.add_argument("--ppo-audit", type=Path, required=True)
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
    ap.add_argument("--critic-warmup", type=int, default=400)
    ap.add_argument("--joint-steps", type=int, default=12000)
    ap.add_argument("--opener-steps", type=int, default=3000)
    ap.add_argument("--wm-updates", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--val-interval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261801)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    catalogue = native_catalogue(args.parent)
    cat_np = catalogue.numpy()
    lookup = {tuple(a): i for i, a in enumerate(cat_np.tolist())}
    mirror_action_map = np.array([lookup.get((-int(m), int(a), int(b)), -1) for m, a, b in cat_np], dtype="i8")

    # Build ~2.86M ScaledRecurrentActor & matched ScaledRecurrentValues from scratch!
    actor_kwargs = dict(d_boss=96, d_disc=48, d_past=64, d_model=384, d_ffn=512, n_blocks=2, n_gru_layers=2, dropout=0.08)
    actor = ScaledRecurrentActor(catalogue, **actor_kwargs).to(device)
    ema_actor = deepcopy(actor).eval().requires_grad_(False)
    values = ScaledRecurrentValues(catalogue, **actor_kwargs).to(device)
    target_values = deepcopy(values).eval().requires_grad_(False)

    d_model = actor.cfg["d_model"]
    occ_head = nn.Linear(d_model, 54).to(device)
    fmov_head = nn.Linear(d_model, 3 * 12).to(device)

    # Jointly trained 3-member Action-Conditioned World Model on 768-dim (spatial_384 + emb_384) latent
    worlds = [World(2 * d_model, len(cat_np)).to(device) for _ in range(3)]
    world_opt = torch.optim.AdamW([p for wm in worlds for p in wm.parameters()], lr=1.0e-3, weight_decay=1e-4)

    n_actor_params = sum(p.numel() for p in actor.parameters())
    print(f"WP18 ScaledRecurrentActor params={n_actor_params:,} cfg={actor.cfg}", flush=True)

    print("Loading ALL 407 fights + computing 54-dim future occupancy fields...", flush=True)
    teacher_groups = read_fights(args.teacher_archive, args.teacher_audit, catalogue)
    sup_acts = [f.action[f.train_rows & f.teacher] for f in teacher_groups["train"].values() if f.pure_teacher and f.won]
    all_sup_act = np.concatenate(sup_acts)
    counts_act = np.bincount(all_sup_act, minlength=len(cat_np)).astype("f4").clip(1.0)
    rare_np = np.minimum((counts_act.max() / counts_act) ** 0.08, 1.35).astype("f4")

    fights_train: list[WP18Fight] = []
    student_holdout: list[WP18Fight] = []

    t_items = list(teacher_groups["train"].items())[: (4 if args.smoke else len(teacher_groups["train"]))]
    for _, f in t_items:
        sup_lbl = np.where(f.train_rows & f.teacher & f.won, f.action, -1).astype("i8")
        fights_train.extend(
            build_fight_pair(f.episode, lookup, sup_lbl, mirror_action_map, cat_np, rare_np, False, False, "teacher", True)
        )

    for kind_tag, npz_p, meta_p, pfx, is_fresh in (
        ("wp6_corrective", args.wp6_npz, args.wp6_meta, "ep{idx:02d}_", False),
        ("recent_corrective", args.wp10_npz, args.wp10_meta, "ep{idx:02d}_", False),
        ("recent_corrective", args.wp11_npz, args.wp11_meta, "ep{idx:02d}_", False),
        ("wp15_corrective", args.wp15_npz, args.wp15_meta, "ep{idx:02d}_", True),
        ("wp17_extra", args.wp17_npz, args.wp17_meta, "ex{idx:03d}_", False),
    ):
        eps = load_packed_cohort(npz_p, meta_p, pfx, has_teacher=True)
        if args.smoke:
            eps = [e for e in eps if e[0]["split"] == "train"][:2] + [e for e in eps if e[0]["split"] == "holdout"][:1]
        for meta_item, ep, t_idx in eps:
            assert t_idx is not None
            f_tmp = prepare_fight(ep, lookup)
            lbl = np.where(f_tmp.train_rows & (t_idx >= 0), t_idx, -1).astype("i8")
            if meta_item["split"] == "holdout":
                student_holdout.extend(
                    build_fight_pair(ep, lookup, lbl, mirror_action_map, cat_np, rare_np, True, is_fresh, kind_tag, False)
                )
            else:
                fights_train.extend(
                    build_fight_pair(ep, lookup, lbl, mirror_action_map, cat_np, rare_np, True, is_fresh, kind_tag, True)
                )

    rl_eps = load_packed_cohort(args.rl_npz, args.rl_meta, "rl{idx:03d}_", has_teacher=False)
    for _, ep, _ in rl_eps[: (3 if args.smoke else len(rl_eps))]:
        fights_train.extend(
            build_fight_pair(ep, lookup, None, mirror_action_map, cat_np, rare_np, False, False, "student_rl", True)
        )

    print(
        json.dumps({
            "train_trajectories": len(fights_train),
            "student_holdout": len(student_holdout),
            "load_seconds": round(time.monotonic() - t0, 2),
        }),
        flush=True,
    )

    sampler = WP18Sampler(fights_train, rng)
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
    j_steps = 2 if args.smoke else args.joint_steps
    total_steps = c_warm + j_steps
    bs = 4 if args.smoke else args.batch_size
    lr_warm = 1 if args.smoke else 600

    actor_opt = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": 3.5e-4, "weight_decay": 3e-4},
            {"params": [*occ_head.parameters(), *fmov_head.parameters()], "lr": 5.0e-4, "weight_decay": 1e-4},
        ]
    )
    critic_opt = torch.optim.AdamW(values.parameters(), lr=3.0e-4, weight_decay=2e-4)

    def lr_lambda_actor(step_idx: int) -> float:
        if step_idx < lr_warm:
            return float(step_idx + 1) / float(max(1, lr_warm))
        prog = float(step_idx - lr_warm) / float(max(1, j_steps - lr_warm))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    actor_sched = torch.optim.lr_scheduler.LambdaLR(actor_opt, lr_lambda_actor)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=max(1, total_steps), eta_min=2.0e-5)

    t_hold_subset = dict(list(teacher_groups["holdout"].items())[: (2 if args.smoke else len(teacher_groups["holdout"]))])

    provenance = {
        "method": "scaled-recurrent-iql-v1",
        "scaled_cfg": actor.cfg,
        "actor_params": n_actor_params,
        "wp_variant": "wp18-scaled-384-2lgru-clean-iql-wm",
        "audit_sha256": sha256(args.ppo_audit),
        "opening_parent_sha256": sha256(args.parent),
        "critic_warmup": c_warm,
        "joint_steps": j_steps,
    }

    best_gated_comp = float("inf")
    best_gated_step = 0
    best_gated_path: Path | None = None
    best_gated_eval: dict[str, Any] | None = None

    best_any_comp = float("inf")
    best_any_step = 0
    best_any_path: Path | None = None
    best_any_eval: dict[str, Any] | None = None

    val_history = []
    logs = []
    world_online_warm = 1 if args.smoke else 400

    for step in range(1, total_steps + 1):
        burn = 0 if step % 4 == 1 else BURN_IN
        rx, r_act, r_rew, r_disc, r_mask, r_occ_tgt, r_occ_m, r_atk_ev, r_hit_ev = (
            t.to(device) for t in sampler.sample_rl(bs, burn)
        )
        r_act = r_act.long()

        # 1. IQL Critic update (Expectile V + Double Q)
        values.train()
        critic_opt.zero_grad(set_to_none=True)
        q1, q2, val = values(rx, burn)
        with torch.no_grad():
            tq1, tq2, _ = target_values(rx[:, :-1], burn)
            low_q = torch.minimum(tq1, tq2).gather(-1, r_act.unsqueeze(-1)).squeeze(-1)
            td_target = r_rew + r_disc * val[:, 1:]
        gap = low_q - val[:, :-1]
        v_loss = (torch.where(gap >= 0, 0.7, 0.3) * gap.square())[r_mask].mean()
        q_loss = sum(
            F.smooth_l1_loss(q[:, :-1].gather(-1, r_act.unsqueeze(-1)).squeeze(-1)[r_mask], td_target[r_mask])
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
            continue

        j_cur = step - c_warm
        actor.train()
        actor_opt.zero_grad(set_to_none=True)

        # 2. Supervised + Corrective + Spatiotemporal Occupancy + Maneuver Heads
        sx, s_lbl, s_w, s_mask, s_occ_tgt, s_occ_m, s_fmov = (t.to(device) for t in sampler.sample_sup(bs, burn))
        s_lbl = s_lbl.long()
        s_fmov = s_fmov.long()
        s_logits, s_factors, s_emb, _ = actor.forward_full(sx[:, :-1], burn)

        flat_logits = s_logits[s_mask]
        flat_factors = s_factors[s_mask]
        flat_lbl = s_lbl[s_mask]
        flat_w = s_w[s_mask]

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

        # 3. Exact Verified Physical Constraints (All indices verified vs SistersTimingState.cs!)
        probs = flat_logits.softmax(-1)
        flat_obs = sx[:, burn:-1, :][s_mask]
        flat_xw = flat_obs[:, 0] * 10.0
        flat_bosses = flat_obs[:, FOES:DISCS].reshape(-1, 3, 36)

        disc_active = (flat_obs[:, 132] > 0.5) | (flat_obs[:, 146] > 0.5)
        # VERIFIED C# FSM OFFSETS: WallSet=27, WallWind=28, WallThrow=29, TwinWall=32..35!
        wall_throw_prep = (
            (flat_bosses[:, :, 0] > 0.5)
            & ((flat_bosses[:, :, 27:30].max(dim=2).values > 0.5) | (flat_bosses[:, :, 32:36].max(dim=2).values > 0.5))
        ).any(dim=1)
        boss_lunge = ((flat_bosses[:, :, 0] > 0.5) & (flat_bosses[:, :, 20:26].max(dim=2).values > 0.5)).any(dim=1)
        near_boss_threat = (
            (flat_bosses[:, :, 0] > 0.5)
            & (flat_bosses[:, :, 1].abs() <= 0.30)
            & (flat_bosses[:, :, 20:26].max(dim=2).values > 0.5)
        ).any(dim=1)
        # Fast threats where 0.25s Quake startup gets interrupted before i-frames (Fall=21, Charge=25)
        fast_threat = (
            (flat_bosses[:, :, 0] > 0.5)
            & (
                ((flat_bosses[:, :, 21] > 0.5) & (flat_bosses[:, :, 1].abs() <= 0.30))
                | ((flat_bosses[:, :, 25] > 0.5) & (flat_bosses[:, :, 1].abs() <= 0.50))
            )
        ).any(dim=1)

        # Onset-only impossible action masks
        no_cast = (flat_obs[:, 168] < 0.5) & ((flat_obs[:, 11] < 0.5) | (flat_obs[:, 5] < (33.0 / 99.0 - 1e-3))) & (~cast_mask[flat_lbl])
        no_focus = (flat_obs[:, 169] < 0.5) & ((flat_obs[:, 12] < 0.5) | (flat_obs[:, 5] < (33.0 / 99.0 - 1e-3)) | (flat_obs[:, 7] < 0.5)) & (~focus_mask[flat_lbl])
        no_dash = (flat_obs[:, 166] < 0.5) & (flat_obs[:, 13] < 0.5) & (~dash_mask[flat_lbl])
        no_jtap = (flat_obs[:, 164] < 0.5) & (flat_obs[:, 7] < 0.5) & (flat_obs[:, 8] < 0.5) & (~jtap_mask[flat_lbl])

        inv_loss = flat_logits.new_zeros(())
        for cond_m, act_m in ((no_cast, cast_mask), (no_focus, focus_mask), (no_dash, dash_mask), (no_jtap, jtap_mask)):
            if cond_m.any():
                inv_loss = inv_loss + 0.70 * probs[cond_m][:, act_m].sum(-1).mean()

        # Fix 1: Shade-Cloak Conservation Loss in safe states
        safe_no_dash = (~near_boss_threat) & (~disc_active) & (~wall_throw_prep) & (flat_xw.abs() < 6.8) & (~dash_mask[flat_lbl])
        dash_cons_loss = (
            0.14 * probs[safe_no_dash][:, dash_mask].sum(-1).mean()
            if safe_no_dash.any()
            else flat_logits.new_zeros(())
        )

        # Fix 2: Anti-Jump during Disc Active OR Wall-Throw Prep (27:30, 32:36)
        bad_disc_jump = (disc_active | wall_throw_prep) & (flat_obs[:, 7] > 0.5) & (~jump_any_mask[flat_lbl])
        disc_jump_loss = (
            0.26 * probs[bad_disc_jump][:, jump_any_mask].sum(-1).mean()
            if bad_disc_jump.any()
            else flat_logits.new_zeros(())
        )

        # Fix 3: Fast-threat Quake Antic Interruption Penalty
        bad_fast_quake = fast_threat & (~cast_down_mask[flat_lbl])
        quake_antic_loss = (
            0.18 * probs[bad_fast_quake][:, cast_down_mask].sum(-1).mean()
            if bad_fast_quake.any()
            else flat_logits.new_zeros(())
        )

        # Instant Threat-Abort for Focus on ALL frames
        unsafe_focus = (~focus_mask[flat_lbl]) & (disc_active | boss_lunge | wall_throw_prep | (flat_xw.abs() > 4.0) | (flat_obs[:, 7] < 0.5))
        ufoc_loss = (
            0.30 * probs[unsafe_focus][:, focus_mask].sum(-1).mean()
            if unsafe_focus.any()
            else flat_logits.new_zeros(())
        )

        cliff = (flat_xw.abs() > 5.2) & (~dash_mask[flat_lbl])
        cliff_loss = (
            0.32 * torch.where(flat_xw > 5.2, probs[:, dash_right].sum(-1), probs[:, dash_left].sum(-1))[cliff].mean()
            if cliff.any()
            else flat_logits.new_zeros(())
        )

        # 4. Offline RL (AWR) + Future-Occupancy + Joint World Model on RL batch
        r_spatial, r_emb, _ = actor.backbone.forward_sequence(rx[:, :-1], burn)
        r_logits = actor.heads(r_emb)[0]
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
            ((r_obs[:, 169] < 0.5) & ((r_obs[:, 12] < 0.5) | (r_obs[:, 5] < (33.0 / 99.0 - 1e-3)) | (r_obs[:, 7] < 0.5)), focus_mask),
            ((r_obs[:, 166] < 0.5) & (r_obs[:, 13] < 0.5), dash_mask),
            ((r_obs[:, 164] < 0.5) & (r_obs[:, 7] < 0.5) & (r_obs[:, 8] < 0.5), jtap_mask),
        ):
            if cond_m.any():
                inv_loss = inv_loss + 0.65 * r_probs[cond_m][:, act_m].sum(-1).mean()
        r_ufoc = r_disc_act | r_lunge | r_wall_prep | (r_xw.abs() > 4.0) | (r_obs[:, 7] < 0.5)
        if r_ufoc.any():
            ufoc_loss = ufoc_loss + 0.25 * r_probs[r_ufoc][:, focus_mask].sum(-1).mean()
        r_cliff = r_xw.abs() > 5.4
        if r_cliff.any():
            cliff_loss = cliff_loss + 0.20 * torch.where(
                r_xw > 5.4, r_probs[:, dash_right].sum(-1), r_probs[:, dash_left].sum(-1)
            )[r_cliff].mean()

        # Online Action-Conditioned World Model auxiliary rollout
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
            fut_atk = torch.stack([
                torch.stack([r_atk_ev[bi, ti + k] for k in range(h)], dim=1).sum(1) for h in HORIZONS
            ], dim=1)
            fut_hit = torch.stack([
                (torch.stack([r_hit_ev[bi, ti + k] for k in range(h)], dim=1).sum(1) > 0).float() for h in HORIZONS
            ], dim=1)
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
            + 0.24 * awr_loss
            + 0.50 * occ_loss
            + 0.15 * fmov_loss
            + 0.06 * wm_aux
            + inv_loss
            + dash_cons_loss
            + disc_jump_loss
            + quake_antic_loss
            + ufoc_loss
            + cliff_loss
        )
        actor_loss.backward()
        nn.utils.clip_grad_norm_([*actor.parameters(), *occ_head.parameters(), *fmov_head.parameters()], 1.5, error_if_nonfinite=True)
        actor_opt.step()
        actor_sched.step()

        # Update online world ensemble on detached latents
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

        if j_cur % 300 == 0 or j_cur in (1, j_steps):
            row = {
                "step": step,
                "q": round(float(q_loss.detach()), 4),
                "sup_ce": round(float(sup_ce.detach()), 4),
                "occ": round(float(occ_loss.detach()), 5),
                "fmov": round(float(fmov_loss.detach()), 4),
                "wm": round(float(wm_aux.detach()), 5),
                "dcons": round(float(dash_cons_loss.detach()), 5),
                "sec": round(time.monotonic() - t0, 1),
            }
            logs.append(row)
            print(json.dumps(row), flush=True)

        if j_cur % args.val_interval == 0 or j_cur == j_steps:
            ev = evaluate_wp18(ema_actor, occ_head, t_hold_subset, student_holdout, cat_np, device)
            val_history.append({"step": step, **ev})
            ckpt_path = ckpt_dir / f"step-{step:05d}.pt"
            torch.save(
                {
                    "state": {k: v.cpu() for k, v in ema_actor.state_dict().items()},
                    "critic": {k: v.cpu() for k, v in values.state_dict().items()},
                    "target": {k: v.cpu() for k, v in target_values.state_dict().items()},
                    "catalogue": catalogue.cpu(),
                    "step": step,
                    "provenance": provenance,
                },
                ckpt_path,
            )
            print(
                "VAL",
                json.dumps({
                    "step": step,
                    "t_top1": round(ev["teacher_win_top1"], 4),
                    "dash_r": round(ev["dash_ratio"], 3),
                    "focus_p": ev["onset_keys"]["focus_hold"]["pred"],
                    "occ_mae": round(ev["occ_mae"], 4),
                    "stu_nll": round(ev["student_holdout"]["all"].get("nll", 0), 4),
                    "proj_top1": round(ev["student_holdout"]["before_proj"].get("top1", 0), 4),
                    "body_top1": round(ev["student_holdout"]["before_body"].get("top1", 0), 4),
                    "inv_onsets": ev["student_holdout"]["all"].get("invalid_onsets", 0),
                    "composite": round(ev["composite_loss"], 4),
                }),
                flush=True,
            )
            if ev["composite_loss"] < best_any_comp:
                best_any_comp = ev["composite_loss"]
                best_any_step = step
                best_any_path = ckpt_path
                best_any_eval = ev
            if (
                0.85 <= ev["dash_ratio"] <= 2.45
                and 6 <= ev["onset_keys"]["focus_hold"]["pred"] <= 48
                and ev["composite_loss"] < best_gated_comp
            ):
                best_gated_comp = ev["composite_loss"]
                best_gated_step = step
                best_gated_path = ckpt_path
                best_gated_eval = ev

    if best_gated_path is not None:
        best_step, best_path, best_eval = best_gated_step, best_gated_path, best_gated_eval
    else:
        assert best_any_path is not None and best_any_eval is not None
        best_step, best_path, best_eval = best_any_step, best_any_path, best_any_eval

    selected_ckpt = args.output / "learner-wp18-iql.pt"
    selected_ckpt.write_bytes(best_path.read_bytes())
    actor_hash = sha256(selected_ckpt)
    print(
        f"Selected WP18 Scaled EMA checkpoint: step={best_step}, dash_r={best_eval['dash_ratio']:.3f}, "
        f"focus_p={best_eval['onset_keys']['focus_hold']['pred']}, stu_nll={best_eval['student_holdout']['all'].get('nll', 0):.4f}, "
        f"sha256={actor_hash}",
        flush=True,
    )

    # Train dedicated Scaled Phase-1 Opener
    opener_actor, opener_info = train_scaled_opener(
        teacher_groups,
        catalogue,
        mirror_action_map,
        steps=4 if args.smoke else args.opener_steps,
        batch_size=4 if args.smoke else args.batch_size,
        seed=args.seed,
        device=device,
    )
    opener_ckpt = args.output / "opener-wp18-scaled.pt"
    torch.save({"state": opener_actor.state_dict(), "catalogue": catalogue.cpu(), "info": opener_info}, opener_ckpt)

    # Export BOTH native binaries (old frozen opener & new scaled opener)
    best_saved = torch.load(selected_ckpt, map_location="cpu", weights_only=True)
    actor_best_export = ScaledRecurrentActor(catalogue, **{k: v for k, v in actor_kwargs.items() if k != "dropout"}).eval()
    actor_best_export.load_state_dict(best_saved["state"])

    exp_old = export_two_phase_scaled(
        actor_best_export, cat_np, args.output / "world_wp18_iql_oldopener.bin", opening_parent_bin=args.parent
    )
    exp_new = export_two_phase_scaled(
        actor_best_export, cat_np, args.output / "world_wp18_iql_scaledopener.bin", opening_actor=opener_actor
    )
    print("WP18_EXPORT_OLD_OPENER", json.dumps(exp_old), flush=True)
    print("WP18_EXPORT_SCALED_OPENER", json.dumps(exp_new), flush=True)

    # Final 3-member Action-Conditioned World Model (768, 75) fit & holdout validation on training-100.zip
    actor_best_dev = actor_best_export.to(device).eval()
    ppo_archive = args.ppo_audit.with_name("training-100.zip")
    ppo_groups = read_fights(ppo_archive, args.ppo_audit, catalogue)
    prepared_wm: dict[str, dict[str, tuple[Tensor, ...]]] = {"train": {}, "holdout": {}}
    for split, fights in ppo_groups.items():
        items = list(fights.items())[: (4 if args.smoke else len(fights))]
        for src, f_w in items:
            bg = build_scaled_world_game(actor_best_dev, f_w, stride=3 if split == "train" else 1, device=device)
            if bg is not None:
                prepared_wm[split][src] = bg
    del actor_best_dev

    wm_sizes = {k: {"fights": len(v), "windows": sum(len(e[0]) for e in v.values())} for k, v in prepared_wm.items()}
    hit_prior = torch.stack([(v[-1][..., 1] > 0).float().mean(0) for v in prepared_wm["train"].values()]).mean(0)
    final_worlds = [World(2 * d_model, len(cat_np)).to(device) for _ in range(3)]
    wm_progress = fit_world(final_worlds, prepared_wm["train"], updates=4 if args.smoke else args.wm_updates, seed=10576018)
    final_worlds_cpu = [m.cpu().eval() for m in final_worlds]
    holdout_cpu = {k: tuple(t.cpu() for t in v) for k, v in prepared_wm["holdout"].items()}
    wm_report = assess_world(final_worlds_cpu, holdout_cpu, hit_prior.cpu())

    world_dir = args.output / "world"
    world_dir.mkdir(exist_ok=True)
    world_pt = world_dir / "world.pt"
    torch.save({"world": [m.state_dict() for m in final_worlds_cpu], "actor_sha256": actor_hash, "horizons": HORIZONS}, world_pt)
    wm_validation = {
        "actor_sha256": actor_hash,
        "audit_sha256": sha256(args.ppo_audit),
        "world_sha256": sha256(world_pt),
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
    print("WP18_WORLD_VALIDATED", beat_all.tolist(), f"({int(beat_all.sum())}/6)", flush=True)

    summary = {
        "provenance": provenance,
        "selected_step": best_step,
        "learner_sha256": actor_hash,
        "opener_info": opener_info,
        "export_old_opener": exp_old,
        "export_scaled_opener": exp_new,
        "world_sha256": wm_validation["world_sha256"],
        "world_validated_components": beat_all.tolist(),
        "selected_evaluation": best_eval,
        "validation_history": val_history,
        "training_log": logs,
        "total_seconds": round(time.monotonic() - t0, 2),
    }
    (args.output / "wp18-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WP18_TRAIN_COMPLETE", json.dumps({"best_step": best_step, "sha256": actor_hash}), flush=True)


if __name__ == "__main__":
    main()
