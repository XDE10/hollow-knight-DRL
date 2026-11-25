"""Train a clean V2 Entity-Attention + 4-Lag Phase-1 Opener (0x3F4C5149) on all 139 clean 99% teacher wins.

Fixes the old opener-wp18-scaled.pt bleed (which mixed 34 legacy 77% fights and used tiny d_disc=32,
losing 9 masks across 6/30 fights in confirm-30 including entering Trio at 6 HP and 7 HP).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
import sys
import time
from pathlib import Path

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
from train_hkrl_wm_recurrent_rl import BURN_IN, GAP, UNROLL, native_catalogue, read_fights
from train_sisters_timing_iql import CONTROL
from train_wp18_scaled_drl import load_packed_cohort, mirror_observation_192, pre_window
from wp22_attn_model import AttnRecurrentActor, motion_history_4lag, serialize_attn_actor


@dataclass
class OpenSeg:
    hx: np.ndarray
    lbl: np.ndarray
    w: np.ndarray
    segs: list[tuple[int, int]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-archive", type=Path, required=True)
    ap.add_argument("--teacher-audit", type=Path, required=True)
    ap.add_argument("--ppo-audit", type=Path, required=True)
    ap.add_argument("--wp22-teacher-npz", type=Path, required=True)
    ap.add_argument("--wp22-teacher-meta", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6500)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20262299)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    catalogue = native_catalogue(args.parent)
    cat_np = catalogue.numpy()
    lookup = {tuple(a): i for i, a in enumerate(cat_np.tolist())}
    mirror_action_map = np.array([lookup.get((-int(m), int(a), int(b)), -1) for m, a, b in cat_np], dtype="i8")

    t100_audit = json.loads(args.ppo_audit.read_bytes())
    t100_sources = {e["provenance"]["source"]: e["provenance"]["split"] for e in t100_audit["episodes"]}
    teacher_groups = read_fights(args.teacher_archive, args.teacher_audit, catalogue)

    def extract_open(ep: Episode, is_train: bool) -> list[OpenSeg]:
        out: list[OpenSeg] = []
        base_lbl = np.array([lookup.get(tuple(k), -1) for k in ep.action], dtype="i8")
        # Reward zero-hit Phase-1 openings heavily (teaches 9/9 HP Trio entry!)
        p1_mask = ep.observation[:, 23] < 0.5
        p1_hits = int(ep.events[p1_mask, 1:5].sum())
        p1_quality = 1.35 if p1_hits == 0 else (0.75 if p1_hits >= 2 else 0.95)

        variants = [(ep.observation, ep.action, base_lbl)]
        if is_train:
            m_obs = mirror_observation_192(ep.observation)
            m_act = ep.action.copy()
            m_act[:, 0] *= -1
            m_lbl = np.where(base_lbl >= 0, mirror_action_map[base_lbl.clip(0)], -1).astype("i8")
            variants.append((m_obs, m_act, m_lbl))

        for obs, act, lbl in variants:
            op = obs[:, 23] < 0.5
            delta = np.diff(ep.clock)
            cuts = np.r_[0, np.flatnonzero((delta > GAP) | (op[1:] != op[:-1])) + 1, len(op)]
            segs = [(int(lo), int(hi)) for lo, hi in pairwise(cuts) if op[lo] and (hi - lo) >= 4]
            if not segs:
                continue
            hx = motion_history_4lag(ep.clock, obs)
            w = np.full(len(obs), p1_quality, dtype="f4")
            before_hit = op & pre_window(ep.clock, ep.events[:, 1:5].sum(1) > 0, 0.60)
            disc_present = op & ((obs[:, 132] > 0.5) | (obs[:, 146] > 0.5))
            w[before_hit] *= 2.5
            w[disc_present] *= 1.65
            valid_l = op & (lbl >= 0)
            if valid_l.any():
                pm = np.rint(obs[:, CONTROL]).astype("i8")
                pa = np.rint(obs[:, CONTROL + 1]).astype("i8")
                pb = ((obs[:, CONTROL + 4 : CONTROL + 13] > 0.5).astype("i8") << np.arange(9, dtype="i8")).sum(1)
                pcmd = np.column_stack((pm, pa, pb))
                tcmd = cat_np[lbl.clip(0)]
                w[valid_l & (tcmd != pcmd).any(1)] *= 1.30
                w[valid_l & ((pb & 4) == 0) & ((tcmd[:, 2] & 4) != 0)] *= 1.35
                w[valid_l & ((pb & 16) == 0) & ((tcmd[:, 2] & 16) != 0)] *= 1.25
            out.append(OpenSeg(hx, np.where(valid_l, lbl, -1).astype("i8"), w, segs))
        return out

    train_segs: list[OpenSeg] = []
    val_segs: list[OpenSeg] = []
    n_clean_tr = n_clean_va = n_ta = 0

    for grp in teacher_groups.values():
        for name, f in grp.items():
            if not (f.pure_teacher and f.won):
                continue
            if name not in t100_sources:
                continue  # Strictly ZERO legacy 77% contamination!
            if t100_sources[name] == "holdout":
                val_segs.extend(extract_open(f.episode, is_train=False))
                n_clean_va += 1
            else:
                train_segs.extend(extract_open(f.episode, is_train=True))
                n_clean_tr += 1

    ta_eps = load_packed_cohort(args.wp22_teacher_npz, args.wp22_teacher_meta, "ta{idx:03d}_", has_teacher=True)
    for meta_item, ep, _ in ta_eps:
        if bool(meta_item["won"]):
            train_segs.extend(extract_open(ep, is_train=True))
            n_ta += 1

    print(f"WP22_OPENER_DATA clean_t100_train={n_clean_tr} track_a={n_ta} clean_holdout={n_clean_va}", flush=True)

    pools: dict[int, list[tuple[OpenSeg, int, int]]] = {}
    for burn in (0, BURN_IN):
        p = []
        for s in train_segs:
            for lo, hi in s.segs:
                valid = s.lbl[lo + burn : hi] >= 0 if burn else s.lbl[lo : min(hi, lo + UNROLL)] >= 0
                if valid.any():
                    p.append((s, lo, hi))
        pools[burn] = p

    opener = AttnRecurrentActor(
        catalogue,
        d_boss=64,
        d_disc=80,
        d_past=48,
        d_tok=96,
        n_heads=4,
        d_model=256,
        d_ffn=512,
        n_blocks=2,
        n_gru_layers=1,
        dropout=0.08,
    ).to(device)
    ema_opener = deepcopy(opener).eval().requires_grad_(False)
    opt = torch.optim.AdamW(opener.parameters(), lr=3.5e-4, weight_decay=8e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=1.5e-5)

    cat_dev = catalogue.to(device)
    bits_dev = opener.basis[:, 6:].to(device)
    jtap_mask = (cat_dev[:, 2] & 1) != 0
    dash_mask = (cat_dev[:, 2] & 4) != 0
    cast_mask = (cat_dev[:, 2] & 16) != 0
    focus_mask = (cat_dev[:, 2] & 32) != 0

    @torch.inference_mode()
    def eval_opener_holdout() -> tuple[float, float, int]:
        ema_opener.eval()
        tot = match = 0
        nll_s = 0.0
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

    best_nll = float("inf")
    best_top1 = 0.0
    best_step = 0
    best_state = {k: v.cpu().clone() for k, v in ema_opener.state_dict().items()}
    val_tot = 0

    for step in range(1, args.steps + 1):
        opener.train()
        burn = 0 if step % 4 == 1 else BURN_IN
        pool = pools[burn]
        xs, lbls, ws, ms = [], [], [], []
        for _ in range(args.batch_size):
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            lg_a, fac_a, _, _ = opener.forward_full(bx, burn)
        lg, fac = lg_a.float(), fac_a.float()
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
                inv = inv + 0.65 * probs[cm][:, am].sum(-1).mean()
        (ce + 0.20 * floss + inv).backward()
        nn.utils.clip_grad_norm_(opener.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pc in zip(ema_opener.parameters(), opener.parameters(), strict=True):
                pe.lerp_(pc, 0.005)
        if step % 500 == 0 or step == args.steps:
            top1_v, nll_v, val_tot = eval_opener_holdout()
            print("OPENER_VAL", json.dumps({"step": step, "top1": round(top1_v, 4), "nll": round(nll_v, 4), "sec": round(time.monotonic() - t0, 1)}), flush=True)
            if top1_v > best_top1 or (abs(top1_v - best_top1) < 5e-4 and nll_v < best_nll):
                best_nll = nll_v
                best_top1 = top1_v
                best_step = step
                best_state = {k: v.cpu().clone() for k, v in ema_opener.state_dict().items()}

    ema_opener.cpu().load_state_dict(best_state)
    blob = serialize_attn_actor(ema_opener, cat_np)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "v2_attn": True,
        "cfg": ema_opener.cfg,
        "steps": args.steps,
        "selected_step": best_step,
        "clean_train_fights": n_clean_tr + n_ta,
        "clean_holdout_fights": n_clean_va,
        "holdout_rows": val_tot,
        "holdout_top1": best_top1,
        "holdout_nll": best_nll,
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
    }
    torch.save({"state": best_state, "catalogue": catalogue.cpu(), "info": info}, args.output)
    print("WP22_OPENER_DONE", json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
