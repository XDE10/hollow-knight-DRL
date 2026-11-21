"""On-policy PPO directly on both native key heads of the learned nine-win controller."""

from __future__ import annotations

import argparse
import copy
import io
import json
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from replay_sisters_primitive_heads import actions, models
from sisters_directional_options import available_keys
from sisters_primitive_rollout import onsets
from torch import nn
from torch.nn import functional as F
from train_sisters_champion_ppo import advantages
from train_sisters_champion_router import digest
from train_sisters_option_ppo import option_rewards, verified_events
from train_sisters_primitive_heads import Outcome, replace_head
from train_sisters_timing_iql import (
    OPTION_MAGIC,
    option_components,
    phase_checkpoint,
    phase_components,
)


def distribution(
    logits: torch.Tensor, available: torch.Tensor, temperature: float, *, smooth: bool = False
) -> torch.Tensor:
    if smooth:
        return 0.98 * (logits / temperature).softmax(1) + 0.02 / logits.shape[1]
    greedy = F.one_hot(logits.argmax(1), logits.shape[1]).bool()
    allowed = available | greedy
    weights = ((logits - logits.max(1, keepdim=True).values) / temperature).clamp_min(
        -50
    ).exp() * allowed
    other = allowed & ~greedy
    maximum = logits.masked_fill(~other, -torch.inf).amax(1, keepdim=True)
    maximum = torch.where(
        other.any(1, keepdim=True), maximum, torch.zeros_like(maximum)
    )
    # Only exponentiate legal alternatives to avoid overflow in the masked entries.
    offset = ((logits - maximum) / temperature).masked_fill(~other, -torch.inf)
    alternative = offset.clamp(-50, 0).exp() * other
    mass = (
        0.78 * greedy
        + 0.12 * weights / weights.sum(1, keepdim=True)
        + 0.08 * alternative / alternative.sum(1, keepdim=True).clamp_min(1)
        + 0.02 * other / other.sum(1, keepdim=True).clamp_min(1)
    )
    return torch.where(other.any(1, keepdim=True), mass, greedy.float())


def read_games(
    data, behavior, nets, cats, gating, values, temperature, clean_cost, *, smooth=False
):
    groups = [[], []]
    anchors = [[], []]
    outcomes = []
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        spec = json.loads(z.read("batch.json"))
        entries = json.loads(z.read("manifest.json"))
        if (
            spec["actor_sha256"] != behavior
            or spec["temperature"] != temperature
            or spec.get("smooth_key_sampling", False) != smooth
            or [g["unity_reset_seed"] for g in entries] != spec["seeds"]
            or len(set(spec["seeds"])) != len(entries)
        ):
            raise ValueError("Wrong behavior or duplicate PPO games")
        for game in entries:
            seed = game["unity_reset_seed"]
            if (
                game["behavior_policy_sha256"] != behavior
                or game["teacher_labels"]
                or game["teacher_fraction"]
                or game["split"] != "train"
                or game["source"] != f"native-development/{seed}"
                or game["router_temperature"] != temperature
                or game.get("smooth_key_sampling", False) != smooth
                or game["sampler_draw_seed"] != spec["draw_seeds"][str(seed)]
            ):
                raise ValueError(
                    "PPO game provenance differs from its assigned policy and RNG"
                )
            raw = z.read(game["file"])
            if digest(raw) != game["sha256"]:
                raise ValueError("Trajectory checksum changed")
            with np.load(io.BytesIO(raw)) as d:
                s, t = d["timing_state"], d["control_time"]
                rows = d["key_rows"]
                ev = verified_events(
                    t, s[:, 4] * 9, d["time"], d["elapsed_ticks"], d["events"]
                )
                loss = ev[:, 1:5].sum(1)
                if (
                    len(np.flatnonzero(ev[:, 6:].any(1))) != 1
                    or bool(ev[:, 6].sum()) != game["won"]
                    or loss.sum() != sum(game["damage_taken"].values())
                    or ev[:, 0].sum() != game["damage"]
                ):
                    raise ValueError("Native rewards disagree with game score")
                if not np.array_equal(onsets(t, s), rows):
                    raise ValueError("Wrong primitive opportunity clock")
                expected, arm, h = actions(t, s, nets, cats, gating)
                record = {
                    k: game[k]
                    for k in (
                        "source",
                        "first_cleared",
                        "won",
                        "damage",
                        "damage_taken",
                        "game_speed",
                    )
                }
                outcomes.append(record)
                draws = d["key_draw"].astype("i8")
                uniform = d["key_uniform"]
                p_taken = d["key_probability"]
                v = np.zeros(len(rows))
                snaps = []
                for a in range(2):
                    chosen = np.flatnonzero(arm[rows] == a)
                    native = rows[chosen]
                    x = h[a][native]
                    eligible = available_keys(cats[a], s[native])
                    logits = nets[a].out(x)
                    p = distribution(
                        logits.double(),
                        torch.from_numpy(eligible),
                        temperature,
                        smooth=smooth,
                    ).numpy()
                    greedy = logits.argmax(1).numpy()
                    index = draws[chosen]
                    cp = p.cumsum(1)
                    local = np.arange(len(chosen))
                    lower = cp[local, index] - p[local, index]
                    if (
                        not np.array_equal(greedy, d["key_greedy"][chosen])
                        or not np.allclose(
                            p[local, index], p_taken[chosen], rtol=0.001, atol=2e-5
                        )
                        or (uniform[chosen] < lower - 2e-5).any()
                        or (uniform[chosen] >= cp[local, index] + 2e-5).any()
                    ):
                        raise ValueError(
                            "Frozen key logits do not reproduce real stochastic inputs"
                        )
                    expected[native] = cats[a][index]
                    v[chosen] = (
                        values[a].state_value(values[a].encoder(x)).flatten().numpy()
                    )
                    safe = np.flatnonzero(
                        (s[:, 23] > 0.5)
                        & (arm == a)
                        & ~np.isin(np.arange(len(t)), rows)
                    )[::25]
                    anchors[a].append(h[a][safe].numpy())
                    snaps.append((chosen, x.numpy(), eligible, index, p))
                trio = s[:, 23] > 0.5
                if not np.array_equal(expected[trio], d["controls"][trio]):
                    raise ValueError(
                        "Actual sampled or intervening ordinary keys differ from the actor"
                    )
                record.update(
                    decisions=len(rows),
                    verified_trio_inputs=int(trio.sum()),
                    sampled_alternatives=int((draws != d["key_greedy"]).sum()),
                )
                if not len(rows):
                    continue
                rewards, gamma, trace = option_rewards(t, loss, ev, rows, clean_cost)
                adv = advantages(rewards, gamma, trace, v)
                for a, (select, x, allowed, index, p) in enumerate(snaps):
                    if len(index):
                        groups[a].append(
                            (
                                x,
                                allowed,
                                index,
                                p,
                                p_taken[select],
                                adv[select],
                                adv[select] + v[select],
                            )
                        )
    if any(not g for g in groups):
        raise ValueError("No sampled opportunities for a neural controller")
    tensors = [
        [torch.from_numpy(np.concatenate(vals)) for vals in zip(*group, strict=True)]
        for group in groups
    ]
    return tensors, [torch.from_numpy(np.concatenate(a)) for a in anchors], outcomes


def optimize(
    raw, net, parent_net, critic, rollout, ordinary, temperature, *, smooth=False
):
    x, allowed, index, p0, taken, adv, target = rollout
    old = net.out(x).detach()
    p0 = p0.float()
    taken = taken.float()
    adv = (
        ((adv - adv.mean()) / adv.std(unbiased=False).clamp_min(0.05))
        .float()
        .clamp(-5, 5)
    )
    delta = nn.Linear(192, old.shape[1])
    nn.init.zeros_(delta.weight)
    nn.init.zeros_(delta.bias)
    optim = torch.optim.Adam(delta.parameters(), lr=0.0008)
    ix = torch.arange(len(x))
    old_safe = net.out(ordinary).detach()
    anchor = parent_net.out(ordinary).detach().softmax(1)
    steps = 0

    def prob():
        return distribution(old + delta(x), allowed, temperature, smooth=smooth)

    def kl(p):
        return (
            (p0 * (p0.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1).mean()
        )

    for _ in range(18):
        p = prob()
        ratio = p[ix, index] / taken
        obj = torch.minimum(ratio * adv, ratio.clamp(0.85, 1.15) * adv).mean()
        safe_loss = F.kl_div(
            (old_safe + delta(ordinary)).log_softmax(1), anchor, reduction="batchmean"
        )
        loss = -obj + 0.15 * safe_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(delta.parameters(), 0.5)
        before = copy.deepcopy(delta.state_dict())
        optim.step()
        with torch.no_grad():
            safe_drift = (
                ((old_safe + delta(ordinary)).argmax(1) != old_safe.argmax(1))
                .float()
                .mean()
            )
            if kl(prob()) > 0.035 or safe_drift > 0.015:
                delta.load_state_dict(before)
                break
        steps += 1
    vo = torch.optim.AdamW(
        [*critic.encoder.parameters(), *critic.state_value.parameters()], lr=0.001
    )
    for _ in range(32):
        value = critic.state_value(critic.encoder(x)).flatten()
        loss = F.smooth_l1_loss(value, target.float())
        vo.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        vo.step()
    with torch.no_grad():
        report = {
            "decisions": len(x),
            "accepted_steps": steps,
            "kl": float(kl(prob())),
            "changed_native_choices": int(
                ((old + delta(x)).argmax(1) != old.argmax(1)).sum()
            ),
            "ordinary_states": len(ordinary),
            "changed_ordinary_keys": int(
                ((old_safe + delta(ordinary)).argmax(1) != old_safe.argmax(1)).sum()
            ),
        }
    return (
        replace_head(raw, delta, 1.0),
        {"critic": critic.state_dict(), "delta": delta.state_dict(), "gain": 1.0},
        report,
    )


def train(
    data: Path,
    parent: Path,
    behavior: Path,
    learner: Path,
    output: Path,
    temperature: float,
    clean_cost: float = 0.0,
    smooth: bool = False,
):
    if output.exists() or not 0 < temperature < 1000 or not 0 <= clean_cost < 1000:
        raise ValueError("Invalid new PPO update")
    torch.set_num_threads(1)
    torch.manual_seed(23670)
    base = parent.read_bytes()
    deployed = behavior.read_bytes()
    nets, baseline, cats, gate = models(base, deployed)
    opening, route = phase_components(deployed)
    h, r, _ = option_components(route)
    ck = torch.load(learner, map_location="cpu", weights_only=True)
    values = [Outcome(len(c)) for c in cats]
    for v, key in zip(values, ("primary", "alternate"), strict=True):
        v.load_state_dict(ck[key]["critic"])
    games, ordinary, episodes = read_games(
        data,
        digest(deployed),
        nets,
        cats,
        gate,
        values,
        temperature,
        clean_cost,
        smooth=smooth,
    )
    fits = [
        optimize(raw, net, original, v, rows, safe, temperature, smooth=smooth)
        for raw, net, original, v, rows, safe in zip(
            (h, r), nets, baseline, values, games, ordinary, strict=True
        )
    ]
    leaves = [f[0] for f in fits]
    native = phase_checkpoint(
        opening,
        struct.pack("<3i", OPTION_MAGIC, *map(len, leaves))
        + b"".join(leaves)
        + gate.tobytes(),
    )
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        (stage / "world_iql.bin").write_bytes(native)
        torch.save(
            {"primary": fits[0][1], "alternate": fits[1][1]}, stage / "learner.pt"
        )
        report = {
            "method": (
                "Clipped on-policy PPO on the two actual primitive action heads "
                "with greedy deployment and elapsed-time GAE."
            ),
            "parent_sha256": digest(base),
            "behavior_sha256": digest(deployed),
            "rollout_sha256": digest(data.read_bytes()),
            "critic_input_sha256": digest(learner.read_bytes()),
            "policy_sha256": digest(native),
            "temperature": temperature,
            "clean_cost": clean_cost,
            "primary": fits[0][2],
            "alternate": fits[1][2],
            "episodes": episodes,
        }
        if smooth:
            report["smooth_key_sampling"] = True
        (stage / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "parent", "behavior", "learner", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--clean-cost", type=float, default=0.0)
    p.add_argument("--smooth", action="store_true")
    train(**vars(p.parse_args()))
