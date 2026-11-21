"""Learn short Knight trajectories and action-blind enemy box fields for native option selection."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_native_iql import attribute_events
from train_sisters_timing_iql import (
    TimingPolicy,
    motion_history,
    phase_components,
    save_native,
)

HORIZON = 8
ENCODED = 192
GEO = 6


def geometry(state: np.ndarray) -> np.ndarray:
    """Native position, velocity, invulnerability and SOUL in the recorded units."""
    return state[:, [0, 1, 2, 3, 9, 5]].astype("f4")


def rollout_starts(clock: np.ndarray, trio: np.ndarray, horizon: int) -> np.ndarray:
    # A rollout needs horizon applied keys and a factual successor observation.
    breaks = np.r_[0, np.cumsum((np.diff(clock) > 0.081) | (trio[:-1] != trio[1:]))]
    return np.flatnonzero(trio[:-horizon] & (breaks[:-horizon] == breaks[horizon:]))


class ActionWorld(nn.Module):
    def __init__(self, behavior_actions: int) -> None:
        super().__init__()
        size = ENCODED + GEO + behavior_actions
        self.core = nn.Sequential(
            nn.Linear(size + 12, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU()
        )
        self.change = nn.Linear(192, size)
        self.field = nn.Sequential(
            nn.Linear(size, 192), nn.SiLU(), nn.Linear(192, 2 * 2 * 9 * 11)
        )
        nn.init.zeros_(self.change.bias)
        nn.init.zeros_(self.change.weight)

    def forward(self, state: Tensor, keys: Tensor, dt: Tensor) -> Tensor:
        h = self.core(torch.cat((state, keys, dt.unsqueeze(-1) / 0.04), -1))
        return (state + self.change(h)).clamp(-12, 12)


def command_features(keys: Tensor) -> Tensor:
    bits = (keys[..., 2, None].long() >> torch.arange(9, device=keys.device)) & 1
    return torch.cat((keys[..., :2], bits), -1).float()


@torch.inference_mode()
def load_replay(
    path: Path,
    parent: TimingPolicy,
    historical: TimingPolicy,
    catalogue: Tensor,
    behavior_hash: str,
    device: torch.device,
) -> dict:
    games = {"train": [], "holdout": []}
    seen = set()
    with zipfile.ZipFile(path) as z:
        for game in json.loads(z.read("manifest.json")):
            source = game["source"]
            raw = z.read(game["file"])
            if (
                source in seen
                or not source.startswith("native-development/")
                or hashlib.sha256(raw).hexdigest() != game["sha256"]
            ):
                raise ValueError("Duplicate, unverified or non-development trajectory")
            seen.add(source)
            with np.load(io.BytesIO(raw)) as d:
                s, t, a = d["timing_state"], d["control_time"], d["controls"]
                trio = s[:, 23] > 0.5
                starts = rollout_starts(t, trio, HORIZON)
                if not len(starts):
                    continue
                h = torch.from_numpy(motion_history(t, s)).to(device)
                encoded = (
                    torch.cat([parent.encode(x) for x in h.split(768)]).cpu().numpy()
                )
                historical_scores = (
                    torch.cat([historical(x)[0] for x in h.split(768)]).cpu().numpy()
                )
                ev = attribute_events(
                    t,
                    d["time"],
                    d["elapsed_ticks"],
                    d["events"],
                    health_masks=s[:, 4] * 9,
                )
                actual = np.maximum(
                    -np.diff(np.rint(s[:, 4] * 9), append=np.rint(s[-1, 4] * 9)), 0
                )
                lost = np.maximum(actual, ev[:, 1:5].sum(1))
                health = np.rint(s[:, 4] * 9)
                lost = np.where(ev[:, 7] > 0, np.maximum(lost, health), lost)
                boxes = np.stack(
                    [
                        np.column_stack(
                            (
                                10 * (s[:, 0] + s[:, k + 1]),
                                8 * (s[:, 1] + s[:, k + 2]),
                                2.5 * s[:, k + 7],
                                2.5 * s[:, k + 8],
                                (s[:, k] > 0.5) & (s[:, k + 9] > 0.5),
                            )
                        )
                        for k in (24, 60, 96, 132, 146)
                    ],
                    axis=1,
                ).astype("f4")
                # Fixed-time enemy field labels use only future observations on the same continuous tape.
                clock = t.astype("f8")
                field_rows = np.stack(
                    [
                        np.searchsorted(clock, clock + h - 1e-5).clip(max=len(t) - 1)
                        for h in (0.16, 0.32)
                    ],
                    1,
                )
                observed = clock[field_rows] - clock[:, None]
                field_known = np.abs(observed - np.array([0.16, 0.32])) < 0.041
                segments = np.cumsum(
                    np.r_[0, (np.diff(t) > 0.081) | (trio[1:] != trio[:-1])]
                )
                field_known &= segments[field_rows] == segments[:, None]
                flags = s[:, [7, 8, 13, 11, 5, 164, 165, 166, 168, 9]]
                columns = (
                    np.c_[encoded, geometry(s), historical_scores],
                    a.astype("i8"),
                    np.r_[np.diff(t), 0.04].astype("f4"),
                    np.column_stack([ev[:, 0] / 21, lost]).astype("f4"),
                    boxes,
                    flags,
                    field_rows.astype("i8"),
                    field_known,
                )
                # Record the exact randomized pair, for blind forecasts from one pre-draw state.
                draws = []
                if "choice_hold_stop" in d and "choice_scores" not in d:
                    if game["behavior_policy_sha256"] != behavior_hash:
                        raise ValueError(
                            "Missing the policy that produced these randomized actions"
                        )
                    issued = d["choice_pairs"][
                        np.arange(len(d["choice_rows"])), d["choice_draw"].astype(int)
                    ]
                    if not np.array_equal(
                        catalogue[issued].cpu().numpy(), a[d["choice_rows"]]
                    ):
                        raise ValueError(
                            "Historical catalogue differs from the issued keys"
                        )
                    for row, stop, pair, arm in zip(
                        d["choice_rows"],
                        d["choice_hold_stop"],
                        d["choice_pairs"],
                        d["choice_draw"],
                        strict=True,
                    ):
                        if (
                            stop < len(t)
                            and stop - row == 4
                            and abs(float(t[stop] - t[row]) - 0.16) < 0.012
                        ) and np.all(np.diff(t[row : stop + 1]) < 0.051):
                            draws.append(
                                (
                                    int(row),
                                    int(stop),
                                    int(pair[0]),
                                    int(pair[1]),
                                    int(arm),
                                )
                            )
                games[game["split"]].append(
                    (source, columns, starts, draws, "teacher_controls" not in d)
                )
    packed = {}
    for split, items in games.items():
        offsets = np.r_[0, np.cumsum([len(x[1][0]) for x in items])]
        tensors = [
            torch.from_numpy(np.concatenate([g[1][k] for g in items])).to(device)
            for k in range(8)
        ]
        tensors[6] += torch.as_tensor(
            np.repeat(offsets[:-1], np.diff(offsets)), device=device
        )[:, None]
        starts = np.concatenate([off + x[2] for x, off in zip(items, offsets)])
        per = np.concatenate([np.full(len(x[2]), 1 / len(x[2])) for x in items])
        per /= sum(per)
        ids = np.concatenate([np.full(len(x[2]), i) for i, x in enumerate(items)])
        draws = [
            (x[0], int(off) + a, int(off) + b, i, j, arm)
            for x, off in zip(items, offsets)
            for a, b, i, j, arm in x[3]
        ]
        packed[split] = {
            "data": tensors,
            "starts": starts,
            "probability": per,
            "game_ids": ids,
            "games": [g[0] for g in items],
            "free": [g[4] for g in items],
            "draws": draws,
        }
    global_rows = torch.as_tensor(packed["train"]["starts"], device=device)
    train = packed["train"]["data"][0][global_rows]
    mean, scale = train.mean(0), train.std(0).clamp_min(0.05)
    for bag in packed.values():
        bag["data"][0] = (bag["data"][0] - mean) / scale
    for group in ("train", "holdout"):
        packed[group]["mean"], packed[group]["scale"] = mean, scale
    packed["mean"], packed["scale"] = mean, scale
    return packed


def fit_world(net: ActionWorld, packed: dict, updates: int, seed: int) -> None:
    data = packed["data"]
    z, keys, dt = data[:3]
    commands = command_features(keys)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0008)
    for step in range(updates):
        starts = torch.as_tensor(
            rng.choice(packed["starts"], 192, p=packed["probability"]), device=z.device
        )
        state = z[starts]
        loss = torch.zeros((), device=z.device)
        nearby, mask = observed_fields(packed, starts, z)
        prediction_field = net.field(state).reshape(-1, 2, 2, 9, 11).sigmoid()
        field_loss = (
            (prediction_field - nearby).square() * (1 + 3 * nearby) * mask
        ).sum() / (mask.sum() * 198).clamp_min(1.0)
        for i in range(HORIZON):
            j = starts + i
            prediction = net(state, commands[j], dt[j])
            nxt = z[j + 1].clamp(-12, 12)
            displacement = F.smooth_l1_loss(
                prediction[:, :ENCODED], nxt[:, :ENCODED], beta=0.15
            )
            positions = F.smooth_l1_loss(
                prediction[:, ENCODED : ENCODED + 6],
                nxt[:, ENCODED : ENCODED + 6],
                beta=0.15,
            )
            behavior_loss = F.smooth_l1_loss(
                prediction[:, ENCODED + GEO :], nxt[:, ENCODED + GEO :], beta=0.15
            )
            loss = loss + displacement + 2 * positions + 0.5 * behavior_loss
            state = prediction
        loss = loss + HORIZON * field_loss
        opt.zero_grad(set_to_none=True)
        (loss / HORIZON).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        if step % 500 == 0:
            print(
                "WORLD",
                seed,
                step,
                round(float(loss.detach()) / HORIZON, 4),
                flush=True,
            )


def native_scores(
    actor: TimingPolicy, z: Tensor, mean: Tensor, scale: Tensor
) -> Tensor:
    restored = z[..., :ENCODED] * scale[:ENCODED] + mean[:ENCODED]
    return actor.heads(restored)[0]


def observed_fields(bag: dict, rows: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    boxes, times, known = (bag["data"][i] for i in (4, 6, 7))
    future = boxes[times[rows]]
    center = (
        z[rows, ENCODED : ENCODED + 2] * bag["scale"][ENCODED : ENCODED + 2]
        + bag["mean"][ENCODED : ENCODED + 2]
    )
    x = center[:, 0, None] * 10 + torch.linspace(-6.25, 6.25, 11, device=z.device)
    y = center[:, 1, None] * 8 + torch.linspace(-5.0, 5.0, 9, device=z.device)
    dx = (future[..., 0, None] - x[:, None, None, :]).abs() - future[..., 2, None]
    dy = (future[..., 1, None] - y[:, None, None, :]).abs() - future[..., 3, None]
    proximity = (
        -torch.maximum(dx.unsqueeze(-2), dy.unsqueeze(-1)).clamp_min(0) / 1.4
    ).exp()
    proximity *= future[..., 4, None, None]
    field = torch.stack((proximity[:, :, :3].amax(2), proximity[:, :, 3:].amax(2)), 2)
    return field, known[rows, :, None, None, None]


def compound_pair(
    scores: Tensor, cat: Tensor, flags: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """The same maneuver classes and availability tests as the native compound trial."""
    b = cat[:, 2]
    group = (
        ((b & 3) != 0).long()
        + 2 * ((b & 4) != 0)
        + 4 * (((b & 16) != 0) & (cat[:, 1] < 0))
    )
    ground, jump, dash, cast, soul, jump_down, held, dash_down, cast_down, invul = (
        flags.T
    )
    primary = scores.argmax(-1)
    allowed = group[None] != group[primary, None]
    allowed &= ~(
        ((b & 1) != 0) & ((jump_down > 0.5) | ((ground < 0.5) & (jump < 0.5)))[:, None]
    )
    allowed &= ~(((b & 3) == 2) & (held < 0.5)[:, None])
    allowed &= ~(((b & 4) != 0) & ((dash < 0.5) | (dash_down > 0.5))[:, None])
    allowed &= ~(
        ((b & 16) != 0) & ((cast < 0.5) | (soul < 33 / 99) | (cast_down > 0.5))[:, None]
    )
    return (
        primary,
        scores.masked_fill(~allowed, -torch.inf).argmax(-1),
        allowed.any(-1) & (invul <= 0.5),
    )


def continuation(scores: Tensor, cat: Tensor, onset: Tensor, mode: str) -> Tensor:
    greedy = scores.argmax(-1)
    if mode == "free":
        return greedy
    allowed = cat[None, :, 0] == cat[onset, 0, None]
    if mode == "compound":
        b = cat[:, 2]
        allowed &= ((b[None] ^ b[onset, None]) & 2) == 0
        allowed &= (b[None] & 21 & ~b[greedy, None]) == 0
    elif mode != "horizontal":
        raise ValueError("Unknown option continuation")
    return torch.where(
        allowed.any(1), scores.masked_fill(~allowed, -torch.inf).argmax(-1), greedy
    )


@torch.no_grad()
def imagined_path(
    world: ActionWorld,
    actor: TimingPolicy,
    cat: Tensor,
    bag: dict,
    initial: Tensor,
    first: Tensor,
    mode: str,
    field: Tensor,
) -> Tensor:
    mean, scale = bag["mean"], bag["scale"]
    z = initial
    command = command_features(cat)
    dt = z.new_full((len(z),), 0.04)
    original = z[:, ENCODED:] * scale[ENCODED:] + mean[ENCODED:]
    result = []
    for step in range(8):
        if step:
            # A descent assignment holds only through 0.16s; compound holds through 0.32s.
            scores = (
                z[:, ENCODED + GEO :] * scale[ENCODED + GEO :] + mean[ENCODED + GEO :]
                if mode == "horizontal"
                else native_scores(actor, z, mean, scale)
            )
            key = continuation(
                scores,
                cat,
                first,
                "free" if mode == "horizontal" and step >= 4 else mode,
            )
        else:
            key = first
        z = world(z, command[key], dt)
        if step in (3, 7):
            state = z[:, ENCODED:] * scale[ENCODED:] + mean[ENCODED:]
            xy = (state[:, :2] - original[:, :2]) * z.new_tensor([10.0, 8.0])
            grid = (xy / z.new_tensor([6.25, 5.0])).view(-1, 1, 1, 2)
            near = F.grid_sample(
                field[:, int(step == 7)],
                grid,
                align_corners=True,
                padding_mode="border",
            ).flatten(1)
            vulnerable = 1 - state[:, 4:5].clamp(0, 1)
            result += [xy, vulnerable, near * vulnerable]
    return torch.cat(result, 1)


@torch.no_grad()
def contrasts(
    worlds: list[ActionWorld],
    actor: TimingPolicy,
    cat: Tensor,
    bag: dict,
    z: Tensor,
    primary: Tensor,
    alternate: Tensor,
    descent: bool,
) -> tuple[Tensor, Tensor]:
    outputs = []
    for world in worlds:
        field = world.field(z).view(-1, 2, 2, 9, 11).sigmoid()
        a = imagined_path(
            world,
            actor,
            cat,
            bag,
            z,
            primary,
            "horizontal" if descent else "free",
            field,
        )
        b = imagined_path(
            world,
            actor,
            cat,
            bag,
            z,
            alternate,
            "horizontal" if descent else "compound",
            field,
        )
        outputs.append(torch.stack((a, b)))
    return torch.stack(outputs).mean(0).unbind(0)


@torch.no_grad()
def check_draws(
    worlds: list[ActionWorld], actor: TimingPolicy, cat: Tensor, bag: dict
) -> dict:
    z = bag["data"][0]
    report = {}
    for name in sorted({g[0] for g in bag["draws"]}):
        rows = np.array([g[1:] for g in bag["draws"] if g[0] == name])
        i, j, a, b, arm = (torch.as_tensor(col, device=z.device) for col in rows.T)
        primary, alternate = contrasts(worlds, actor, cat, bag, z[i], a, b, True)
        both = torch.stack((primary, alternate), 1)
        k = torch.arange(len(i), device=z.device)
        dx = (z[j, ENCODED] - z[i, ENCODED]) * bag["scale"][ENCODED] * 10
        actual = both[k, arm]
        swapped = both[k, 1 - arm]
        lost = torch.stack([bag["data"][3][i + n, 1] for n in range(4)]).sum(0)
        # The assigned arm is random. This estimate compares a rule choosing less
        # predicted contact with always taking the original arm, on independent games.
        select = alternate[:, 3:5].amax(1) < primary[:, 3:5].amax(1)
        report[name] = {
            "draws": len(i),
            "actual_native_masks_lost": int(lost.sum()),
            "predicted_safer_alternate": int(select.sum()),
            "assigned_x_error": float((actual[:, 0] - dx).abs().mean()),
            "swapped_x_error": float((swapped[:, 0] - dx).abs().mean()),
            "ips_mask_change": float((select * (2 * arm - 1) * 2 * lost).mean()),
        }
    return report


@torch.no_grad()
def projection(
    worlds: list[ActionWorld],
    actor: TimingPolicy,
    cat: Tensor,
    bag: dict,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(sample_count, len(bag["starts"]))
    indices = rng.choice(bag["starts"], count, replace=True, p=bag["probability"])
    xx, yy = [], []
    for ix in np.array_split(indices, max(1, (count + 255) // 256)):
        z = bag["data"][0][ix]
        a, b, ok = compound_pair(
            native_scores(actor, z, bag["mean"], bag["scale"]), cat, bag["data"][5][ix]
        )
        if not ok.any():
            continue
        u, v = contrasts(worlds, actor, cat, bag, z[ok], a[ok], b[ok], False)
        xx.append(
            (z[ok, :ENCODED] * bag["scale"][:ENCODED] + bag["mean"][:ENCODED])
            .cpu()
            .numpy()
        )
        yy.append((v - u).cpu().numpy())
    x, y = np.vstack(xx).astype("f8"), np.vstack(yy).astype("f8")
    x -= x.mean(0)
    y -= y.mean(0)
    std = x.std(0).clip(0.05)
    xs = x / std
    target = y / y.std(0).clip(0.01)
    slope = (
        np.linalg.solve(xs.T @ xs + np.eye(ENCODED) * len(x) * 0.02, xs.T @ target)
        / std[:, None]
    )
    return slope.astype("<f4")


def train(
    archive: Path,
    parent: Path,
    output: Path,
    updates: int,
    projection_samples: int,
    seed: int,
    behavior: Path,
    behavior_policy: Path,
) -> None:
    if output.exists() or min(updates, projection_samples) < 1:
        raise ValueError("Expected a fresh output and a positive training budget")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(parent, weights_only=True, map_location="cpu")
    cat = ck["catalogue"].long().to(device)
    actor = TimingPolicy(cat, True).to(device).eval().requires_grad_(False)
    actor.load_state_dict(ck["actor"])
    historical_ck = torch.load(behavior, weights_only=True, map_location="cpu")
    historical_cat = historical_ck["catalogue"].long()
    historical = TimingPolicy(historical_cat, True).eval().requires_grad_(False)
    historical.load_state_dict(historical_ck["actor"])
    deployed = behavior_policy.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        leaf = Path(tmp) / "historical.bin"
        save_native(historical, historical_cat.numpy(), leaf)
        reference = phase_components(deployed)[1]
        exported = leaf.read_bytes()
        tail = len(reference) - len(historical_cat) * (ENCODED + 1) * 4
        if len(reference) != len(exported) or exported[:tail] != reference[:tail]:
            raise ValueError("Historical catalogue or encoder differs from the game")
        native_head = np.frombuffer(reference[tail:], "<f4")
        if not np.allclose(
            native_head, np.frombuffer(exported[tail:], "<f4"), rtol=1e-6, atol=1e-6
        ):
            raise ValueError("Historical action scores differ from the game")
        # Training combines factor scores in PyTorch. The game reads the folded head.
        # Use the archived game coefficients, including its exact floating point rounding.
        weights = torch.from_numpy(native_head.copy())
        historical.out.weight.copy_(
            weights[: -len(historical_cat)].reshape_as(historical.out.weight)
        )
        historical.out.bias.copy_(weights[-len(historical_cat) :])
        historical.factors.weight.zero_()
        historical.factors.bias.zero_()
    behavior_hash = hashlib.sha256(deployed).hexdigest()
    historical_cat = historical_cat.to(device)
    data = load_replay(
        archive, actor, historical.to(device), historical_cat, behavior_hash, device
    )
    print(
        "REPLAY", len(data["train"]["games"]), len(data["holdout"]["games"]), flush=True
    )
    worlds = []
    for member in (seed, seed + 1):
        torch.manual_seed(member)
        model = ActionWorld(len(historical_cat)).to(device)
        fit_world(model, data["train"], updates, member)
        worlds.append(model.eval())
    held = check_draws(worlds, actor, historical_cat, data["holdout"])
    projected = projection(worlds, actor, cat, data["train"], projection_samples, seed)
    if not np.isfinite(projected).all():
        raise ValueError("Nonfinite projection")
    output.mkdir(parents=True)
    np.save(output / "risk-projection.npy", projected, allow_pickle=False)
    torch.save(
        {
            "worlds": [w.cpu().state_dict() for w in worlds],
            "mean": data["mean"].cpu(),
            "scale": data["scale"].cpu(),
        },
        output / "worlds.pt",
    )
    report = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "learner_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "historical_behavior_sha256": behavior_hash,
        "updates_per_world": updates,
        "projection_samples": projection_samples,
        "seed": seed,
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "training_fights": len(data["train"]["games"]),
        "heldout_fights": len(data["holdout"]["games"]),
        "randomized_descent": held,
        "projection_sha256": hashlib.sha256(
            (output / "risk-projection.npy").read_bytes()
        ).hexdigest(),
        "scope": "Proximity to observed collision boxes along imagined native inputs. Independent randomized option returns determine the policy update.",
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FIT", len(held), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "parent", "output", "behavior", "behavior-policy"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1800)
    parser.add_argument("--projection-samples", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=96331)
    train(**vars(parser.parse_args()))
