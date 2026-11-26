"""Fit an arena policy with entity timings, teacher labels and offline game returns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from sisters_future_occupancy import TARGET_SHAPE, future_occupancy
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS, attribute_events, quantile_loss

WIDTH = 192
HISTORY_LAGS = (0.16, 0.48)
HISTORY_BLOCK = WIDTH + 1
FOES, DISCS, CONTROL = 24, 132, 160
MIRROR = [0, 2, 6, CONTROL]
for offset in (24, 60, 96):
    MIRROR.extend(offset + j for j in (1, 3, 5, 11, 12, 15, 18))
for offset in (132, 146):
    MIRROR.extend(offset + j for j in (1, 3, 5, 11, 12))
# Predict measured physical variables only. History and key ages are deterministic.
PHYSICAL = [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20]
for offset in (24, 60, 96, 132, 146):
    PHYSICAL.extend(offset + j for j in range(10))


class TimingPolicy(nn.Module):
    def __init__(self, catalogue: Tensor, history: bool = False) -> None:
        super().__init__()
        catalogue = catalogue.long()
        self.boss = nn.Linear(36, 48)
        self.disc = nn.Linear(14, 24)
        self.past = nn.Linear(HISTORY_BLOCK, 32) if history else None
        self.dense = nn.Sequential(
            nn.Linear(WIDTH + 72 + (64 if history else 0), 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
        )
        self.out = nn.Linear(192, len(catalogue))
        # Per-factor supervision shares the same representation; these heads are
        # also folded into the exported joint logits.
        self.factors = nn.Linear(192, 15)
        bits = (
            (catalogue[:, 2, None] >> torch.arange(9, device=catalogue.device)) & 1
        ).float()
        basis = torch.cat(
            [
                F.one_hot(catalogue[:, 0] + 1, 3),
                F.one_hot(catalogue[:, 1] + 1, 3),
                bits,
            ],
            dim=1,
        )
        self.register_buffer("basis", basis.float())

    @staticmethod
    def pool(slots: Tensor, layer: nn.Linear) -> Tensor:
        return (F.relu(layer(slots)) * slots[:, :, :1]).amax(1)

    def encode(self, state: Tensor) -> Tensor:
        bosses = state[:, FOES:DISCS].reshape(-1, 3, 36)
        discs = state[:, DISCS:CONTROL].reshape(-1, 2, 14)
        bosses = bosses * bosses[:, :, :1]
        discs = discs * discs[:, :, :1]
        boss = self.pool(bosses, self.boss)
        disc = self.pool(discs, self.disc)
        features = [
            state[:, :FOES],
            bosses.flatten(1),
            discs.flatten(1),
            state[:, CONTROL:WIDTH],
            boss,
            disc,
        ]
        if self.past is not None:
            past = state[:, WIDTH:].reshape(-1, HISTORY_BLOCK)
            foes = past[:, FOES:DISCS].reshape(-1, 3, 36)
            blades = past[:, DISCS:CONTROL].reshape(-1, 2, 14)
            measured = torch.cat(
                (
                    past[:, :FOES],
                    (foes * foes[:, :, :1]).flatten(1),
                    (blades * blades[:, :, :1]).flatten(1),
                    past[:, CONTROL:],
                ),
                1,
            )
            embedded = F.relu(self.past(measured)) * (past[:, -1:] > 0)
            features.append(embedded.reshape(len(state), -1))
        return self.dense(torch.cat(features, 1))

    def heads(self, x: Tensor) -> tuple[Tensor, Tensor]:
        factors = self.factors(x)
        return self.out(x) + factors @ self.basis.T, factors

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.heads(self.encode(state))


def save_native(actor: TimingPolicy, catalogue: np.ndarray, path: Path) -> None:
    def array(t: Tensor) -> bytes:
        return t.detach().cpu().numpy().astype("<f4").tobytes()

    with path.open("wb") as out:
        magic = 0x344C5149 if actor.past is not None else 0x324C5149
        out.write(struct.pack("<5i", magic, WIDTH, 256, 192, len(catalogue)))
        out.write(catalogue.astype("<i4").tobytes())
        for layer in (
            actor.boss,
            actor.disc,
            actor.past,
            actor.dense[0],
            actor.dense[2],
        ):
            if layer is None:
                continue
            out.write(array(layer.weight))
            out.write(array(layer.bias))
        out.write(array(actor.out.weight + actor.basis @ actor.factors.weight))
        out.write(array(actor.out.bias + actor.basis @ actor.factors.bias))


def teacher_history_age(
    times: np.ndarray, applied: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Seconds since the last earlier input where the teacher did not control the game."""
    different = np.any(applied != labels, axis=1)
    last_mismatch = np.maximum.accumulate(np.where(different, times, -np.inf))
    return times - np.r_[-np.inf, last_mismatch[:-1]]


def motion_history(times: np.ndarray, state: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Attach the last observation available at each lag on this uninterrupted tape."""
    clock = times.astype("f8")
    all_rows = np.arange(len(clock))
    if rows is None:
        rows = all_rows
    segment = np.maximum.accumulate(
        np.where(np.r_[True, np.diff(clock) > 0.24001], all_rows, 0)
    )
    blocks = [state[rows]]
    for lag in HISTORY_LAGS:
        j = np.searchsorted(clock, clock[rows] - lag + 0.00001, side="right") - 1
        age = clock[rows] - clock[j.clip(0)]
        known = (j >= segment[rows]) & (age <= lag + 0.24001)
        block = np.zeros((len(rows), HISTORY_BLOCK), dtype="f4")
        block[known, :WIDTH] = state[j[known]]
        block[known, -1] = age[known]
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


MOVEMENT_HORIZONS = (0.04, 0.12, 0.28)


def future_movements(
    clock: np.ndarray, actions: np.ndarray, teacher: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """Targets for a training-only head, measured on uninterrupted matching controls."""
    t = clock.astype("f8")
    rows = np.arange(len(t))
    mismatch = np.r_[0, np.cumsum((actions != teacher).any(1))]
    segment = np.r_[0, np.cumsum((np.diff(t) > .081) | (phase[1:] != phase[:-1]))]
    flags = teacher[:, 2]
    movement = (teacher[:, 0] + 1) * 4 + 2 * ((flags & 3) != 0) + ((flags & 4) != 0)
    targets = np.full((len(t), len(MOVEMENT_HORIZONS)), -1, dtype="i1")
    for k, seconds in enumerate(MOVEMENT_HORIZONS):
        desired = t + seconds
        upper = np.searchsorted(t, desired).clip(0, len(t) - 1)
        lower = (upper - 1).clip(0)
        at = np.where(abs(t[upper] - desired) < abs(t[lower] - desired), upper, lower)
        valid = ((at > rows) & (abs(t[at] - desired) <= .021) &
                 (segment[at] == segment) & (mismatch[at + 1] == mismatch[rows]))
        targets[valid, k] = movement[at[valid]]
    return targets


def measured_returns(
    times: np.ndarray,
    rewards: np.ndarray,
    discounts: np.ndarray,
    valid: np.ndarray,
    seconds: float,
    controller_boundaries: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate observed rewards on one tape. Stop at death, gaps or the horizon.

    The returned index selects the successor of the last included transition.
    A gap ends a window before the gap and preserves bootstrapping at its start.
    """
    n = len(times)
    rows = np.arange(n)
    first_missing = np.minimum.accumulate(np.where(valid, n, rows)[::-1])[::-1]
    end = np.minimum(
        np.searchsorted(times.astype("f8"), times.astype("f8") + seconds), first_missing
    )
    if controller_boundaries is not None:
        # Bootstrap the state reached at handoff before counting any rewards
        # obtained by the newly drawn controller.
        cuts = np.asarray(controller_boundaries)
        if (
            cuts.ndim != 1
            or cuts.dtype.kind not in "iu"
            or (np.diff(cuts.astype("i8")) <= 0).any()
        ):
            raise ValueError("Invalid controller boundaries")
        if len(cuts) and (cuts[0] < 0 or cuts[-1] >= n):
            raise ValueError("Controller boundary outside trajectory")
        end = np.minimum(end, np.r_[cuts, n][np.searchsorted(cuts, rows, side="right")])
    total = np.zeros(n, dtype="f4")
    remaining = np.ones(n, dtype="f4")
    last = rows.copy()
    for k in range(int(np.max(end - rows, initial=0))):
        active = (rows + k < end) & (remaining > 0)
        j = rows[active] + k
        total[active] += remaining[active] * rewards[j]
        remaining[active] *= discounts[j]
        last[active] = j
    return total, remaining, last


def opening_rewards(
    state: np.ndarray, reward: np.ndarray, terminal: np.ndarray, reserve: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """End the opening subtask on the first observed trio and value resources carried in."""
    reward, terminal = reward.copy(), terminal.copy()
    switches = np.flatnonzero(state[:, 23] > 0.5)
    stop = int(switches[0]) if len(switches) else len(state)
    pairs = min(stop, len(state) - 1)
    # Actual net changes count, including health restored before the transition.
    # Taking damage and rebuilding soul by attacking remain visible to the learner.
    changes = np.diff(state[: pairs + 1, 4:6], axis=0)
    reward[:pairs] += 1.25 * 9 * changes[:, 0] + 1.5 * changes[:, 1]
    if 0 < stop < len(state):
        entry = state[stop]
        reward[stop - 1] += 5 + 0.5 * 9 * entry[4] + 1.5 * entry[5]
        if reserve:
            charges = np.floor((99 * np.clip(entry[5], 0, 1) + 1e-4) / 33)
            reward[stop - 1] += 3 * charges
        terminal[stop - 1] = True
    return reward, terminal


def no_hit_rewards(
    reward: np.ndarray,
    events: np.ndarray,
    state: np.ndarray,
    cost: float,
    *,
    opening: bool = False,
) -> np.ndarray:
    """Penalize real damage and reward an observed clean subtask completion."""
    masks = events[:, 1:5].sum(axis=1)
    adjusted = reward - cost * masks
    hits = np.flatnonzero(masks > 0)
    if len(hits):
        adjusted[hits[0]] -= cost
    clean = (masks.cumsum() == 0) & np.logical_and.accumulate(state[:, 4] >= 1)
    if opening:
        trio = np.flatnonzero(state[:, 23] > 0.5)
        if len(trio) and trio[0] > 0:
            entry = int(trio[0])
            if clean[entry - 1] and state[entry, 4] >= 1 and not events[:entry, 7].any():
                adjusted[entry - 1] += 2 * cost
    else:
        adjusted[(events[:, 6] > 0) & (events[:, 7] == 0) & clean] += 2 * cost
    return adjusted


def eligible_teacher_fight(won: bool, times: np.ndarray, minimum_hz: float) -> bool:
    if minimum_hz == 0:
        return True
    return bool(won and len(times) > 1 and len(times) - 1 >= minimum_hz * (times[-1] - times[0]))


def early_trio_rows(times: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Mark the first ten game seconds after the trio is first observed."""
    trio = state[:, 23] > 0.5
    if not trio.any():
        return trio
    return trio & (times < float(times[trio.argmax()]) + 10 - 1e-5)


def build_data(
    directory: Path,
    device: str,
    mirror: bool = True,
    forecast: bool = False,
    trio_only: bool = False,
    return_seconds: float = 0.0,
    history: bool = False,
    plans: bool = False,
    opening_only: bool = False,
    no_hit_cost: float = 0.0,
    teacher_win_min_hz: float = 0.0,
    future_keys: bool = False,
    terminal_advantage: bool = False,
    executed_teacher: bool = False,
    reserve_opening: bool = False,
    entry_replay: bool = False,
) -> tuple[dict, np.ndarray, dict]:
    games = json.loads((directory / "manifest.json").read_bytes())
    if len({g["source"] for g in games}) != len(games) or any(
        not g["source"].startswith("native-development/")
        or g["split"] not in ("train", "holdout")
        for g in games
    ):
        raise ValueError("Expected unique development games with explicit splits")
    parts: dict[str, list] = {"train": [], "holdout": []}
    fields: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    windows: dict[str, list[tuple]] = {"train": [], "holdout": []}
    options: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    movements: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    outcomes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"train": [], "holdout": []}
    entry_rows: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    stats: dict[str, dict] = {}
    mirror_axes = list(MIRROR)
    if history:
        for k in range(len(HISTORY_LAGS)):
            mirror_axes.extend(WIDTH + k * HISTORY_BLOCK + j for j in MIRROR)
    for game in games:
        path = directory / game["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != game["sha256"]:
            raise ValueError(f"Changed tape {game['source']}")
        with np.load(path, allow_pickle=False) as f:
            s = f["timing_state"].astype("f4")
            t, act = f["control_time"], f["controls"]
            labeled = game.get("teacher_labels", True)
            neural_trial = bool(game.get("alternate_policy_sha256"))
            if neural_trial and (
                labeled
                or not game.get("student_policy_sha256")
                or game.get("candidate_probability") != 0.5
                or game["alternate_policy_sha256"] == game["student_policy_sha256"]
                or "block_starts" not in f
                or "candidate_draw" not in f
                or f["candidate_draw"].dtype != np.bool_
                or f["candidate_draw"].shape != t.shape
            ):
                raise ValueError(
                    f"Invalid neural controller record in {game['source']}"
                )
            if labeled != ("teacher_controls" in f) or (
                not labeled
                and (
                    not (game.get("behavior_policy_sha256") or neural_trial)
                    or "plan_costs" in f
                )
            ):
                raise ValueError(f"Inconsistent teacher labels in {game['source']}")
            acted = None
            if (executed_teacher or reserve_opening) and labeled:
                if "teacher_acted" not in f:
                    raise ValueError("Executed-teacher learning requires recorded controller assignments")
                acted = f["teacher_acted"]
                if acted.dtype != np.bool_ or acted.shape != t.shape or not np.array_equal(
                    act[acted], f["teacher_controls"][acted]
                ):
                    raise ValueError("Executed teacher labels differ from the native controls")
            boundaries = None
            if neural_trial:
                boundaries = f["block_starts"]
            elif "teacher_acted" in f:
                boundaries = np.flatnonzero(np.diff(f["teacher_acted"])) + 1
            elif "choice_rows" in f:
                boundaries = f["choice_rows"]
            elif labeled and not np.array_equal(act, f["teacher_controls"]):
                # Historical mixed tapes without assignments cannot identify a handoff
                # when both controllers happened to choose the same input.
                boundaries = np.arange(len(t)) if return_seconds else None
            teacher = (
                f["teacher_controls"] if labeled else np.full_like(act, -1, dtype="i2")
            )
            if (
                s.shape != (len(t), WIDTH)
                or not np.isfinite(s).all()
                or (np.diff(t) <= 0).any()
            ):
                raise ValueError("Invalid native state or timestamp")
            ev = attribute_events(
                t, f["time"], f["elapsed_ticks"], f["events"], health_masks=s[:, 4] * 9
            )
            if plans:
                from sisters_plan_labels import PLAN_ACTIONS, plan_preferences

                if ("plan_costs" in f) != ("plan_choice" in f):
                    raise ValueError("Incomplete teacher plan tape")
                preferences = (
                    plan_preferences(f["plan_costs"], f["plan_choice"], teacher)
                    if "plan_costs" in f
                    else np.zeros((len(t), PLAN_ACTIONS), dtype="f4")
                )
        terminal = ev[:, 6:].any(1)
        limit = np.flatnonzero(terminal)
        n = int(limit[0]) + 1 if len(limit) else len(t)
        s, t, act, teacher, ev, terminal = [
            v[:n] for v in (s, t, act, teacher, ev, terminal)
        ]
        reward = ev @ REWARD_WEIGHTS
        if no_hit_cost:
            reward = no_hit_rewards(reward, ev, s, no_hit_cost, opening=opening_only)
        if opening_only:
            reward, terminal = opening_rewards(s, reward, terminal, reserve_opening)
        dt = np.r_[np.diff(t), 0.04].astype("f4")
        # A 40-game-second time constant lets rewards at the handoff influence
        # earlier casts in the roughly one-minute opening subtask.
        d = (np.exp(-dt / 40) if reserve_opening else 0.995 ** (dt / 0.04)) * (~terminal)
        good = (dt <= 0.24) & (np.r_[np.ones(n - 1, bool), False] | terminal)
        if history:
            s = motion_history(t, s)
        next_state = np.r_[s[1:], s[-1:]]
        agreement = teacher_history_age(t, act, teacher) if labeled else np.full(len(t), -1, "f4")
        if acted is not None:
            agreement[~acted[:n]] = -np.inf
        if reserve_opening:
            entry = np.flatnonzero(s[:, 23] > 0.5)
            if not len(entry) or s[entry[0], 4] < .999 or s[entry[0], 5] * 99 < 66 - 1e-4:
                # The actual transitions, costs and terminal still train both critics.
                agreement.fill(-np.inf)
        selected = s[:, 23] > 0.5 if trio_only else slice(None)
        if opening_only:
            selected = s[:, 23] <= 0.5
        if (trio_only or opening_only) and not selected.any():
            continue
        if terminal_advantage:
            from sisters_episode_returns import outcome_samples

            context, terminal_return = outcome_samples(t, s, ev, bool(game["won"]))
            outcomes[game["split"]].append((context[selected], terminal_return[selected]))
        if entry_replay:
            entry_rows[game["split"]].append(early_trio_rows(t, s)[selected])
        if teacher_win_min_hz and not eligible_teacher_fight(game["won"], t[selected], teacher_win_min_hz):
            # Only the teacher/plan labels are excluded. Actual rewards and
            # actions from lost fights still train the world, value and critic.
            agreement.fill(-np.inf)
        if forecast:
            fields[game["split"]].append(future_occupancy(t, s[:, :WIDTH])[selected])
        if plans:
            options[game["split"]].append(preferences[:n][selected])
        if future_keys:
            targets = (future_movements(t, act, teacher, s[:, 23]) if labeled
                       else np.full((len(t), len(MOVEMENT_HORIZONS)), -1, "i1"))
            movements[game["split"]].append(targets[selected])
        if return_seconds:
            total, remaining, last = measured_returns(
                t, reward, d, good, return_seconds,
                boundaries[boundaries < n] if boundaries is not None else None,
            )
            windows[game["split"]].append(
                (total[selected], remaining[selected], next_state[last[selected]])
            )
        parts[game["split"]].append(
            tuple(
                x[selected]
                for x in (s, act, teacher, next_state, reward, d, good, agreement)
            )
        )
    if any(not games for games in parts.values()):
        raise ValueError("Need both fight splits")
    if terminal_advantage:
        from sisters_episode_returns import outcome_advantages

        terminal_weights, terminal_stats = outcome_advantages(outcomes)
    result = {}
    catalogue = None
    for split, items in parts.items():
        s, a, label, ns, r, d, g, age = [
            np.concatenate(x) for x in zip(*items, strict=True)
        ]
        stats[split] = {"fights": len(items), "decisions": len(s)}
        if terminal_advantage:
            stats[split]["whole_fight_outcomes"] = terminal_stats[split]
        if split == "train" and mirror:
            ms, mns, ma, ml = s.copy(), ns.copy(), a.copy(), label.copy()
            ms[:, mirror_axes] *= -1
            mns[:, mirror_axes] *= -1
            ma[:, 0] *= -1
            ml[:, 0] *= -1
            s, a, label, ns = [
                np.concatenate(pair)
                for pair in ((s, ms), (a, ma), (label, ml), (ns, mns))
            ]
            r, d, g, age = [np.tile(x, 2) for x in (r, d, g, age)]
        if split == "train":
            catalogue = np.unique(np.concatenate((a, label[label[:, 2] >= 0])).astype("i8"), axis=0)
        lookup = {tuple(cmd): i for i, cmd in enumerate(catalogue)}
        yi = np.array([lookup.get(tuple(cmd), -1) for cmd in a], dtype="i8")
        li = np.array([lookup.get(tuple(cmd), -1) for cmd in label], dtype="i8")
        # -1 is a genuine holdout label outside the training action catalogue.
        # -2 is an actual interaction collected without any teacher.
        li[label[:, 2] < 0] = -2
        result[split] = tuple(
            torch.from_numpy(x).to(device) for x in (s, yi, li, ns, r, d, g, age)
        )
        if forecast:
            volumes = np.concatenate(fields[split])
            if split == "train" and mirror:
                volumes = np.concatenate((volumes, volumes[..., ::-1]))
            flat = volumes.reshape(len(volumes), -1)
            result[split] += (torch.from_numpy(flat).to(device),)
        if plans:
            weights = np.concatenate(options[split])
            if split == "train" and mirror:
                flipped = weights.reshape(-1, 3, 4)[:, ::-1].reshape(-1, 12)
                weights = np.concatenate((weights, flipped))
            result[split] += (torch.from_numpy(weights).to(device),)
        if future_keys:
            targets = np.concatenate(movements[split])
            if split == "train" and mirror:
                reflected = np.where(targets >= 0, (2 - targets // 4) * 4 + targets % 4, -1)
                targets = np.concatenate((targets, reflected))
            result[split] += (torch.from_numpy(targets).to(device),)
        if terminal_advantage:
            excess = np.concatenate(terminal_weights[split])
            if split == "train" and mirror:
                excess = np.tile(excess, 2)
            result[split] += (torch.from_numpy(excess).to(device),)
        if entry_replay:
            entry = np.concatenate(entry_rows[split])
            if split == "train" and mirror:
                entry = np.tile(entry, 2)
            result[split] += (torch.from_numpy(entry).to(device),)
        if return_seconds:
            total, remaining, successor = [
                np.concatenate(values) for values in zip(*windows[split], strict=True)
            ]
            if split == "train" and mirror:
                mirrored = successor.copy()
                mirrored[:, mirror_axes] *= -1
                successor = np.concatenate((successor, mirrored))
                total, remaining = np.tile(total, 2), np.tile(remaining, 2)
            result[split] += tuple(
                torch.from_numpy(x).to(device) for x in (total, remaining, successor)
            )
    return result, catalogue, stats


def mlp(n: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n, 256),
        nn.ReLU(),
        nn.Linear(256, 192),
        nn.ReLU(),
        nn.Linear(192, out),
    )


def previous_commands(states: Tensor) -> Tensor:
    """Decode the keys actually applied on the previous native commit."""
    bits = states[:, CONTROL + 4 : CONTROL + 13].long()
    flags = (bits << torch.arange(9, device=states.device)).sum(dim=1)
    return torch.cat((states[:, CONTROL : CONTROL + 2].long(), flags[:, None]), 1)


def movement_groups(catalogue: Tensor) -> Tensor:
    buttons = catalogue[:, 2]
    return (catalogue[:, 0] + 1) * 4 + 2 * ((buttons & 3) != 0) + ((buttons & 4) != 0)


def movement_margins(logits: Tensor, membership: Tensor, chosen: Tensor) -> Tensor:
    """Log odds of the issued motion against each motion in the native catalogue."""
    grouped = logits[:, None, :].masked_fill(~membership, -1e9).logsumexp(2)
    return grouped.gather(1, chosen[:, None]) - grouped


def greedy_motion_loss(logits: Tensor, groups: Tensor, issued: Tensor) -> Tensor:
    """Compare the best native key in the issued movement with the best competing key."""
    correct = groups[None, :] == issued[:, None]
    best_issued = logits.masked_fill(~correct, -torch.inf).amax(1)
    competing = logits.masked_fill(correct, -torch.inf).amax(1)
    return F.softplus(competing - best_issued)


OPTION_MAGIC = 0x354C5149
OPTION_FEATURES = 12


def _timing_component(component: bytes) -> bool:
    return (
        len(component) >= 20
        and struct.unpack_from("<i", component)[0] in (0x324C5149, 0x344C5149)
        and struct.unpack_from("<3i", component, 4) == (WIDTH, 256, 192)
    )


def option_components(component: bytes) -> tuple[bytes, bytes, np.ndarray]:
    """Unpack two history policies and the standardized, clipped projection head."""
    if len(component) < 12:
        raise ValueError("Truncated option policy")
    magic, first, second = struct.unpack_from("<3i", component)
    size = (WIDTH * OPTION_FEATURES + 3 * OPTION_FEATURES + 1) * 4
    if magic != OPTION_MAGIC or min(first, second) < 20 or 12 + first + second + size != len(component):
        raise ValueError("Invalid option policy lengths")
    primary, alternate = component[12:12 + first], component[12 + first:12 + first + second]
    for actor in (primary, alternate):
        if not _timing_component(actor) or struct.unpack_from("<i", actor)[0] != 0x344C5149:
            raise ValueError("Options require history actors")
    weights = np.frombuffer(component[-size:], "<f4")
    scale = weights[(WIDTH+1)*OPTION_FEATURES:(WIDTH+2)*OPTION_FEATURES]
    if not np.isfinite(weights).all() or np.any(scale <= 0):
        raise ValueError("Invalid option projection parameters")
    return primary, alternate, weights


COMPOUND_MAGIC = 0x364C5149
POLICY_GRADIENT_MAGIC = 0x394C5149
DIRECTIONAL_MAGIC = 0x3A4C5149
ROUTER_RESIDUAL_MAGIC = 0x3B4C5149
FOCUS_MAGIC = 0x374C5149
FOCUS_FEATURES = 16
MEMORY_MAGIC = 0x384C5149
ENCODED_MEMORY_MAGIC = 0x3C4C5149
CONTINUOUS_MEMORY_MAGIC = 0x3D4C5149


def memory_components(component: bytes) -> tuple[bytes, np.ndarray]:
    if len(component) < 28:
        raise ValueError("Truncated recurrent controller")
    magic, size = struct.unpack_from("<2i", component)
    if magic in (ENCODED_MEMORY_MAGIC, CONTINUOUS_MEMORY_MAGIC):
        hidden = struct.unpack_from("<i", component, 8)[0]
        start = 12
    elif magic == MEMORY_MAGIC:
        hidden, start = 24, 8
    else:
        raise ValueError("Wrong recurrent control format")
    leaf = component[start:start + size]
    if not _timing_component(leaf) or struct.unpack_from("<i", leaf)[0] != 0x344C5149:
        raise ValueError("Invalid recurrent control base")
    actions = struct.unpack_from("<i", leaf, 16)[0]
    count = 3 * hidden * (WIDTH + hidden + 2) + (hidden + 1) * actions
    if (not 1 <= hidden <= 256 or not 1 <= actions <= 4096
            or len(component) != start + size + 4 * count):
        raise ValueError("Invalid recurrent control weight count")
    parameters = np.frombuffer(component, "<f4", offset=start + size)
    if not np.isfinite(parameters).all():
        raise ValueError("Non-finite recurrent control parameters")
    return leaf, parameters


def focus_components(component: bytes) -> tuple[bytes, np.ndarray]:
    if len(component) < 8:
        raise ValueError("Truncated focus option")
    magic, size = struct.unpack_from("<2i", component)
    if magic != FOCUS_MAGIC or size < 20 or len(component) != 8 + size + 4 * FOCUS_FEATURES:
        raise ValueError("Invalid focus option size")
    leaf = component[8:8 + size]
    weights = np.frombuffer(component, "<f4", offset=8 + size)
    if not _timing_component(leaf) or struct.unpack_from("<i", leaf)[0] != 0x344C5149 or not np.isfinite(weights).all():
        raise ValueError("Invalid focus option parameters")
    return leaf, weights


def compound_components(component: bytes) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Unpack the frozen neural actor, prospective risk readout, and option selector."""
    if len(component) < 8:
        raise ValueError("Truncated compound policy")
    magic, size = struct.unpack_from("<2i", component)
    if magic != COMPOUND_MAGIC or size < 20 or len(component) != size + 8 + (WIDTH + 1)*8:
        raise ValueError("Invalid compound policy sizes")
    actor = component[8:8 + size]
    if not _timing_component(actor) or struct.unpack_from("<i", actor)[0] != 0x344C5149:
        raise ValueError("Expected a history policy for compound controls")
    weights = np.frombuffer(component, "<f4", offset=8+size).reshape(2, WIDTH+1)
    if not np.isfinite(weights).all():
        raise ValueError("Invalid compound readout weights")
    return actor, weights[0], weights[1]


def phase_checkpoint(opener: bytes, trio: bytes) -> bytes:
    if not _timing_component(opener):
        raise ValueError("Expected a timing actor for the opening")
    if len(trio) >= 4 and struct.unpack_from("<i", trio)[0] == ROUTER_RESIDUAL_MAGIC:
        router_residual_components(trio)
    elif len(trio) >= 4 and struct.unpack_from("<i", trio)[0] in (POLICY_GRADIENT_MAGIC, DIRECTIONAL_MAGIC):
        policy_gradient_components(trio)
    elif len(trio) >= 4 and struct.unpack_from("<i", trio)[0] in (
        MEMORY_MAGIC, ENCODED_MEMORY_MAGIC, CONTINUOUS_MEMORY_MAGIC,
    ):
        memory_components(trio)
    elif len(trio) >= 4 and struct.unpack_from("<i", trio)[0] == FOCUS_MAGIC:
        focus_components(trio)
    elif len(trio) >= 4 and struct.unpack_from("<i", trio)[0] == COMPOUND_MAGIC:
        compound_components(trio)
    elif not _timing_component(trio):
        option_components(trio)
    return struct.pack("<3i", 0x334C5149, len(opener), len(trio)) + opener + trio


def router_residual_components(component: bytes) -> tuple[bytes, np.ndarray]:
    if len(component) < 12:
        raise ValueError("Truncated router residual")
    magic, size = struct.unpack_from("<2i", component)
    n = 1 + 40*32 + 32 + 32*16 + 16 + 16 + 1
    if magic != ROUTER_RESIDUAL_MAGIC or size < 12 or len(component) != 8 + size + 4*n:
        raise ValueError("Invalid router residual layout")
    original = component[8:8+size]
    option_components(original)
    parameters = np.frombuffer(component, "<f4", offset=8+size)
    if not np.isfinite(parameters).all() or not 0 < parameters[0] < 1000:
        raise ValueError("Invalid router residual parameters")
    return original, parameters


def policy_gradient_components(component: bytes) -> tuple[bytes, bytes]:
    """Read the native actor and exact head shared by stochastic rollouts and greedy execution."""
    if len(component) < 16:
        raise ValueError("Truncated learned sampling policy")
    magic, n = struct.unpack_from("<2i", component)
    leaf = component[8 : 8 + n]
    head = component[8 + n :]
    if (
        magic not in (POLICY_GRADIENT_MAGIC, DIRECTIONAL_MAGIC)
        or n < 20
        or not _timing_component(leaf)
        or struct.unpack_from("<i", leaf)[0] != 0x344C5149
        or len(head) != 8 + (WIDTH + 1) * 4 * (7 if magic == DIRECTIONAL_MAGIC else 2)
    ):
        raise ValueError("Invalid learned sampling policy")
    marker, interval = struct.unpack_from("<if", head)
    if (
        marker != (0x33504753 if magic == DIRECTIONAL_MAGIC else 0x31504753)
        or not 0.31999 <= interval <= 8.40001
        or not np.isfinite(np.frombuffer(head, "<f4", offset=8)).all()
    ):
        raise ValueError("Invalid learned sampling head")
    return leaf, head


def phase_components(checkpoint: bytes) -> tuple[bytes, bytes]:
    """Read either a shared actor or both actors from a deployed checkpoint."""
    if len(checkpoint) >= 12 and struct.unpack_from("<i", checkpoint)[0] == 0x334C5149:
        _, first, late = struct.unpack_from("<3i", checkpoint)
        if min(first, late) < 20 or 12 + first + late != len(checkpoint):
            raise ValueError("Invalid phase checkpoint lengths")
        opener, trio = checkpoint[12 : 12 + first], checkpoint[12 + first :]
        phase_checkpoint(opener, trio)
        return opener, trio
    phase_checkpoint(checkpoint, checkpoint)
    return checkpoint, checkpoint


def late_component(checkpoint: bytes) -> bytes:
    return phase_components(checkpoint)[1]


def initialize_actor(actor: TimingPolicy, commands: np.ndarray, path: Path) -> int:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    source = TimingPolicy(saved["catalogue"], "past.weight" in saved["actor"])
    source.load_state_dict(saved["actor"])
    for name in ("boss", "disc", "factors"):
        getattr(actor, name).load_state_dict(getattr(source, name).state_dict())
    if actor.past is not None and source.past is not None:
        actor.past.load_state_dict(source.past.state_dict())
    actor.dense[2].load_state_dict(source.dense[2].state_dict())
    previous = {tuple(row): j for j, row in enumerate(saved["catalogue"].tolist())}
    matched = 0
    with torch.no_grad():
        dim = min(actor.dense[0].in_features, source.dense[0].in_features)
        actor.dense[0].weight.zero_()
        actor.dense[0].weight[:, :dim].copy_(source.dense[0].weight[:, :dim])
        actor.dense[0].bias.copy_(source.dense[0].bias)
        for j, cmd in enumerate(commands):
            index = previous.get(tuple(cmd))
            if index is not None:
                actor.out.weight[j].copy_(source.out.weight[index])
                actor.out.bias[j].copy_(source.out.bias[index])
                matched += 1
    return matched


def nearby_descending_sister(states: Tensor) -> Tensor:
    """Observed vulnerable trio rows within two horizontal units of a dropping or landing boss."""
    bosses = states[:, FOES:DISCS].reshape(-1, 3, 36)
    descending = bosses[:, :, 20:23].amax(2) > 0.5
    near = (bosses[:, :, 0] > 0.5) & (bosses[:, :, 1].abs() <= 0.2)
    return (states[:, 23] > 0.5) & (states[:, 9] < 0.5) & (near & descending).any(1)


def train(
    data_dir: Path,
    output: Path,
    bc_steps: int,
    world_steps: int,
    rl_steps: int,
    batch: int,
    seed: int,
    mirror: bool = True,
    switch_weight: float = 1.0,
    agreement_seconds: float = 0.0,
    forecast_weight: float = 0.0,
    opener_path: Path | None = None,
    initial_learner: Path | None = None,
    return_seconds: float = 0.0,
    history: bool = False,
    plan_weight: float = 0.0,
    late_path: Path | None = None,
    no_hit_cost: float = 0.0,
    teacher_win_min_hz: float = 0.0,
    future_key_weight: float = 0.0,
    terminal_advantage_weight: float = 0.0,
    drop_replay_fraction: float = 0.0,
    world_only: bool = False,
    initial_world: Path | None = None,
    executed_teacher_advantage: bool = False,
    reserve_opening: bool = False,
    entry_replay_fraction: float = 0.0,
) -> None:
    if reserve_opening and (late_path is None or executed_teacher_advantage):
        raise ValueError("Reserve training requires a frozen late policy and an opening learner")
    if executed_teacher_advantage and return_seconds <= 0:
        raise ValueError("Teacher advantage requires an observed reward horizon")
    if world_only and (bc_steps or rl_steps):
        raise ValueError("World pretraining requires zero actor and IQL steps")
    if (
        not 0 <= drop_replay_fraction < 1
        or not 0 <= entry_replay_fraction < 1
        or drop_replay_fraction + entry_replay_fraction >= 1
    ):
        raise ValueError("Replay fractions must be nonnegative and sum to less than one")
    if entry_replay_fraction and late_path is not None:
        raise ValueError("Trio entry replay requires a learner that trains on trio states")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    if opener_path is not None and late_path is not None:
        raise ValueError("Select exactly one frozen phase when fitting a specialist")
    opener = (
        phase_components(opener_path.read_bytes())[0] if opener_path is not None else None
    )
    late = late_component(late_path.read_bytes()) if late_path is not None else None
    data, commands, counts = build_data(
        data_dir,
        device,
        mirror,
        forecast_weight > 0,
        opener is not None,
        return_seconds,
        history,
        plan_weight > 0,
        late is not None,
        no_hit_cost,
        teacher_win_min_hz,
        future_key_weight > 0,
        terminal_advantage_weight > 0,
        executed_teacher_advantage,
        reserve_opening,
        entry_replay_fraction > 0,
    )
    output.mkdir(exist_ok=False, parents=True)
    cat = torch.as_tensor(commands, device=device)
    s, y, expert, ns, r, d, good, age = data["train"][:8]
    eligible = (expert >= 0) & (age >= agreement_seconds)
    if executed_teacher_advantage:
        eligible &= good & (y == expert)
    supervised = eligible.nonzero().flatten()
    if not len(supervised):
        raise ValueError("No teacher decisions meet the configured fight and agreement criteria")
    has_unlabeled = bool((expert < 0).any())
    switch = (cat[expert.clamp_min(0)] != previous_commands(s)).any(1)
    corrective_weight = 1 + (switch_weight - 1) * switch.float()
    valid = good.nonzero().flatten()
    dynamics_rows = (good & (d > 0)).nonzero().flatten()
    drop_count = int(batch * drop_replay_fraction)
    drop_mask = nearby_descending_sister(s) if drop_count else torch.zeros_like(good)
    drop_valid = valid[drop_mask[valid]]
    drop_dynamics = dynamics_rows[drop_mask[dynamics_rows]]
    if drop_count and (not len(drop_valid) or not len(drop_dynamics)):
        raise ValueError("No eligible descent or landing transitions for targeted replay")
    entry_count = int(batch * entry_replay_fraction)
    entry_mask = (
        data["train"][-4 if return_seconds else -1]
        if entry_replay_fraction
        else torch.zeros_like(good)
    )
    entry_valid = valid[entry_mask[valid]]
    entry_dynamics = dynamics_rows[entry_mask[dynamics_rows]]
    if entry_count and (not len(entry_valid) or not len(entry_dynamics)):
        raise ValueError("No eligible trio entry transitions for targeted replay")
    actor = TimingPolicy(cat, history).to(device)
    groups = movement_groups(cat)
    baseline = None
    motion_examples = None
    if executed_teacher_advantage:
        danger = eligible & nearby_descending_sister(s)
        changed = groups[y.clamp_min(0)] != movement_groups(previous_commands(s))
        changes, holds = (danger & changed).nonzero().flatten(), (danger & ~changed).nonzero().flatten()
        if not len(changes) or not len(holds):
            raise ValueError("Need both changed and held movements near a descending Sister")
        motion_examples = (changes, holds)
    membership = F.one_hot(groups, 12).T.bool()
    plan_index = 8 + int(forecast_weight > 0)
    if plan_weight > 0:
        plan_weights = data["train"][plan_index] * membership.any(1)
        plan_rows = (
            ((plan_weights.sum(1) > 0) & (age >= agreement_seconds)).nonzero().flatten()
        )
        if not len(plan_rows):
            raise ValueError("No eligible recorded teacher alternatives in training")
    inherited = (
        initialize_actor(actor, commands, initial_learner)
        if initial_learner is not None
        else 0
    )
    if executed_teacher_advantage:
        baseline = copy.deepcopy(actor).eval().requires_grad_(False)
    forecast_head = None
    movement_head = None
    movement_index = plan_index + int(plan_weight > 0)
    terminal_index = movement_index + int(future_key_weight > 0)
    if future_key_weight > 0:
        movement_head = nn.Linear(192, len(MOVEMENT_HORIZONS) * 12).to(device)
    if forecast_weight > 0:
        forecast_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, int(np.prod(TARGET_SHAPE))),
            nn.Sigmoid(),
        ).to(device)
    counts_act = expert[supervised].bincount(minlength=len(cat)).float().clamp_min(1)
    rare = (counts_act.max() / counts_act).pow(0.175).clamp_max(4)
    bits = actor.basis[:, 6:]
    bit_pos = bits[expert[supervised]].mean(0)
    bit_weight = ((1 - bit_pos) / bit_pos.clamp_min(1e-4)).sqrt().clamp(1, 12)
    action_encoding = torch.cat((cat[:, :2], bits), 1).float()
    report = {
        "counts": counts,
        "seed": seed,
        "batch": batch,
        "actions": len(cat),
        "mirrored_training": mirror,
        "drop_replay_fraction": drop_replay_fraction,
        "drop_replay_training_rows": len(drop_valid),
        "entry_replay_fraction": entry_replay_fraction,
        "entry_replay_training_rows": len(entry_valid),
        "teacher_switch_weight": switch_weight,
        "executed_teacher_advantage": executed_teacher_advantage,
        "reserve_opening": reserve_opening,
        "opening_discount_seconds": 40 if reserve_opening else None,
        "teacher_agreement_seconds": agreement_seconds,
        "future_box_loss_weight": forecast_weight,
        "opener_sha256": hashlib.sha256(opener).hexdigest()
        if opener is not None
        else None,
        "initial_learner_sha256": hashlib.sha256(
            initial_learner.read_bytes()
        ).hexdigest()
        if initial_learner
        else None,
        "inherited_action_rows": inherited,
        "supervised_fraction": len(supervised) / len(s),
        "train_switch_fraction": switch[expert >= 0].float().mean().item(),
        "bc_steps": bc_steps,
        "world_steps": world_steps,
        "rl_steps": rl_steps,
        "measured_return_seconds": return_seconds,
        "motion_history_lags": list(HISTORY_LAGS) if history else [],
        "teacher_plan_loss_weight": plan_weight,
        "no_hit_cost": no_hit_cost,
        "future_key_loss_weight": future_key_weight,
        "terminal_advantage_weight": terminal_advantage_weight,
        "future_key_seconds": list(MOVEMENT_HORIZONS) if movement_head is not None else [],
        "teacher_win_min_hz": teacher_win_min_hz,
        "late_sha256": hashlib.sha256(late).hexdigest() if late is not None else None,
        "teacher_plan_train_rows": len(plan_rows) if plan_weight > 0 else 0,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "manifest_sha256": hashlib.sha256(
            (data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }

    def update(opt: torch.optim.Optimizer, loss: Tensor) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def indices(n: int) -> Tensor:
        return torch.randint(n, (batch,), device=device)

    def replay_rows(pool: Tensor, drops: Tensor, entries: Tensor) -> Tensor:
        selected = pool[indices(len(pool))]
        if drop_count:
            selected[:drop_count] = drops[torch.randint(len(drops), (drop_count,), device=device)]
        if entry_count:
            selected[drop_count : drop_count + entry_count] = entries[
                torch.randint(len(entries), (entry_count,), device=device)
            ]
        return selected

    def factor_loss(factor: Tensor, label: Tensor) -> Tensor:
        return (
            F.cross_entropy(factor[:, :3], cat[label, 0] + 1)
            + F.cross_entropy(factor[:, 3:6], cat[label, 1] + 1)
            + F.binary_cross_entropy_with_logits(
                factor[:, 6:], bits[label], pos_weight=bit_weight
            )
        )

    def save(name: str) -> None:
        was_training = actor.training
        actor.eval()
        with torch.no_grad():
            hs, _, hl = data["holdout"][:3]
            pred = torch.cat([actor(x)[0].argmax(1) for x in hs.split(1024)])
            labeled = hl != -2
            report[name + "_holdout_exact"] = (
                (pred[labeled] == hl[labeled]).float().mean().item() if labeled.any() else None
            )
            known = hl >= 0
            changes = (cat[hl.clamp_min(0)] != previous_commands(hs)).any(1)
            for group, mask in (("switch", changes), ("hold", ~changes)):
                selected = known & mask
                if selected.any():
                    report[name + "_holdout_" + group + "_known_exact"] = (
                        (pred[selected] == hl[selected]).float().mean().item()
                    )
            report["holdout_label_coverage"] = (
                known[labeled].float().mean().item() if labeled.any() else None
            )
            report["holdout_unlabeled_decisions"] = int((~labeled).sum())
            path = output / (name + ".bin")
            save_native(actor, commands, path)
            if opener is not None:
                path.write_bytes(phase_checkpoint(opener, path.read_bytes()))
            elif late is not None:
                path.write_bytes(phase_checkpoint(path.read_bytes(), late))
        actor.train(was_training)
        print(name, report[name + "_holdout_exact"], flush=True)

    def forecast_loss(encoded: Tensor, rows: Tensor) -> Tensor:
        if forecast_head is None:
            return encoded.new_zeros(())
        target = data["train"][8][rows]
        known = target.isfinite()
        target = target.nan_to_num()
        weights = known * (1 + 4 * target)
        return (
            (forecast_head(encoded) - target).square() * weights
        ).sum() / weights.sum().clamp_min(1)

    def plan_loss() -> Tensor:
        if not plan_weight:
            return s.new_zeros(())
        j = plan_rows[indices(len(plan_rows))]
        margin = movement_margins(actor(s[j])[0], membership, groups[expert[j]])
        weight = plan_weights[j]
        return (F.softplus(-margin) * weight).sum() / weight.sum().clamp_min(1)

    def future_key_loss(encoded: Tensor, rows: Tensor) -> Tensor:
        if movement_head is None:
            return encoded.new_zeros(())
        target = data["train"][movement_index][rows].long()
        logits = movement_head(encoded).reshape(-1, 12)
        cost = F.cross_entropy(logits, target.flatten(), ignore_index=-1, reduction="none")
        weights = (target >= 0) * (1 + (target != groups[expert[rows], None]).float())
        return (cost * weights.flatten()).sum() / weights.sum().clamp_min(1)

    parameters = list(actor.parameters())
    if movement_head is not None:
        parameters.extend(movement_head.parameters())
    if forecast_head is not None:
        parameters.extend(forecast_head.parameters())
    opt = torch.optim.AdamW(parameters, lr=0.00065)
    for step in range(bc_steps):
        i = supervised[indices(len(supervised))]
        encoded = actor.encode(s[i])
        logits, factor = actor.heads(encoded)
        w = rare[expert[i]] * corrective_weight[i]
        loss = (
            F.cross_entropy(logits, expert[i], reduction="none") * w
        ).sum() / w.sum()
        update(
            opt,
            loss
            + 0.30 * factor_loss(factor, expert[i])
            + forecast_weight * forecast_loss(encoded, i)
            + future_key_weight * future_key_loss(encoded, i)
            + plan_weight * plan_loss(),
        )
        if (step + 1) % 2000 == 0:
            print("teacher", step + 1, flush=True)
    if not world_only:
        save("bc")
    ref = copy.deepcopy(actor).requires_grad_(False)
    width = s.shape[1]
    world = mlp(width + 11, len(PHYSICAL)).to(device)
    wopt = torch.optim.AdamW(world.parameters(), lr=0.0008)
    prior_updates = 0
    if initial_world is not None:
        saved_world = torch.load(initial_world, weights_only=True, map_location=device)
        if saved_world["physical"] != PHYSICAL or saved_world["state_width"] != width:
            raise ValueError("Pretrained world model uses different state coordinates")
        world.load_state_dict(saved_world["world"])
        wopt.load_state_dict(saved_world["optimizer"])
        prior_updates = saved_world["updates"]
        report["initial_world_sha256"] = hashlib.sha256(initial_world.read_bytes()).hexdigest()
    report["total_world_updates"] = prior_updates + world_steps
    for _ in range(world_steps):
        i = replay_rows(dynamics_rows, drop_dynamics, entry_dynamics)
        delta = world(torch.cat((s[i], action_encoding[y[i]]), 1))
        update(
            wopt,
            F.smooth_l1_loss(delta, ns[i][:, PHYSICAL] - s[i][:, PHYSICAL], beta=0.025),
        )
    world.eval().requires_grad_(False)
    hs, hy, _, hn, _, hd, hg, _ = data["holdout"][:8]
    vi = ((hy >= 0) & hg & (hd > 0)).nonzero().flatten()[::4]
    with torch.no_grad():
        delta = torch.cat(
            [
                world(torch.cat((hs[j], action_encoding[hy[j]]), 1))
                for j in vi.split(1024)
            ]
        )
        actual = hn[vi][:, PHYSICAL] - hs[vi][:, PHYSICAL]
        report.update(
            world_mae=(delta - actual).abs().mean().item(),
            persistence_mae=actual.abs().mean().item(),
        )
        if drop_count:
            subset = nearby_descending_sister(hs[vi])
            report["heldout_drop_dynamics_rows"] = int(subset.sum())
            if subset.any():
                report["heldout_drop_world_mae"] = (delta[subset] - actual[subset]).abs().mean().item()
                report["heldout_drop_persistence_mae"] = actual[subset].abs().mean().item()
    if world_only:
        target = output / "world.pt"
        torch.save({"physical": PHYSICAL, "state_width": width, "updates": prior_updates + world_steps,
                    "world": world.state_dict(), "optimizer": wopt.state_dict()}, target)
        report["world_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    q1, q2 = mlp(width, len(cat)).to(device), mlp(width, len(cat)).to(device)
    t1, t2 = (
        copy.deepcopy(q1).requires_grad_(False),
        copy.deepcopy(q2).requires_grad_(False),
    )
    val = mlp(width, 1).to(device)
    qopt = torch.optim.AdamW([*q1.parameters(), *q2.parameters()], lr=0.00035)
    vopt = torch.optim.AdamW(val.parameters(), lr=0.00035)
    aopt = torch.optim.AdamW(parameters, lr=0.00009)
    for step in range(rl_steps):
        i = replay_rows(valid, drop_valid, entry_valid)
        x, u, observed = s[i], y[i], ns[i]
        with torch.no_grad():
            key = u[:, None]
            old = torch.minimum(t1(x), t2(x)).gather(1, key).flatten()
            imagined = observed.clone()
            imagined[:, PHYSICAL] = x[:, PHYSICAL] + world(
                torch.cat((x, action_encoding[u]), 1)
            )
            target = r[i] + d[i] * (
                0.75 * val(observed).flatten() + 0.25 * val(imagined).flatten()
            )
            if return_seconds:
                payout, continuation, successor = data["train"][-3:]
                measured = payout[i] + continuation[i] * val(successor[i]).flatten()
                # Replay continuations come from a mixture of policies. Retain the
                # one-step IQL target alongside their measured short-term outcomes.
                target = 0.5 * (target + measured)
        v = val(x).flatten()
        update(vopt, quantile_loss(old - v, 0.7))
        update(
            qopt,
            sum(
                F.smooth_l1_loss(q(x).gather(1, key).flatten(), target)
                for q in (q1, q2)
            ),
        )
        with torch.no_grad():
            for slow, fast in ((t1, q1), (t2, q2)):
                for a, b in zip(slow.parameters(), fast.parameters(), strict=True):
                    a.lerp_(b, 0.005)
            aw = (6 * (old - v.detach()).clamp(-1, 1)).exp().clamp_max(12) * rare[u]
            if terminal_advantage_weight > 0:
                credit = data["train"][terminal_index][i]
                aw *= (terminal_advantage_weight * credit).clamp(-1, 1).exp()
            prior = ref(x)[0].softmax(1)
        if step >= min(1000, rl_steps // 4):
            encoded = actor.encode(x)
            logits, factor = actor.heads(encoded)
            loss = (F.cross_entropy(logits, u, reduction="none") * aw).sum() / aw.sum()
            if reserve_opening or executed_teacher_advantage or agreement_seconds > 0 or teacher_win_min_hz > 0 or has_unlabeled:
                j = supervised[indices(len(supervised))]
                labeled_encoding = actor.encode(s[j])
                supervised_logits, factor = actor.heads(labeled_encoding)
            else:
                j, supervised_logits = i, logits
                labeled_encoding = encoded
            lab = expert[j]
            w = rare[lab] * corrective_weight[j]
            motion_cost = 0.0
            if executed_teacher_advantage:
                # The teacher actually issued each selected key. Use rewards from
                # that controller's own segment and the critic's continuation value.
                payout, continuation, successor = data["train"][-3:]
                with torch.no_grad():
                    advantage = payout[j] + continuation[j] * val(successor[j]).flatten() - val(s[j]).flatten()
                    w = w * advantage.clamp(-1.5, 1.5).exp()
                change_rows, hold_rows = motion_examples
                half = batch // 2
                pick = torch.cat((change_rows[torch.randint(len(change_rows), (half,), device=device)],
                                  hold_rows[torch.randint(len(hold_rows), (batch-half,), device=device)]))
                with torch.no_grad():
                    observed_adv = payout[pick] + continuation[pick] * val(successor[pick]).flatten() - val(s[pick]).flatten()
                    weight = observed_adv.clamp(-1.5, 1.5).exp()
                penalties = greedy_motion_loss(actor(s[pick])[0], groups, groups[y[pick]])
                motion_cost = (weight * penalties).sum() / weight.sum()
            supervision = (
                F.cross_entropy(supervised_logits, lab, reduction="none") * w
            ).sum() / w.sum()
            update(
                aopt,
                loss
                + 0.7 * F.kl_div(logits.log_softmax(1), prior, reduction="batchmean")
                + 0.85 * supervision
                + 0.5 * motion_cost
                + 0.2 * factor_loss(factor, lab)
                + forecast_weight * forecast_loss(encoded, i)
                + future_key_weight * future_key_loss(labeled_encoding, j)
                + plan_weight * plan_loss(),
            )
        if (step + 1) % 2000 == 0:
            print("iql", step + 1, flush=True)
    save("world_iql")
    if baseline is not None:
        with torch.inference_mode():
            hs, actual, hl, _, _, _, usable, hage = data["holdout"][:8]
            selected = (hl >= 0) & (hl == actual) & usable & (hage >= agreement_seconds) & nearby_descending_sister(hs)
            h = hs[selected]
            if len(h):
                before = groups[torch.cat([baseline(x)[0].argmax(1) for x in h.split(512)])]
                after = groups[torch.cat([actor(x)[0].argmax(1) for x in h.split(512)])]
                target = groups[hl[selected]]
                switches = target != movement_groups(previous_commands(h))
                report["heldout_descent_executed_motion"] = {
                    name: {"rows": int(m.sum()), "baseline_correct": int(((before == target) & m).sum()),
                           "candidate_correct": int(((after == target) & m).sum()),
                           "gained": int(((before != target) & (after == target) & m).sum()),
                           "lost": int(((before == target) & (after != target) & m).sum())}
                    for name, m in (("change", switches), ("hold", ~switches))
                }
    if movement_head is not None:
        with torch.no_grad():
            prediction = torch.cat([movement_head(actor.encode(x)).reshape(
                -1, len(MOVEMENT_HORIZONS), 12).argmax(2) for x in hs.split(1024)])
            target = data["holdout"][movement_index]
            current = groups[data["holdout"][2].clamp_min(0), None]
            valid_targets = ((target >= 0) & (data["holdout"][2][:, None] >= 0) &
                             (data["holdout"][7][:, None] >= agreement_seconds))
            changing = valid_targets & (current != target)
            for name, mask in (("all", valid_targets), ("change", changing)):
                report["future_key_holdout_" + name] = {
                    "eligible": int(mask.sum()),
                    "correct": int(((prediction == target) & mask).sum()),
                    "retain_current_correct": int(((current == target) & mask).sum()),
                }
    if plan_weight:
        with torch.no_grad():
            hold = data["holdout"]
            weights = hold[plan_index] * membership.any(1)
            usable = (
                (weights.sum(1) > 0) & (hold[2] >= 0) & (hold[7] >= agreement_seconds)
            )
            rows = usable.nonzero().flatten()
            report["teacher_plan_holdout_rows"] = len(rows)
            if len(rows):
                correct = s.new_zeros(())
                for j in rows.split(1024):
                    margin = movement_margins(
                        actor(hs[j])[0], membership, groups[hold[2][j]]
                    )
                    correct += (weights[j] * (margin > 0)).sum()
                report["teacher_plan_holdout_weighted_order"] = (
                    correct / weights[rows].sum()
                ).item()
    if return_seconds:
        with torch.no_grad():
            payout, continuation, successor = data["holdout"][-3:]
            rows = (hg & (hy >= 0)).nonzero().flatten()
            errors = []
            for j in rows.split(1024):
                observed = payout[j] + continuation[j] * val(successor[j]).flatten()
                estimate = (
                    torch.minimum(q1(hs[j]), q2(hs[j])).gather(1, hy[j, None]).flatten()
                )
                errors.append(
                    torch.stack(
                        (
                            (observed - estimate).abs(),
                            (observed - val(hs[j]).flatten()).abs(),
                        )
                    )
                )
            report["heldout_return_q_mae"], report["heldout_return_value_mae"] = (
                torch.cat(errors, dim=1).mean(dim=1).tolist()
            )
    if forecast_head is not None:
        with torch.no_grad():
            estimated = torch.cat(
                [forecast_head(actor.encode(x)) for x in hs.split(1024)]
            )
            target = data["holdout"][8]
            report["future_box_holdout_mae"] = (
                (estimated - target).abs().nanmean().item()
            )
            report["future_box_zero_baseline_mae"] = target.abs().nanmean().item()
    torch.save(
        {
            "actor": actor.cpu().state_dict(),
            "catalogue": cat.cpu(),
            "world": world.cpu().state_dict(),
            "q1": q1.cpu().state_dict(),
            "q2": q2.cpu().state_dict(),
            "value": val.cpu().state_dict(),
            **({"future_keys": movement_head.cpu().state_dict()} if movement_head is not None else {}),
            **({"opener_native": opener} if opener is not None else {}),
            **({"late_native": late} if late is not None else {}),
            **(
                {"future_boxes": forecast_head.cpu().state_dict()}
                if forecast_head is not None
                else {}
            ),
        },
        output / "learner.pt",
    )
    (output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bc-steps", type=int, default=11000)
    parser.add_argument("--world-steps", type=int, default=2500)
    parser.add_argument("--rl-steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--teacher-agreement-seconds", type=float, default=0.0)
    parser.add_argument("--forecast-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-win-min-hz",
        type=float,
        default=0.0,
        help="Fit teacher labels on full victories meeting this decision cadence; retain all real RL transitions.",
    )
    parser.add_argument(
        "--no-hit-cost",
        type=float,
        default=0.0,
        help="Extra cost per lost mask and first hit, plus twice this amount for a clean clear.",
    )
    parser.add_argument(
        "--late-policy",
        type=Path,
        help="Fit the opening subtask and preserve this network's late-phase policy.",
    )
    parser.add_argument(
        "--plan-weight",
        type=float,
        default=0.0,
        help="Supervise movement rankings from optional development planner traces.",
    )
    parser.add_argument(
        "--motion-history",
        action="store_true",
        help="Encode two earlier full arena observations from this episode.",
    )
    parser.add_argument("--future-key-weight", type=float, default=0.0,
                        help="Train a shared auxiliary head on upcoming demonstrated movements.")
    parser.add_argument("--terminal-advantage-weight", type=float, default=0.0,
                        help="Weight executed actions by the measured whole-fight return above a held-out state baseline.")
    parser.add_argument(
        "--return-seconds",
        type=float,
        default=0.0,
        help="Mix short real trajectories into the IQL target, measured in game seconds.",
    )
    parser.add_argument(
        "--opener-policy",
        type=Path,
        help="Retain this frozen timing actor for the opener; fit only on trio states.",
    )
    parser.add_argument(
        "--initial-learner",
        type=Path,
        help="Initialize the learned actor from a prior timing checkpoint.",
    )
    parser.add_argument("--drop-replay-fraction", type=float, default=0.0,
                        help="Reserve a share of world-model and IQL minibatches for observed nearby boss drops.")
    parser.add_argument("--entry-replay-fraction", type=float, default=0.0,
                        help="Reserve a share of world-model and IQL minibatches for the first ten seconds of the trio.")
    parser.add_argument("--world-only", action="store_true",
                        help="Save only the trained dynamics model and optimizer.")
    parser.add_argument("--initial-world", type=Path,
                        help="Resume dynamics training from a verified world-only checkpoint.")
    parser.add_argument("--executed-teacher-advantage", action="store_true",
                        help="Anchor policy improvement on issued teacher keys weighted by their observed short return.")
    parser.add_argument("--reserve-opening", action="store_true",
                        help="Learn from actually issued high-resource openings and value spell charges at the handoff.")
    args = parser.parse_args()
    if min(args.world_steps, args.batch) < 1 or (not args.world_only and min(args.bc_steps, args.rl_steps) < 1):
        parser.error("Positive budgets required")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("Switch weight must be finite and at least one")
    if (
        not np.isfinite(args.teacher_agreement_seconds)
        or args.teacher_agreement_seconds < 0
    ):
        parser.error("Teacher agreement duration must be finite and nonnegative")
    if not np.isfinite(args.forecast_weight) or args.forecast_weight < 0:
        parser.error("Forecast weight must be finite and nonnegative")
    if not 0 <= args.return_seconds <= 4:
        parser.error("Measured return horizon must be between zero and four seconds")
    if not np.isfinite(args.plan_weight) or args.plan_weight < 0:
        parser.error("Plan weight must be finite and nonnegative")
    if not np.isfinite(args.future_key_weight) or args.future_key_weight < 0:
        parser.error("Future-key weight must be finite and nonnegative")
    if not np.isfinite(args.terminal_advantage_weight) or args.terminal_advantage_weight < 0:
        parser.error("Terminal-advantage weight must be finite and nonnegative")
    if not np.isfinite(args.no_hit_cost) or args.no_hit_cost < 0:
        parser.error("No-hit cost must be finite and nonnegative")
    if not np.isfinite(args.teacher_win_min_hz) or args.teacher_win_min_hz < 0:
        parser.error("Teacher decision cadence must be finite and nonnegative")
    train(
        args.data,
        args.output,
        args.bc_steps,
        args.world_steps,
        args.rl_steps,
        args.batch,
        args.seed,
        not args.no_mirror,
        args.switch_weight,
        args.teacher_agreement_seconds,
        args.forecast_weight,
        args.opener_policy,
        args.initial_learner,
        args.return_seconds,
        args.motion_history,
        args.plan_weight,
        args.late_policy,
        args.no_hit_cost,
        args.teacher_win_min_hz,
        args.future_key_weight,
        args.terminal_advantage_weight,
        args.drop_replay_fraction,
        args.world_only,
        args.initial_world,
        args.executed_teacher_advantage,
        args.reserve_opening,
        args.entry_replay_fraction,
    )
