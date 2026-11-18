"""Episode-local replay for supervised movement and buttons alongside local recurrent PPO."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hkrl import spaces
from hkrl.models.base import ActorCritic
from hkrl.models.heads import ACTION_TENSOR_DIM_NO_MACRO, CompositeActionDistribution
from hkrl.training.sisters_dodges import DODGE_BUTTONS

DEMO_OBS = ("global", "player", "entities", "entity_mask", "prev_action", "prev_reward")


def split_demonstrations(directory: Path) -> tuple[list[Path], list[Path]]:
    """Hold out every fourth whole episode in collector order."""
    files = sorted(directory.glob("episode-*.npz"))
    return [p for i, p in enumerate(files) if i % 4 != 3], files[3::4]


def read_demonstration(path: Path, payload: bytes) -> dict[str, Tensor]:
    """Recover the same recurrent context as the live worker at each episode boundary."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        episode = {key: torch.from_numpy(archive[key]) for key in archive.files}
    actions = episode["actions"].long()
    if len(actions) == 0 or any(len(x) != len(actions) for x in episode.values()):
        raise ValueError(f"inconsistent or empty demonstration episode: {path}")
    if actions.ndim != 2 or actions.shape[-1] not in (
        ACTION_TENSOR_DIM_NO_MACRO,
        ACTION_TENSOR_DIM_NO_MACRO + 1,
    ):
        raise ValueError(f"invalid packed demonstration actions: {path}")
    macro = actions.shape[-1] == ACTION_TENSOR_DIM_NO_MACRO + 1
    noop = actions.new_tensor(spaces.canonical_noop_action_values(enable_macro=macro))
    episode["prev_action"] = torch.cat((noop[None], actions[:-1])).float()
    rewards = episode["rewards"].float().reshape(-1, 1)
    episode["prev_reward"] = torch.cat((torch.zeros_like(rewards[:1]), rewards[:-1]))
    episode["actions"] = actions
    return episode


def compact_obs(data: dict[str, Tensor]) -> dict[str, Tensor]:
    """Discard only entity columns padded in every frame of this minibatch."""
    obs = {key: data[key] for key in DEMO_OBS}
    keep = obs["entity_mask"].bool().any(dim=(0, 1))
    if not keep.any():
        keep[0] = True
    obs["entities"] = obs["entities"][:, :, keep]
    obs["entity_mask"] = obs["entity_mask"][:, :, keep]
    return obs


class DemonstrationReplay:
    def __init__(self, directory: Path, length: int, burn_in: int) -> None:
        self.length = length
        self.burn_in = burn_in
        self.episodes: list[dict[str, Tensor]] = []
        self.windows: list[tuple[int, int]] = []
        self.dodge_windows: list[int] = []
        self.hashes: dict[str, str] = {}
        files, holdout = split_demonstrations(directory)
        if not holdout or length <= 0 or burn_in < 0:
            raise ValueError("demonstration replay needs a whole-episode split and valid lengths")
        for path in files:
            payload = path.read_bytes()
            data = read_demonstration(path, payload)
            episode = {key: data[key] for key in (*DEMO_OBS, "actions", "masks")}
            actions = episode["actions"]
            index = len(self.episodes)
            if "dodge_outcomes" in data:
                grades = data["dodge_outcomes"]
                if (
                    grades.shape != (len(actions), len(DODGE_BUTTONS))
                    or not ((grades >= -1) & (grades <= 2)).all()
                ):
                    raise ValueError(f"invalid dodge outcomes: {path}")
                episode["dodge_outcomes"] = grades
            if self.episodes and self.episodes[0].keys() != episode.keys():
                raise ValueError("cannot mix annotated and unannotated demonstrations")
            for start in range(0, len(actions), length):
                if (
                    "dodge_outcomes" in episode
                    and (episode["dodge_outcomes"][start : start + length] > 0).any()
                ):
                    self.dodge_windows.append(len(self.windows))
                self.windows.append((index, start))
            self.episodes.append(episode)
            self.hashes[path.name] = hashlib.sha256(payload).hexdigest()

    def loss(self, model: ActorCritic, rng: np.random.Generator, device: torch.device) -> Tensor:
        clips = []
        choices = rng.integers(len(self.windows), size=3)
        if self.dodge_windows:
            choices[0] = rng.choice(self.dodge_windows)
        for choice in choices:
            index, start = self.windows[int(choice)]
            episode = self.episodes[index]
            lo = max(0, start - self.burn_in)
            hi = min(start + self.length, len(episode["actions"]))
            times = torch.arange(lo, lo + self.length + self.burn_in)
            clip = {k: t[times.clamp(max=hi - 1)] for k, t in episode.items()}
            clip["valid"] = (times >= start) & (times < hi)
            clips.append(clip)
        data = {k: torch.stack([clip[k] for clip in clips]).to(device) for k in clips[0]}
        mask = data.pop("masks").bool()
        valid = data.pop("valid")
        actions = data.pop("actions")
        dist, _, _ = model(compact_obs(data), action_mask=mask)
        if "dodge_outcomes" in data:
            return dodge_imitation_loss(dist, actions, mask, valid, data["dodge_outcomes"])
        move_loss = -dist.movement_x.log_prob(actions[..., 0])[valid].mean()
        # An unavailable attack/jump conveys no evidence about whether to press it.
        button_targets = actions[..., 2 : 2 + spaces.N_BUTTONS].float()
        button_logp = dist.buttons.log_prob(button_targets)
        loss = move_loss
        for name in ("attack", "jump_tap"):
            bit = spaces.BUTTON_BITS[name]
            valid_button = valid & mask[..., spaces.N_MOVEMENT_X + spaces.N_AIM_Y + bit]
            weights = valid_button * (1 + 3 * button_targets[..., bit])
            loss = loss - (weights * button_logp[..., bit]).sum() / weights.sum().clamp(min=1)
        return loss


def dodge_imitation_loss(
    dist: CompositeActionDistribution,
    actions: Tensor,
    mask: Tensor,
    valid: Tensor,
    grades: Tensor,
) -> Tensor:
    """Keep ordinary movement and attacks while learning infrequent verified evasions."""
    loss = -dist.movement_x.log_prob(actions[..., 0])[valid].mean()
    loss -= 0.35 * dist.aim_y.log_prob(actions[..., 1])[valid].mean()
    targets = actions[..., 2 : 2 + spaces.N_BUTTONS].float()
    logp = dist.buttons.log_prob(targets)
    offset = spaces.N_MOVEMENT_X + spaces.N_AIM_Y
    for name in ("attack", "jump_hold", *DODGE_BUTTONS):
        bit = spaces.BUTTON_BITS[name]
        legal = valid & mask[..., offset + bit]
        if name in DODGE_BUTTONS:
            grade = grades[..., DODGE_BUTTONS.index(name)]
            positive = (grade > 0) & legal
            negative = (targets[..., bit] == 0) & legal
            # A failed or unobserved attempt is neither a positive nor a negative.
            weights = positive * (1 + (grade == 2).float())
            pos_nll = -(weights * logp[..., bit]).sum() / weights.sum().clamp(min=1)
            neg_nll = -(negative * logp[..., bit]).sum() / negative.sum().clamp(min=1)
            loss = loss + 0.45 * pos_nll + 0.55 * neg_nll
        else:
            weights = legal * (1 + 3 * targets[..., bit])
            nll = -(weights * logp[..., bit]).sum() / weights.sum().clamp(min=1)
            loss = loss + nll * (0.35 if name == "jump_hold" else 1)
    return loss
