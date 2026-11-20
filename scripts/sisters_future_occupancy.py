"""Build supervised future collision-box proximity targets from one recorded fight."""

from __future__ import annotations

import numpy as np

HORIZONS = (0.12, 0.32, 0.60)
LATERAL = (-2.5, 0.0, 2.5)
HEIGHT = (7.41, 9.8, 12.2)
FAMILIES = ((24, 60, 96), (132, 146))
TARGET_SHAPE = (len(HORIZONS), len(FAMILIES), len(HEIGHT), len(LATERAL))


def future_occupancy(times: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Measure distance to future boxes around the Knight's current horizontal position.

    Absent entities produce zero proximity. Missing future observations produce NaN
    so that terminations and timeouts cannot become artificial safe labels.
    """
    result = np.full((len(times), *TARGET_SHAPE), np.nan, dtype=np.float32)
    probe_x = states[:, 0, None] * 10 + 30 + np.asarray(LATERAL, dtype=np.float32)
    probe_y = np.asarray(HEIGHT, dtype=np.float32)
    for horizon_index, horizon in enumerate(HORIZONS):
        target_time = times + horizon
        ahead = np.searchsorted(times, target_time - 1e-5)
        inside = ahead < len(times)
        ahead = np.minimum(ahead, len(times) - 1)
        inside &= np.abs(times[ahead] - target_time) <= 0.08
        future = states[ahead]
        hero_x, hero_y = future[:, 0] * 10 + 30, future[:, 1] * 8 + 7.41
        for family_index, slots in enumerate(FAMILIES):
            field = np.zeros((len(times), len(HEIGHT), len(LATERAL)), dtype=np.float32)
            for offset in slots:
                present = (future[:, offset] > 0) & (future[:, offset + 9] > 0)
                x = hero_x + future[:, offset + 1] * 10
                y = hero_y + future[:, offset + 2] * 8
                half_w = future[:, offset + 7] * 2.5
                half_h = future[:, offset + 8] * 2.5
                gap_x = np.abs(x[:, None] - probe_x) - half_w[:, None]
                gap_y = np.abs(y[:, None] - probe_y) - half_h[:, None]
                separation = np.maximum(gap_x[:, None, :], gap_y[:, :, None])
                proximity = np.exp(-np.maximum(separation, 0) / 1.4)
                np.maximum(field, proximity * present[:, None, None], out=field)
            result[inside, horizon_index, family_index] = field[inside]
    return result
