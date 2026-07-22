from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

def _lag_offsets(
    shape: Tuple[int, int],
    lag_min: float,
    lag_max: float,
    pixel_scale_x: float,
    pixel_scale_y: float,
    exclude_self: bool,
) -> np.ndarray:
    """Return integer (dy, dx) offsets inside one lag annulus."""
    ny, nx = shape

    max_dx = min(nx - 1, int(np.ceil(lag_max / pixel_scale_x)))
    max_dy = min(ny - 1, int(np.ceil(lag_max / pixel_scale_y)))

    lag_min_sq = lag_min * lag_min
    lag_max_sq = lag_max * lag_max
    dx_all = np.arange(-max_dx, max_dx + 1, dtype=np.int32)
    dx_distance_sq = (dx_all.astype(np.float64) * pixel_scale_x) ** 2

    rows: List[np.ndarray] = []
    for dy in range(-max_dy, max_dy + 1):
        distance_sq = dx_distance_sq + (dy * pixel_scale_y) ** 2
        keep = (distance_sq >= lag_min_sq) & (distance_sq < lag_max_sq)
        if exclude_self and dy == 0:
            keep &= dx_all != 0

        if np.any(keep):
            selected_dx = dx_all[keep]
            selected_dy = np.full(selected_dx.size, dy, dtype=np.int32)
            rows.append(np.column_stack((selected_dy, selected_dx)))

    if not rows:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(rows, axis=0)


def structurefunction_sampled(
    Qmap: Any,
    Umap: Any,
    lag: float = 4.0,
    s_lag: float = 1.0,
    mask: Optional[Any] = None,
    header: Optional[Any] = None,
    *,
    n_pairs: int = 100_000,
    batch_size: int = 100_000,
    max_draws: Optional[int] = None,
    random_state: Any = None,
    exclude_self: bool = True,
) -> Dict[str, Union[float, int]]:
    """
    Estimate the second-order polarization-angle structure function by
    uniformly sampling valid ordered pixel pairs in the requested lag bin.

    The estimator is

        S2 = sqrt(mean(delta_psi**2))

    where delta_psi is calculated directly from Stokes Q and U.

    Parameters
    ----------
    Qmap, Umap
        Two-dimensional Stokes maps with identical shapes.
    lag, s_lag
        Accept pairs with separation in [lag - s_lag, lag + s_lag).
        With a FITS header, units are the units of CDELT1/CDELT2; otherwise
        separations are measured in pixels.
    mask
        Nonzero pixels are usable. Pixels with non-finite Q or U are excluded.
    header
        Optional FITS-like header. This preserves the original function's
        linear CDELT1/CDELT2 distance approximation.
    n_pairs
        Target number of accepted measurement pairs.
    batch_size
        Number of proposals processed at once. This controls peak temporary RAM.
    max_draws
        Maximum number of proposed pairs. The function may return fewer than
        n_pairs when the valid mask is sparse. Defaults to 50 * n_pairs.
    random_state
        Seed or NumPy Generator for reproducible sampling.
    exclude_self
        Exclude zero-separation pairs.

    Returns
    -------
    dict
        S2                : sampled structure-function estimate in radians
        npairs            : number of accepted sampled pairs
        draws             : number of pair proposals
        acceptance        : accepted proposals / draws
        candidate_offsets : number of integer offsets in the lag annulus
    """
    q = np.asarray(Qmap)
    u = np.asarray(Umap)

    if q.ndim != 2 or u.ndim != 2:
        raise ValueError("Qmap and Umap must be two-dimensional")
    if q.shape != u.shape:
        raise ValueError("Dimensions of Qmap and Umap must match")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lag < 0 or s_lag < 0:
        raise ValueError("lag and s_lag must be non-negative")

    lag_min = max(0.0, float(lag) - float(s_lag))
    lag_max = float(lag) + float(s_lag)
    if lag_max <= lag_min:
        raise ValueError("The lag interval must have positive width")

    if mask is None:
        valid = np.ones(q.shape, dtype=bool)
    else:
        mask_array = np.asarray(mask)
        if mask_array.shape != q.shape:
            raise ValueError("Dimensions of mask and Qmap must match")
        valid = mask_array != 0

    valid &= np.isfinite(q) & np.isfinite(u)
    valid_flat = np.flatnonzero(valid)
    if valid_flat.size < (1 if not exclude_self else 2):
        return {
            "S2": np.nan,
            "npairs": 0,
            "draws": 0,
            "acceptance": 0.0,
            "candidate_offsets": 0,
        }

    if header is None:
        pixel_scale_x = 1.0
        pixel_scale_y = 1.0
    else:
        try:
            pixel_scale_x = abs(float(header["CDELT1"]))
            pixel_scale_y = abs(float(header["CDELT2"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "header must provide numeric CDELT1 and CDELT2 values"
            ) from exc

    if (
        not np.isfinite(pixel_scale_x)
        or not np.isfinite(pixel_scale_y)
        or pixel_scale_x <= 0
        or pixel_scale_y <= 0
    ):
        raise ValueError("Pixel scales must be finite and positive")

    offsets = _lag_offsets(
        q.shape,
        lag_min,
        lag_max,
        pixel_scale_x,
        pixel_scale_y,
        exclude_self,
    )
    if offsets.size == 0:
        return {
            "S2": np.nan,
            "npairs": 0,
            "draws": 0,
            "acceptance": 0.0,
            "candidate_offsets": 0,
        }

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if max_draws is None:
        max_draws = max(50 * n_pairs, batch_size)
    if max_draws <= 0:
        raise ValueError("max_draws must be positive")

    ny, nx = q.shape
    accepted = 0
    draws = 0
    sum_delta_psi_sq = 0.0

    while accepted < n_pairs and draws < max_draws:
        current_batch = min(batch_size, max_draws - draws)

        source_choices = rng.integers(
            0, valid_flat.size, size=current_batch
        )
        offset_choices = rng.integers(
            0, offsets.shape[0], size=current_batch
        )

        source_flat = valid_flat[source_choices]
        y1 = source_flat // nx
        x1 = source_flat - y1 * nx

        selected_offsets = offsets[offset_choices]
        y2 = y1 + selected_offsets[:, 0]
        x2 = x1 + selected_offsets[:, 1]

        in_bounds = (y2 >= 0) & (y2 < ny) & (x2 >= 0) & (x2 < nx)
        y1 = y1[in_bounds]
        x1 = x1[in_bounds]
        y2 = y2[in_bounds]
        x2 = x2[in_bounds]

        second_is_valid = valid[y2, x2]
        y1 = y1[second_is_valid]
        x1 = x1[second_is_valid]
        y2 = y2[second_is_valid]
        x2 = x2[second_is_valid]

        delta_psi = 0.5 * np.arctan2(
            q[y1, x1] * u[y2, x2] - q[y2, x2] * u[y1, x1],
            q[y1, x1] * q[y2, x2] + u[y1, x1] * u[y2, x2],
        )

        remaining = n_pairs - accepted
        if delta_psi.size > remaining:
            delta_psi = delta_psi[:remaining]

        sum_delta_psi_sq += np.sum(
            delta_psi * delta_psi, dtype=np.float64
        )
        accepted += delta_psi.size
        draws += current_batch

    s2 = (
        float(np.sqrt(sum_delta_psi_sq / accepted))
        if accepted
        else np.nan
    )

    if accepted < n_pairs:
        warnings.warn(
            f"Accepted {accepted:,} pairs out of the requested {n_pairs:,}. "
            "Increase max_draws or use a less restrictive mask.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "S2": s2,
        "npairs": int(accepted),
        "draws": int(draws),
        "acceptance": float(accepted / draws) if draws else 0.0,
        "candidate_offsets": int(offsets.shape[0]),
    }
