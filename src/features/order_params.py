"""Collective order parameters for behavior classification."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "phi_trans",
    "phi_tan",
    "phi_rad_pm",
    "phi_tan_unsigned",
    "phi_local",
)

R_LOCAL = 90.0
K_LOCAL = 5
_EPS = 1e-12
_SPEED_EPS = 1e-9


def _unit_vectors(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speeds = np.linalg.norm(vel, axis=-1)
    hat_v = np.zeros_like(vel)
    moving = speeds > _SPEED_EPS
    hat_v[moving] = vel[moving] / speeds[moving, None]
    return hat_v, moving


def _local_alignment(
    pos: np.ndarray,
    hat_v: np.ndarray,
    moving: np.ndarray,
    radius: float = R_LOCAL,
    k: int = K_LOCAL,
) -> np.ndarray:
    """Mean neighborhood polarization using up to `k` nearest moving neighbors
    strictly inside `radius`.

    The focal fish is excluded. Zero-speed neighbors are omitted. Fish with no
    valid neighbor are skipped; if none remain, the frame value is 0.
    """
    t, n, _ = pos.shape
    if n <= 1:
        return np.zeros(t)
    k = min(int(k), n - 1)
    r2 = float(radius) ** 2
    out = np.zeros(t)
    step = 64 if n * n * 64 * 8 < 256 * 1024 * 1024 else max(1, 8_000_000 // max(n * n, 1))
    idx = np.arange(n)
    for start in range(0, t, step):
        sl = slice(start, min(t, start + step))
        block = pos[sl]
        hv = hat_v[sl]
        mv = moving[sl]
        tb = block.shape[0]
        delta = block[:, :, None, :] - block[:, None, :, :]
        d2 = np.einsum("tijn,tijn->tij", delta, delta)
        d2[:, idx, idx] = np.inf
        d2 = np.where((d2 < r2) & mv[:, None, :], d2, np.inf)
        nn = np.argpartition(d2, kth=k - 1, axis=-1)[..., :k]
        t_idx = np.arange(tb)[:, None, None]
        i_idx = np.arange(n)[None, :, None]
        valid = np.isfinite(d2[t_idx, i_idx, nn])
        neigh_v = hv[t_idx, nn]
        counts = valid.sum(axis=-1)
        has = counts > 0
        summed = np.sum(np.where(valid[..., None], neigh_v, 0.0), axis=2)
        mean_v = np.zeros_like(summed)
        np.divide(summed, counts[..., None], out=mean_v, where=has[..., None])
        mags = np.linalg.norm(mean_v, axis=-1)
        n_has = has.sum(axis=1)
        np.divide((mags * has).sum(axis=1), n_has, out=out[sl], where=n_has > 0)
    return out


def compute_order_params(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, float]:
    """positions, velocities: (N, 2)"""
    series = compute_order_params_series(positions[None, ...], velocities[None, ...])
    return {k: float(v[0]) for k, v in series.items()}


def compute_order_params_series(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized over time. positions, velocities: (T, N, 2).

    Φ_trans = ||⟨v̂_i⟩|| over moving fish.

    Anisotropy-corrected correlations (centered unit fields):
        v'_i = v̂_i − ⟨v̂⟩ ,  r'_i = r̂_i − ⟨r̂⟩
        D = sqrt( (∑_i ||v'_i||²) (∑_i ||r'_i||²) )
        Φ_rad^± = (∑_i r'_i · v'_i) / D
        Φ_tan   = |∑_i (r'_i × v'_i)_z| / D

    Degenerate frames:
        zero speed            — omit from heading averages; if none move, Φ_trans = 0
                                and the centered correlations are 0
        fish at the centroid  — omit from r̂; it does not enter Φ_tan or Φ_rad^±
        D ≈ 0                 — Φ_tan = Φ_rad^± = 0
        no centroid-relative motion — Φ_tan^unsigned = 0
        no moving neighbor inside R_LOCAL among the K_LOCAL nearest — skip that
                                fish in Φ_local; if none remain, 0
    """
    # Convert positions and velocities to float arrays
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)

    # If input is only a single frame, expand dims for consistent processing (T, N, 2)
    if pos.ndim == 2:
        pos = pos[None, ...]
        vel = vel[None, ...]
    t, n, _ = pos.shape

    # Handle case with no individuals: return zero arrays for each feature
    if n == 0:
        z = np.zeros(t)
        return {k: z.copy() for k in FEATURE_NAMES}

    # Compute unit velocity vectors and a boolean mask for moving fish (speed > _SPEED_EPS)
    hat_v, moving = _unit_vectors(vel)

    # --- φ_trans: Translational global order ---
    # Mean of the unit velocity vectors for each timepoint
    mean_hat = hat_v.mean(axis=1)
    # Magnitude of mean unit velocity: global alignment of fish velocities
    phi_trans = np.linalg.norm(mean_hat, axis=-1)

    # --- Center and normalize positions to school centroid direction vectors (r̂) ---
    centroid = pos.mean(axis=1, keepdims=True)      # Centroid position per frame
    q = pos - centroid                             # Vector from centroid to each fish
    q_norm = np.linalg.norm(q, axis=-1)            # Distance from centroid
    off_center = q_norm > _SPEED_EPS               # Exclude fish at centroid (numerical stability)

    # Compute unit direction ("r̂") from centroid, handle near-zero division robustly
    r_hat = np.zeros_like(q)
    r_hat[..., 0] = np.where(off_center, q[..., 0] / np.maximum(q_norm, _EPS), 0.0)
    r_hat[..., 1] = np.where(off_center, q[..., 1] / np.maximum(q_norm, _EPS), 0.0)

    # --- Mask for fish with valid velocity and not at centroid ---
    valid = moving & off_center
    n_valid = valid.sum(axis=1)  # Number of valid fish per frame

    # Compute mean r̂ and mean v̂ over valid fish
    sum_r = np.sum(r_hat * valid[..., None], axis=1)
    sum_v = np.sum(hat_v * valid[..., None], axis=1)
    r_bar = np.zeros_like(sum_r)
    v_bar = np.zeros_like(sum_v)
    np.divide(sum_r, n_valid[:, None], out=r_bar, where=n_valid[:, None] > 0)
    np.divide(sum_v, n_valid[:, None], out=v_bar, where=n_valid[:, None] > 0)

    # --- Centered r̂ and v̂: subtract their means from each fish ("primed" fields) ---
    r_p = np.where(valid[..., None], r_hat - r_bar[:, None, :], 0.0)
    v_p = np.where(valid[..., None], hat_v - v_bar[:, None, :], 0.0)

    # Sum of squares of centered v̂ and r̂ for denominator in correlation order params
    sum_v2 = np.sum(v_p[..., 0] ** 2 + v_p[..., 1] ** 2, axis=1)
    sum_r2 = np.sum(r_p[..., 0] ** 2 + r_p[..., 1] ** 2, axis=1)
    denom = np.sqrt(sum_v2 * sum_r2)  # Anisotropy-correction normalization

    # --- φ_rad_pm: Radial correlation (dot product) ---
    phi_rad_pm = np.zeros(t)
    np.divide(
        np.sum(r_p * v_p, axis=(1, 2)),
        denom,
        out=phi_rad_pm,
        where=denom > _EPS,
    )

    # --- φ_tan: Tangential correlation (z-component of cross product, summed) ---
    cross_z = r_p[..., 0] * v_p[..., 1] - r_p[..., 1] * v_p[..., 0]
    phi_tan = np.zeros(t)
    np.divide(
        np.abs(np.sum(cross_z, axis=1)),
        denom,
        out=phi_tan,
        where=denom > _EPS,
    )

    # --- φ_tan_unsigned: "Unsigned tangential fraction" (energy) ---
    # Computes fraction of school velocity variance along tangential direction
    u = vel - vel.mean(axis=1, keepdims=True)  # Centered velocities
    u_norm2 = np.sum(u[..., 0] ** 2 + u[..., 1] ** 2, axis=1)
    cross_u = r_hat[..., 0] * u[..., 1] - r_hat[..., 1] * u[..., 0]
    phi_tan_unsigned = np.zeros(t)
    np.divide(
        np.sum(cross_u ** 2, axis=1),
        u_norm2,
        out=phi_tan_unsigned,
        where=u_norm2 > _EPS,
    )

    # --- φ_local: Mean local alignment among up to K_LOCAL neighbors inside R_LOCAL ---
    phi_local = _local_alignment(pos, hat_v, moving, radius=R_LOCAL, k=K_LOCAL)

    return {
        "phi_trans": phi_trans,
        "phi_tan": phi_tan,
        "phi_rad_pm": phi_rad_pm,
        "phi_tan_unsigned": phi_tan_unsigned,
        "phi_local": phi_local,
    }
