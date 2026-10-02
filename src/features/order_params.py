"""Collective order parameters for five-class motion classification."""

from __future__ import annotations

import numpy as np

# Instantaneous per-frame values are the classifier inputs.
FEATURE_NAMES = (
    "phi_trans",
    "phi_tan",
    "phi_rad_pm",
    "phi_tan_unsigned",
    "phi_local",
)

K_LOCAL = 5


def _unit_vectors(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speeds = np.linalg.norm(vel, axis=-1)
    hat_v = np.zeros_like(vel)
    moving = speeds > 1e-9
    hat_v[moving] = vel[moving] / speeds[moving, None]
    return hat_v, moving


def _local_alignment(pos: np.ndarray, hat_v: np.ndarray, k: int = K_LOCAL) -> np.ndarray:
    """Mean neighborhood polarization over k nearest neighbors (excluding self)."""
    t, n, _ = pos.shape
    if n <= 1:
        return np.zeros(t)
    k = min(int(k), n - 1)
    out = np.zeros(t)
    # Chunk time so the T×N×N distance cube stays in cache / RAM.
    step = 64 if n * n * 64 * 8 < 256 * 1024 * 1024 else max(1, 8_000_000 // max(n * n, 1))
    idx = np.arange(n)
    for start in range(0, t, step):
        sl = slice(start, min(t, start + step))
        block = pos[sl]
        hv = hat_v[sl]
        tb = block.shape[0]
        delta = block[:, :, None, :] - block[:, None, :, :]
        d2 = np.einsum("tijn,tijn->tij", delta, delta)
        d2[:, idx, idx] = np.inf
        nn = np.argpartition(d2, kth=k - 1, axis=-1)[..., :k]
        t_idx = np.arange(tb)[:, None, None]
        neigh_v = hv[t_idx, nn]
        out[sl] = np.linalg.norm(neigh_v.mean(axis=2), axis=-1).mean(axis=1)
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

    Φ_trans = ||⟨v̂_i⟩||

    Center unit headings and radial vectors:
        v'_i = v̂_i − v̄ ,  r'_i = r̂_i − r̄
        D = sqrt( (∑_i ||v'_i||²) (∑_i ||r'_i||²) )

    Φ_rad^± = (∑_i v'_i · r'_i) / D
    Φ_tan   = |∑_i (r'_i × v'_i)_z| / D

    With actual relative kinematics q_i = x_i − x̄, u_i = v_i − v̄, r̂_i = q_i / ||q_i||:
    Φ_tan^unsigned = ∑_i [(r̂_i × u_i)_z]² / ∑_i ||u_i||²
    Φ_local        = (1/N) ∑_i || (1/|N_i|) ∑_{j∈N_i} v̂_j ||   (k=5 nearest neighbors)
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[None, ...]
        vel = vel[None, ...]
    t, n, _ = pos.shape
    if n == 0:
        z = np.zeros(t)
        return {k: z.copy() for k in FEATURE_NAMES}

    hat_v, _ = _unit_vectors(vel)

    phi_trans = np.linalg.norm(hat_v.mean(axis=1), axis=-1)

    centroid = pos.mean(axis=1, keepdims=True)
    q = pos - centroid
    q_norm = np.linalg.norm(q, axis=-1)
    safe = q_norm > 1e-9
    r_hat = np.zeros_like(q)
    r_hat[..., 0] = np.where(safe, q[..., 0] / np.maximum(q_norm, 1e-12), 0.0)
    r_hat[..., 1] = np.where(safe, q[..., 1] / np.maximum(q_norm, 1e-12), 0.0)

    v_p = hat_v - hat_v.mean(axis=1, keepdims=True)
    r_p = r_hat - r_hat.mean(axis=1, keepdims=True)
    sum_v2 = np.sum(v_p[..., 0] ** 2 + v_p[..., 1] ** 2, axis=1)
    sum_r2 = np.sum(r_p[..., 0] ** 2 + r_p[..., 1] ** 2, axis=1)
    denom = np.sqrt(sum_v2 * sum_r2)

    phi_rad_pm = np.zeros(t)
    np.divide(np.sum(v_p * r_p, axis=(1, 2)), denom, out=phi_rad_pm, where=denom > 1e-12)

    cross_z = r_p[..., 0] * v_p[..., 1] - r_p[..., 1] * v_p[..., 0]
    phi_tan = np.zeros(t)
    np.divide(np.abs(np.sum(cross_z, axis=1)), denom, out=phi_tan, where=denom > 1e-12)

    u = vel - vel.mean(axis=1, keepdims=True)
    u_norm2 = np.sum(u[..., 0] ** 2 + u[..., 1] ** 2, axis=1)
    cross_u = r_hat[..., 0] * u[..., 1] - r_hat[..., 1] * u[..., 0]
    phi_tan_unsigned = np.zeros(t)
    np.divide(np.sum(cross_u ** 2, axis=1), u_norm2, out=phi_tan_unsigned, where=u_norm2 > 1e-12)

    phi_local = _local_alignment(pos, hat_v, k=K_LOCAL)

    return {
        "phi_trans": phi_trans,
        "phi_tan": phi_tan,
        "phi_rad_pm": phi_rad_pm,
        "phi_tan_unsigned": phi_tan_unsigned,
        "phi_local": phi_local,
    }


def aggregate_series(
    series: dict[str, np.ndarray],
    fps: float = 30.0,
) -> dict[str, float]:
    del fps  # reserved for future temporal derivatives
    feat: dict[str, float] = {}
    for k, arr in series.items():
        feat[f"{k}_mean"] = float(np.mean(arr))
        feat[f"{k}_std"] = float(np.std(arr))
    return feat


AGG_FEATURE_NAMES = [f"{k}_mean" for k in FEATURE_NAMES]
