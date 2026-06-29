"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

fwi/metrics.py
──────────────
Objective functions and quality metrics for the BO loop.

Two objectives are defined:

  supervised    J(θ) = RMSE(m_est, m_true) / ‖m_true‖     ← uses ground truth
  unsupervised  Ju(θ) = Σ_ω Σ_s ‖R p_s(m_est,ω) − d^obs_s(ω)‖²  ← field-realistic

Both are returned by evaluate_result().  The BO surrogate is trained on
the supervised objective during development.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Primary supervised objective (used to train the GP surrogate)
# ──────────────────────────────────────────────────────────────────────────────

def normalized_rmse(
    m_est:  np.ndarray,
    m_true: np.ndarray,
    mask:   Optional[np.ndarray] = None,
) -> float:
    """
    Normalised root-mean-squared error.

        J = ‖m_est − m_true‖_F / ‖m_true‖_F

    Parameters
    ----------
    m_est, m_true : (nz, nx) velocity arrays [m/s]
    mask          : boolean array; if given, only masked cells are compared.
                    Useful for focusing on the target depth range.
    Returns
    -------
    float in [0, ∞)   (0 is perfect reconstruction)
    """
    if mask is not None:
        diff  = (m_est - m_true)[mask]
        ref   = m_true[mask]
    else:
        diff  = m_est  - m_true
        ref   = m_true

    return float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-30))


def rmse_ms(m_est: np.ndarray, m_true: np.ndarray) -> float:
    """Absolute RMSE in m/s (not normalised)."""
    return float(np.sqrt(np.mean((m_est - m_true) ** 2)))


def r_squared(m_est: np.ndarray, m_true: np.ndarray) -> float:
    """Coefficient of determination R² ∈ (−∞, 1]."""
    ss_res = np.sum((m_true - m_est) ** 2)
    ss_tot = np.sum((m_true - np.mean(m_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-30))


def ssim(
    m_est:  np.ndarray,
    m_true: np.ndarray,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> float:
    """
    Structural Similarity Index (SSIM) between two velocity models.
    Values in [−1, 1]; 1 means identical.

    Normalises by the global max before computing to make c1, c2 scale-invariant.
    """
    scale = max(m_true.max(), m_est.max()) + 1e-30
    x, y  = m_est / scale, m_true / scale
    mu_x, mu_y   = x.mean(), y.mean()
    sig_x, sig_y = x.std(),  y.std()
    sig_xy        = np.mean((x - mu_x) * (y - mu_y))
    return float(
        (2*mu_x*mu_y + c1) * (2*sig_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (sig_x**2 + sig_y**2 + c2))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unsupervised objective (field-realistic)
# ──────────────────────────────────────────────────────────────────────────────

def data_residual_norm(
    d_cal: np.ndarray,   # (n_recv, n_src, n_freq) complex predicted data
    d_obs: np.ndarray,   # (n_recv, n_src, n_freq) complex observed data
    normalize: bool = True,
) -> float:
    """
    Normalised L² data residual on a held-out frequency set.

        Ju = ‖d_cal − d_obs‖_F² / ‖d_obs‖_F²   (if normalize=True)
           = ‖d_cal − d_obs‖_F²                   (if normalize=False)

    Pass the validation frequencies that were NOT used in the inversion.
    """
    diff = d_cal - d_obs
    num  = float(np.linalg.norm(diff) ** 2)
    if normalize:
        den = float(np.linalg.norm(d_obs) ** 2) + 1e-30
        return num / den
    return num


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated evaluation — call this after every FWI run
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_result(
    m_est:   np.ndarray,
    m_true:  np.ndarray,
    d_cal_val:  Optional[np.ndarray] = None,
    d_obs_val:  Optional[np.ndarray] = None,
    depth_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full metric suite for one FWI result.

    Parameters
    ----------
    m_est     : inverted velocity model  (nz, nx) [m/s]
    m_true    : true Marmousi model      (nz, nx) [m/s]
    d_cal_val : predicted data at held-out validation frequencies (optional)
    d_obs_val : observed data at held-out validation frequencies (optional)
    depth_mask: boolean mask for region of interest (optional)

    Returns
    -------
    dict with keys:
      'J'       — normalised RMSE (the BO objective to MINIMISE)
      'rmse_ms' — absolute RMSE [m/s]
      'r2'      — coefficient of determination
      'ssim'    — structural similarity
      'Ju'      — unsupervised data-residual objective (if data provided)
    """
    metrics: Dict[str, float] = {
        "J":       normalized_rmse(m_est, m_true, mask=depth_mask),
        "rmse_ms": rmse_ms(m_est, m_true),
        "r2":      r_squared(m_est, m_true),
        "ssim":    ssim(m_est, m_true),
    }

    if d_cal_val is not None and d_obs_val is not None:
        metrics["Ju"] = data_residual_norm(d_cal_val, d_obs_val)
    else:
        metrics["Ju"] = float("nan")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Marmousi model utilities
# ──────────────────────────────────────────────────────────────────────────────

def smooth_model(
    model: np.ndarray,
    sigma_m: float = 200.0,   # smoothing length in metres
    dx:      float = 1.25,    # grid spacing [m] — match your cfg.dx
    dz:      float = 1.25,    # grid spacing [m] — match your cfg.dz
) -> np.ndarray:
    """
    Apply Gaussian smoothing to create a starting model.

    sigma_m is in metres; internally converted to grid-point standard deviations.
    Typical values for Marmousi: 100–500 m.
    """
    from scipy.ndimage import gaussian_filter
    sigma_x = sigma_m / dx
    sigma_z = sigma_m / dz
    return gaussian_filter(model, sigma=[sigma_z, sigma_x])


def add_noise_to_data(
    data: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to seismogram data to reach target SNR.

    SNR (dB) = 20 log₁₀(‖signal‖ / ‖noise‖)

    Parameters
    ----------
    data   : complex or real array of seismograms
    snr_db : target signal-to-noise ratio in dB
    rng    : numpy random generator (for reproducibility)
    """
    rng         = rng or np.random.default_rng(42)
    signal_norm = np.linalg.norm(data)
    snr_linear  = 10.0 ** (snr_db / 20.0)
    noise_std   = signal_norm / (snr_linear * np.sqrt(data.size))

    if np.iscomplexobj(data):
        noise = noise_std * (rng.standard_normal(data.shape)
                             + 1j * rng.standard_normal(data.shape)) / np.sqrt(2)
    else:
        noise = noise_std * rng.standard_normal(data.shape)

    return data + noise


def depth_zone_mask(
    nz: int, nx: int,
    z_min_m: float, z_max_m: float, dz: float,
) -> np.ndarray:
    """
    Boolean mask selecting rows (depth samples) between z_min_m and z_max_m.
    Use to restrict RMSE computation to a target depth interval.
    """
    z_idx_min = max(0,  int(z_min_m / dz))
    z_idx_max = min(nz, int(z_max_m / dz) + 1)
    mask = np.zeros((nz, nx), dtype=bool)
    mask[z_idx_min:z_idx_max, :] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Sample-efficiency helper (used by plots.py)
# ──────────────────────────────────────────────────────────────────────────────

def evaluations_to_threshold(
    objective_trace: np.ndarray,   # incumbent values over iterations
    threshold: float,
) -> int:
    """
    Return the iteration index at which the incumbent first drops below
    `threshold`.  Returns len(trace) if the threshold is never reached.
    """
    for i, val in enumerate(objective_trace):
        if val <= threshold:
            return i
    return len(objective_trace)
