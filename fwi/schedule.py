"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

fwi/schedule.py
───────────────
Frequency-schedule parameterisation for multi-scale FWI.

A schedule is encoded by five continuous parameters:

  theta = [f_min, f_max, K_raw, gamma, beta]

  f_min   (Hz) : lowest frequency to invert
  f_max   (Hz) : highest frequency to invert
  K_raw        : real-valued proxy for number of groups; rounded to int K ∈ [2,10]
  gamma   (>0) : spacing exponent.  gamma=1 → linear; gamma>1 → denser at low end
  beta   [0,1) : group-overlap fraction (0 = adjacent groups touch; 0.4 = 40% overlap)

Public API
──────────
  make_schedule(theta)          → FrequencySchedule
  lhs_sample(n, seed)           → array (n, 5)   Latin-Hypercube init samples
  normalize(theta)              → array (5,)      [0,1]^5
  denormalize(theta_norm)       → array (5,)      natural units
  BOUNDS_LOWER, BOUNDS_UPPER    → array (5,)      search-space edges
  PARAM_NAMES                   → list[str]       for labelling plots/tables
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrequencyGroup:
    """One contiguous frequency band handed to a single toy2dac FWI run."""
    index:       int            # 0-based position in the schedule
    f_center:    float          # nominal centre frequency (Hz)
    f_lo:        float          # lower edge (Hz) as passed to toy2dac
    f_hi:        float          # upper edge (Hz) as passed to toy2dac
    frequencies: np.ndarray     # discrete Hz values inside this band

    def __repr__(self) -> str:
        return (f"Group {self.index:02d}: [{self.f_lo:.2f}, {self.f_hi:.2f}] Hz"
                f"  centre={self.f_center:.2f} Hz  n_freq={len(self.frequencies)}")


@dataclass
class FrequencySchedule:
    """An ordered list of FrequencyGroups generated from a parameter vector."""
    groups: List[FrequencyGroup]
    theta:  np.ndarray          # the (5,) vector that produced this schedule

    # ── convenience ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.groups)

    def __repr__(self) -> str:
        header = (f"FrequencySchedule  K={len(self)}  "
                  f"theta={np.round(self.theta, 3)}")
        rows   = [f"  {g}" for g in self.groups]
        return "\n".join([header] + rows)

    @property
    def all_frequencies(self) -> np.ndarray:
        """Flat sorted array of every discrete frequency in the schedule."""
        return np.unique(np.concatenate([g.frequencies for g in self.groups]))

    def summary_dict(self) -> dict:
        """Serialisable summary for logging."""
        return {
            "K":      len(self),
            "f_min":  float(self.groups[0].f_lo),
            "f_max":  float(self.groups[-1].f_hi),
            "gamma":  float(self.theta[3]),
            "beta":   float(self.theta[4]),
            "groups": [{"f_lo": g.f_lo, "f_hi": g.f_hi,
                        "f_center": g.f_center,
                        "n_freq": len(g.frequencies)} for g in self.groups],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Search-space bounds   ← ADAPT THESE TO YOUR MARMOUSI SETUP
# ──────────────────────────────────────────────────────────────────────────────

# Key numbers to check in your toy2dac parameter file / acquisition geometry:
#   - What is the dominant frequency of your source wavelet?
#   - What is the grid spacing?  => Nyquist determines f_max.
#   - How much low-frequency content does the Marmousi synthetic data have?

SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "f_min":  ( 2.0,   8.0),   # Hz  — never go below 1 Hz or above wavelet energy
    "f_max":  (15.0,  30.0),   # Hz  — Marmousi grid typically handles up to ~25-30 Hz
    "K_raw":  ( 2.0,  10.0),   # real-valued; rounded to int in [2, 10]
    "gamma":  ( 0.5,   3.0),   # exponent controlling density at low vs high freqs
    "beta":   ( 0.0,   0.45),  # overlap: 0 = touching, 0.45 = 45% overlap
}

PARAM_NAMES   = list(SEARCH_BOUNDS.keys())
BOUNDS_LOWER  = np.array([v[0] for v in SEARCH_BOUNDS.values()], dtype=np.float64)
BOUNDS_UPPER  = np.array([v[1] for v in SEARCH_BOUNDS.values()], dtype=np.float64)
DIM           = len(PARAM_NAMES)   # 5


# ──────────────────────────────────────────────────────────────────────────────
# Core constructor
# ──────────────────────────────────────────────────────────────────────────────

def make_schedule(
    theta: Sequence[float] | np.ndarray,
    f_sample: float = 0.5,          # Hz — discrete frequency step inside groups
) -> FrequencySchedule:
    """
    Build a FrequencySchedule from a parameter vector.

    Parameters
    ----------
    theta : array-like of length 5
        [f_min, f_max, K_raw, gamma, beta] in natural units.
    f_sample : float
        Spacing (Hz) of discrete frequencies passed to toy2dac for each group.
        Should match the frequency resolution used in your observed-data simulation.

    Returns
    -------
    FrequencySchedule
    """
    theta  = np.asarray(theta, dtype=np.float64)
    f_min, f_max, K_raw, gamma, beta = theta

    # ── Validate & clip ───────────────────────────────────────────────────────
    f_min  = float(np.clip(f_min, BOUNDS_LOWER[0], BOUNDS_UPPER[0]))
    f_max  = float(np.clip(f_max, BOUNDS_LOWER[1], BOUNDS_UPPER[1]))
    K      = int(np.clip(round(K_raw), 2, 10))
    gamma  = float(np.clip(gamma, 0.1, 5.0))
    beta   = float(np.clip(beta,  0.0, 0.49))

    if f_min >= f_max:
        raise ValueError(
            f"f_min ({f_min:.2f} Hz) must be strictly less than f_max ({f_max:.2f} Hz)."
        )

    # ── Centre frequencies via power-law spacing ─────────────────────────────
    #   t in [0,1] parametrises position; t^gamma compresses toward 0
    #   => more groups at low frequencies when gamma > 1  (typical for FWI)
    t       = np.linspace(0.0, 1.0, K)
    centers = f_min + (f_max - f_min) * (t ** gamma)

    # ── Half-widths (= half the gap to the nearest neighbour × (1+beta)) ─────
    half_widths = _half_widths(centers, f_min, f_max, beta)

    # ── Build FrequencyGroup objects ──────────────────────────────────────────
    groups = []
    for i, (fc, hw) in enumerate(zip(centers, half_widths)):
        f_lo  = float(np.clip(fc - hw, f_min, f_max))
        f_hi  = float(np.clip(fc + hw, f_min, f_max))
        freqs = _discrete_freqs(f_lo, f_hi, f_sample, f_min, f_max)
        groups.append(FrequencyGroup(
            index=i, f_center=float(fc),
            f_lo=f_lo, f_hi=f_hi, frequencies=freqs,
        ))

    return FrequencySchedule(groups=groups, theta=np.array(theta))


# ──────────────────────────────────────────────────────────────────────────────
# Predefined baseline schedules  (for comparison experiments)
# ──────────────────────────────────────────────────────────────────────────────

def expert_schedule(
    f_min: float = 3.0,
    f_max: float = 20.0,
    groups: List[tuple[float,float]] | None = None,
    f_sample: float = 0.5,
) -> FrequencySchedule:
    """
    Hand-crafted multi-scale schedule that a practitioner would use.

    Default: three groups — low / mid / high — mimicking the common
    "Bunks-style" strategy.  Override `groups` with explicit (f_lo, f_hi)
    pairs to replicate whatever you'd normally do in your toy2dac runs.
    """
    if groups is None:
        span = f_max - f_min
        groups = [
            (f_min,          f_min + span/3),
            (f_min + span/4, f_min + 2*span/3),
            (f_min + span/2, f_max),
        ]
    fgroups = []
    for i, (lo, hi) in enumerate(groups):
        fc    = 0.5 * (lo + hi)
        freqs = _discrete_freqs(lo, hi, f_sample, f_min, f_max)
        fgroups.append(FrequencyGroup(index=i, f_center=fc,
                                      f_lo=lo, f_hi=hi, frequencies=freqs))
    # Synthesise a plausible theta (used only for logging, not optimisation)
    pseudo_theta = np.array([f_min, f_max, len(fgroups), 1.0, 0.0])
    return FrequencySchedule(groups=fgroups, theta=pseudo_theta)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers (used by the BO surrogate)
# ──────────────────────────────────────────────────────────────────────────────

def normalize(theta: np.ndarray) -> np.ndarray:
    """Map natural theta to [0, 1]^5."""
    return (np.asarray(theta) - BOUNDS_LOWER) / (BOUNDS_UPPER - BOUNDS_LOWER)

def denormalize(theta_norm: np.ndarray) -> np.ndarray:
    """Map [0, 1]^5 back to natural theta."""
    return BOUNDS_LOWER + np.asarray(theta_norm) * (BOUNDS_UPPER - BOUNDS_LOWER)


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers (used for LHS warm-start)
# ──────────────────────────────────────────────────────────────────────────────

def lhs_sample(n: int, seed: int = 42) -> np.ndarray:
    """
    Latin-Hypercube sample of n schedules in natural parameter space.

    Returns
    -------
    np.ndarray of shape (n, 5)
    """
    from scipy.stats import qmc
    sampler      = qmc.LatinHypercube(d=DIM, seed=seed)
    unit_samples = sampler.random(n=n)                  # (n, 5) in [0,1]
    return qmc.scale(unit_samples, BOUNDS_LOWER, BOUNDS_UPPER)

def random_theta(rng: np.random.Generator | None = None) -> np.ndarray:
    """Single uniform-random sample in natural bounds."""
    rng = rng or np.random.default_rng()
    return rng.uniform(BOUNDS_LOWER, BOUNDS_UPPER)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _half_widths(
    centers: np.ndarray,
    f_min: float,
    f_max: float,
    beta:  float,
) -> np.ndarray:
    """Half-width for each group, with optional overlap controlled by beta."""
    n = len(centers)
    hw = np.empty(n)
    for i in range(n):
        if n == 1:
            half_gap = 0.5 * (f_max - f_min)
        elif i == 0:
            half_gap = 0.5 * (centers[1] - centers[0])
        elif i == n - 1:
            half_gap = 0.5 * (centers[-1] - centers[-2])
        else:
            half_gap = 0.5 * min(centers[i] - centers[i-1],
                                  centers[i+1] - centers[i])
        hw[i] = half_gap * (1.0 + 2.0 * beta)   # extend by beta fraction
    return hw

def _discrete_freqs(
    f_lo: float, f_hi: float,
    f_sample: float,
    f_min: float, f_max: float,
) -> np.ndarray:
    """Generate the discrete Hz values that toy2dac will receive for this group."""
    freqs = np.arange(f_lo, f_hi + 0.5 * f_sample, f_sample)
    return np.clip(freqs, f_min, f_max)


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    theta = [3.0, 22.0, 4.5, 1.8, 0.15]
    sched = make_schedule(theta)
    print(sched)
    print(f"\nAll unique freqs: {sched.all_frequencies}")
    print(f"\nExpert schedule:\n{expert_schedule()}")
    samples = lhs_sample(n=10)
    print(f"\nLHS samples (10 × 5):\n{np.round(samples, 2)}")
