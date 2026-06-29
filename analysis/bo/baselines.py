"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

bo/baselines.py
───────────────
Baseline optimisers to compare against Bayesian optimisation.

  RandomSearch  — uniform random sampling (equal budget as BO)
  GridSearch    — exhaustive grid over a coarse discretisation
  ExpertRunner  — evaluates the hand-crafted practitioner schedule once

All baselines share the same objective interface as BayesianOptimizer,
so they produce directly comparable results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from fwi.schedule import (
    BOUNDS_LOWER, BOUNDS_UPPER, DIM, PARAM_NAMES,
    lhs_sample, random_theta, expert_schedule, normalize,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared result container (mirrors BOResult for easy comparison)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    method:       str
    best_theta:   np.ndarray
    best_J:       float
    best_metrics: Dict
    all_theta:    np.ndarray      # (n_eval, 5)
    all_J:        np.ndarray      # (n_eval,)
    all_metrics:  List[Dict]
    wall_times:   np.ndarray      # (n_eval,) seconds per evaluation

    @property
    def incumbent_trace(self) -> np.ndarray:
        """Running minimum of J over evaluations."""
        return np.minimum.accumulate(self.all_J)

    @property
    def n_eval(self) -> int:
        return len(self.all_J)


# ──────────────────────────────────────────────────────────────────────────────
# Random search
# ──────────────────────────────────────────────────────────────────────────────

class RandomSearch:
    """
    Pure random search over the schedule parameter space.

    Used to answer: *does BO's GP surrogate add value over random sampling
    at the same evaluation budget?*

    Budget should equal n_init + n_iter from BOConfig so the comparison is fair.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], Dict],
        n_eval:  int   = 50,
        seed:    int   = 0,
    ):
        self.objective_fn = objective_fn
        self.n_eval       = n_eval
        self.rng          = np.random.default_rng(seed)

    def run(self) -> BaselineResult:
        logger.info(f"RandomSearch: {self.n_eval} evaluations")

        all_theta:   List[np.ndarray] = []
        all_J:       List[float]      = []
        all_metrics: List[Dict]       = []
        wall_times:  List[float]      = []

        incumbent_J    = float("inf")
        best_theta     = None
        best_metrics   = {}

        for i in range(self.n_eval):
            theta = random_theta(self.rng)
            t0    = time.perf_counter()
            try:
                metrics = self.objective_fn(theta)
            except Exception as exc:
                logger.warning(f"RS eval {i} failed: {exc}")
                metrics = {"J": 999.0}
            elapsed = time.perf_counter() - t0

            J = metrics["J"]
            all_theta.append(theta)
            all_J.append(J)
            all_metrics.append(metrics)
            wall_times.append(elapsed)

            if J < incumbent_J:
                incumbent_J  = J
                best_theta   = theta.copy()
                best_metrics = metrics.copy()

            logger.info(f"  RS {i:02d}/{self.n_eval}  J={J:.6f}  "
                        f"best={incumbent_J:.6f}  [{elapsed:.1f}s]")

        return BaselineResult(
            method="Random Search",
            best_theta=best_theta,
            best_J=incumbent_J,
            best_metrics=best_metrics,
            all_theta=np.array(all_theta),
            all_J=np.array(all_J),
            all_metrics=all_metrics,
            wall_times=np.array(wall_times),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Grid search
# ──────────────────────────────────────────────────────────────────────────────

class GridSearch:
    """
    Coarse grid over (f_min, f_max, K) with gamma and beta fixed to defaults.

    Full 5-D grids are too expensive; we fix the two less-impactful parameters
    (gamma, beta) to domain-knowledge defaults and sweep the three most
    important ones.  This gives a fair, manually-inspired baseline.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], Dict],
        f_min_values:   Optional[List[float]] = None,
        f_max_values:   Optional[List[float]] = None,
        K_values:       Optional[List[int]]   = None,
        gamma_fixed:    float = 1.5,   # domain-knowledge default
        beta_fixed:     float = 0.1,
    ):
        self.objective_fn = objective_fn
        self.f_min_values  = f_min_values  or [2.0, 3.0, 5.0, 7.0]
        self.f_max_values  = f_max_values  or [15.0, 20.0, 25.0]
        self.K_values      = K_values      or [3, 4, 5, 6]
        self.gamma_fixed   = gamma_fixed
        self.beta_fixed    = beta_fixed

    @property
    def grid_size(self) -> int:
        return len(self.f_min_values) * len(self.f_max_values) * len(self.K_values)

    def run(self) -> BaselineResult:
        logger.info(f"GridSearch: {self.grid_size} grid points")

        all_theta:   List[np.ndarray] = []
        all_J:       List[float]      = []
        all_metrics: List[Dict]       = []
        wall_times:  List[float]      = []

        incumbent_J  = float("inf")
        best_theta   = None
        best_metrics = {}
        i            = 0

        for f_min in self.f_min_values:
            for f_max in self.f_max_values:
                if f_min >= f_max:
                    continue
                for K in self.K_values:
                    theta = np.array([f_min, f_max, float(K),
                                      self.gamma_fixed, self.beta_fixed])
                    t0 = time.perf_counter()
                    try:
                        metrics = self.objective_fn(theta)
                    except Exception as exc:
                        logger.warning(f"GS eval {i} failed: {exc}")
                        metrics = {"J": 999.0}
                    elapsed = time.perf_counter() - t0

                    J = metrics["J"]
                    all_theta.append(theta)
                    all_J.append(J)
                    all_metrics.append(metrics)
                    wall_times.append(elapsed)

                    if J < incumbent_J:
                        incumbent_J  = J
                        best_theta   = theta.copy()
                        best_metrics = metrics.copy()

                    logger.info(
                        f"  GS {i:02d}/{self.grid_size}  "
                        f"f=[{f_min:.1f},{f_max:.1f}] K={K}  "
                        f"J={J:.6f}  best={incumbent_J:.6f}  [{elapsed:.1f}s]"
                    )
                    i += 1

        return BaselineResult(
            method="Grid Search",
            best_theta=best_theta,
            best_J=incumbent_J,
            best_metrics=best_metrics,
            all_theta=np.array(all_theta),
            all_J=np.array(all_J),
            all_metrics=all_metrics,
            wall_times=np.array(wall_times),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Expert schedule runner
# ──────────────────────────────────────────────────────────────────────────────

class ExpertRunner:
    """
    Evaluates one hand-crafted schedule (the practitioner baseline).

    The expert schedule mimics what you would normally do when running
    toy2dac: a few frequency groups based on domain knowledge.
    Specify your own groups as (f_lo, f_hi) pairs.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], Dict],
        expert_groups: Optional[List[tuple]] = None,
        f_min: float = 3.0,
        f_max: float = 20.0,
    ):
        self.objective_fn  = objective_fn
        self.expert_groups = expert_groups
        self.f_min         = f_min
        self.f_max         = f_max

    def run(self) -> BaselineResult:
        """
        Evaluate the expert schedule.  Returns a BaselineResult with one entry.
        """
        from fwi.schedule import expert_schedule
        schedule = expert_schedule(
            f_min=self.f_min, f_max=self.f_max, groups=self.expert_groups
        )
        # Expert theta = the pseudo-theta stored in the schedule
        theta = schedule.theta

        logger.info(f"ExpertRunner: evaluating {schedule}")
        t0      = time.perf_counter()
        metrics = self.objective_fn(theta)
        elapsed = time.perf_counter() - t0
        J       = metrics["J"]
        logger.info(f"  Expert  J={J:.6f}  [{elapsed:.1f}s]")

        return BaselineResult(
            method="Expert Schedule",
            best_theta=theta,
            best_J=J,
            best_metrics=metrics,
            all_theta=theta[np.newaxis],
            all_J=np.array([J]),
            all_metrics=[metrics],
            wall_times=np.array([elapsed]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed runner (for mean ± std reporting required by reviewers)
# ──────────────────────────────────────────────────────────────────────────────

def run_random_search_multi_seed(
    objective_fn: Callable[[np.ndarray], Dict],
    n_eval:  int,
    seeds:   List[int] = (0, 1, 2, 3, 4),
) -> Dict:
    """
    Run random search with multiple random seeds and aggregate statistics.
    This is what reviewers expect to see alongside BO results.

    Returns
    -------
    dict with keys:
      'results'           : list of BaselineResult (one per seed)
      'incumbent_mean'    : np.ndarray (n_eval,)  mean incumbent J
      'incumbent_std'     : np.ndarray (n_eval,)  std  incumbent J
      'best_J_all_seeds'  : np.ndarray (n_seeds,) final best J per seed
    """
    results = []
    traces  = []

    for seed in seeds:
        logger.info(f"RandomSearch seed={seed}")
        rs     = RandomSearch(objective_fn, n_eval=n_eval, seed=seed)
        result = rs.run()
        results.append(result)
        traces.append(result.incumbent_trace)

    traces_arr = np.stack(traces, axis=0)   # (n_seeds, n_eval)
    return {
        "results":          results,
        "incumbent_mean":   traces_arr.mean(axis=0),
        "incumbent_std":    traces_arr.std(axis=0),
        "best_J_all_seeds": np.array([r.best_J for r in results]),
    }
