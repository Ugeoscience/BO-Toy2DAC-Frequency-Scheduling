"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

bo/bayesian_optimizer.py
─────────────────────────
Bayesian optimisation loop for the frequency-schedule search.

Architecture
────────────
  SingleTaskGP (BoTorch)     — Gaussian-process surrogate
  Matérn-5/2 kernel          — twice-differentiable, robust default
  Expected Improvement (EI)  — acquisition function
  optimize_acqf              — multi-restart gradient-based inner optimiser

The GP always operates in normalised input space [0,1]^5 and
standardised output space (zero mean, unit variance).

Usage
─────
  from bo.bayesian_optimizer import BayesianOptimizer, BOConfig

  cfg = BOConfig(n_init=10, n_iter=40)
  opt = BayesianOptimizer(objective_fn, cfg)
  result = opt.run()
  print(result.best_theta, result.best_J)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from fwi.schedule import (
    DIM, BOUNDS_LOWER, BOUNDS_UPPER, PARAM_NAMES,
    denormalize, normalize, lhs_sample, make_schedule,
)

logger = logging.getLogger(__name__)

# Use float64 throughout — BoTorch default, important for GP conditioning
torch.set_default_dtype(torch.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BOConfig:
    # Warm-start
    n_init:      int   = 10          # LHS samples before BO starts
    lhs_seed:    int   = 42          # reproducibility seed for LHS
    # Budget
    n_iter:      int   = 40          # BO iterations AFTER warm-start
    # Acquisition
    acq_type:    str   = "EI"        # "EI" | "qEI"
    q:           int   = 1           # batch size (use q>1 for parallel FWI)
    xi:          float = 0.0         # exploration bonus for EI
    acq_restarts: int  = 10          # multi-restart gradient ascent on acqf
    acq_samples:  int  = 512         # raw Sobol samples for acqf initialisation
    # GP kernel
    nu:          float = 2.5         # Matérn smoothness (0.5 | 1.5 | 2.5)
    # Convergence
    ei_threshold: float = 1e-6       # stop early if EI < this  (1e-3 is right for real FWI; 1e-5 lets mock run to full budget)
    # I/O
    log_dir:     str   = "./results/logs"
    checkpoint_every: int = 5        # save checkpoint every N BO iterations


# ──────────────────────────────────────────────────────────────────────────────
# Iteration record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BOIteration:
    iteration:    int
    theta:        np.ndarray        # natural-space parameters
    J:            float             # objective value (normalised RMSE)
    metrics:      Dict              # full metric dict from evaluate_result()
    incumbent_J:  float             # best J seen so far (including this iter)
    acq_value:    float             # EI at the chosen theta
    wall_time_s:  float             # time for this FWI run

    def to_dict(self) -> dict:
        d = asdict(self) if hasattr(self, '__dataclass_fields__') else self.__dict__.copy()
        d["theta"] = self.theta.tolist()
        return d


@dataclass
class BOResult:
    best_theta:   np.ndarray        # θ* — optimal schedule parameters
    best_J:       float             # J(θ*)
    best_metrics: Dict
    iterations:   List[BOIteration]
    gp_model:     object            # fitted BoTorch GP (for surrogate plots)
    train_X:      Tensor            # all evaluated X (normalised)
    train_Y:      Tensor            # all evaluated Y (standardised)

    @property
    def incumbent_trace(self) -> np.ndarray:
        """Best J seen after each iteration."""
        return np.array([it.incumbent_J for it in self.iterations])

    @property
    def all_J(self) -> np.ndarray:
        return np.array([it.J for it in self.iterations])

    @property
    def all_theta(self) -> np.ndarray:
        return np.array([it.theta for it in self.iterations])


# ──────────────────────────────────────────────────────────────────────────────
# Main optimiser class
# ──────────────────────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Bayesian optimiser wrapping BoTorch.

    Parameters
    ----------
    objective_fn : callable
        f(theta: np.ndarray) -> dict   with at least key 'J' (float, to minimise)
        theta is in natural parameter space (not normalised).
        The objective should call toy2dac and return evaluate_result().
    cfg : BOConfig
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], Dict],
        cfg: BOConfig = BOConfig(),
    ):
        self.objective_fn = objective_fn
        self.cfg          = cfg
        self.log_dir      = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # BoTorch unit-cube bounds  (DIM × 2 tensor)
        self._bounds = torch.zeros(2, DIM)
        self._bounds[1] = 1.0             # upper = 1 (normalised space)

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> BOResult:
        """
        Execute the full BO loop:
          1. LHS warm-start
          2. BO iterations (fit GP → optimise EI → evaluate → update)
        """
        cfg = self.cfg
        iterations: List[BOIteration] = []

        # ── Warm start (LHS) ──────────────────────────────────────────────────
        logger.info(f"BO warm-start: {cfg.n_init} LHS samples")
        lhs_thetas = lhs_sample(cfg.n_init, seed=cfg.lhs_seed)

        train_X_list: List[np.ndarray] = []
        train_Y_list: List[float]      = []
        incumbent_J  = float("inf")

        for i, theta in enumerate(lhs_thetas):
            t_start = time.perf_counter()
            metrics = self._safe_evaluate(theta)
            J       = metrics["J"]
            elapsed = time.perf_counter() - t_start

            train_X_list.append(normalize(theta))
            train_Y_list.append(J)

            if J < incumbent_J:
                incumbent_J   = J
                best_theta    = theta.copy()
                best_metrics  = metrics.copy()

            it = BOIteration(
                iteration=i, theta=theta, J=J,
                metrics=metrics, incumbent_J=incumbent_J,
                acq_value=float("nan"), wall_time_s=elapsed,
            )
            iterations.append(it)
            self._log_iteration(it)
            logger.info(f"  LHS {i:02d}/{cfg.n_init}  J={J:.6f}  "
                        f"best={incumbent_J:.6f}  [{elapsed:.1f}s]")

        # Convert to tensors for BoTorch
        train_X = torch.tensor(np.array(train_X_list))           # (n, 5)
        train_Y = torch.tensor(np.array(train_Y_list)).unsqueeze(-1)  # (n, 1)

        # ── BO loop ───────────────────────────────────────────────────────────
        logger.info(f"BO loop: {cfg.n_iter} iterations  acq={cfg.acq_type}")
        gp_model = None

        for bo_i in range(cfg.n_iter):
            # Fit / refit GP
            gp_model = self._fit_gp(train_X, train_Y)

            # Optimise acquisition → next candidate
            theta_norm, acq_val = self._next_candidate(gp_model, train_Y)
            theta               = denormalize(theta_norm.cpu().numpy().squeeze())

            # Early stopping
            if acq_val < cfg.ei_threshold:
                logger.info(f"EI={acq_val:.2e} < threshold={cfg.ei_threshold:.2e} → stopping early.")
                break

            # Evaluate FWI
            t_start = time.perf_counter()
            metrics = self._safe_evaluate(theta)
            J       = metrics["J"]
            elapsed = time.perf_counter() - t_start

            # Update training data
            x_new = torch.tensor(theta_norm.cpu().numpy()).squeeze().unsqueeze(0)
            y_new = torch.tensor([[J]])
            train_X = torch.cat([train_X, x_new], dim=0)
            train_Y = torch.cat([train_Y, y_new], dim=0)

            # Update incumbent
            if J < incumbent_J:
                incumbent_J  = J
                best_theta   = theta.copy()
                best_metrics = metrics.copy()

            iter_idx = cfg.n_init + bo_i
            it = BOIteration(
                iteration=iter_idx, theta=theta, J=J,
                metrics=metrics, incumbent_J=incumbent_J,
                acq_value=float(acq_val), wall_time_s=elapsed,
            )
            iterations.append(it)
            self._log_iteration(it)
            logger.info(f"  BO {bo_i:02d}/{cfg.n_iter}  J={J:.6f}  "
                        f"best={incumbent_J:.6f}  EI={acq_val:.4f}  [{elapsed:.1f}s]")

            # Checkpoint
            if (bo_i + 1) % cfg.checkpoint_every == 0:
                self._checkpoint(iterations, train_X, train_Y)

        return BOResult(
            best_theta=best_theta,
            best_J=incumbent_J,
            best_metrics=best_metrics,
            iterations=iterations,
            gp_model=gp_model,
            train_X=train_X,
            train_Y=train_Y,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GP fitting
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_gp(self, train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
        """
        Fit a SingleTaskGP with Matérn-ν kernel and automatic input/output
        normalisation.
        """
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=ScaleKernel(
                MaternKernel(nu=self.cfg.nu, ard_num_dims=DIM)
            ),
            input_transform=Normalize(d=DIM),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return model

    # ──────────────────────────────────────────────────────────────────────────
    # Acquisition optimisation
    # ──────────────────────────────────────────────────────────────────────────

    def _next_candidate(
        self, model: SingleTaskGP, train_Y: Tensor
    ) -> Tuple[Tensor, float]:
        """
        Maximise the acquisition function to find the next schedule to evaluate.
        Returns the candidate in *normalised* [0,1]^5 space.
        """
        cfg    = self.cfg
        best_f = train_Y.min().item()          # incumbent (lower is better)

        # LogExpectedImprovement is numerically stable and eliminates
        # the NumericsWarning from vanilla EI on near-zero EI values.
        # It returns log(EI); we exponentiate for display & thresholding.
        if cfg.acq_type in ("EI", "LogEI"):
            acqf = LogExpectedImprovement(
                model=model, best_f=best_f - cfg.xi, maximize=False
            )
        elif cfg.acq_type in ("qEI", "qLogEI"):
            acqf = qLogExpectedImprovement(model=model, best_f=best_f - cfg.xi)
        else:
            raise ValueError(f"Unknown acq_type: {cfg.acq_type!r}")

        candidate, log_acq = optimize_acqf(
            acq_function=acqf,
            bounds=self._bounds,
            q=cfg.q,
            num_restarts=cfg.acq_restarts,
            raw_samples=cfg.acq_samples,
        )
        # Back to linear EI for comparison with ei_threshold and for logging
        acq_value = float(torch.exp(log_acq).item())
        return candidate, acq_value

    # ──────────────────────────────────────────────────────────────────────────
    # GP surrogate inspection (for paper figures)
    # ──────────────────────────────────────────────────────────────────────────

    def posterior_on_grid(
        self,
        model: SingleTaskGP,
        param_i: int,
        param_j: int,
        n_grid: int = 30,
        fixed_theta_norm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate GP posterior mean and std on a 2-D grid for params i and j,
        with all other parameters fixed to fixed_theta_norm (or 0.5 if None).

        Returns
        -------
        xi_grid, xj_grid : (n_grid,) arrays in natural units
        mu_grid, std_grid : (n_grid, n_grid) arrays
        """
        if fixed_theta_norm is None:
            fixed_theta_norm = np.full(DIM, 0.5)

        xi_norm = np.linspace(0, 1, n_grid)
        xj_norm = np.linspace(0, 1, n_grid)
        XI, XJ  = np.meshgrid(xi_norm, xj_norm)

        # Build input tensor
        n_pts   = n_grid * n_grid
        X_flat  = np.tile(fixed_theta_norm, (n_pts, 1))
        X_flat[:, param_i] = XI.ravel()
        X_flat[:, param_j] = XJ.ravel()

        X_t = torch.tensor(X_flat)
        with torch.no_grad():
            post    = model.posterior(X_t)
            mu      = post.mean.cpu().numpy().reshape(n_grid, n_grid)
            std     = post.variance.sqrt().cpu().numpy().reshape(n_grid, n_grid)

        xi_natural = denormalize(
            np.stack([xi_norm, np.zeros_like(xi_norm), np.zeros_like(xi_norm),
                      np.zeros_like(xi_norm), np.zeros_like(xi_norm)], axis=-1)
        )[:, param_i]
        xj_natural = denormalize(
            np.stack([np.zeros_like(xj_norm), xj_norm, np.zeros_like(xj_norm),
                      np.zeros_like(xj_norm), np.zeros_like(xj_norm)], axis=-1)
        )[:, param_j]

        return xi_natural, xj_natural, mu, std

    def parameter_importances(self, model: SingleTaskGP) -> Dict[str, float]:
        """
        Proxy for parameter importance via the fitted GP length scales.
        Inverse length scale ∝ sensitivity: large 1/ℓ means the output
        changes quickly along that dimension.
        """
        try:
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            ls = ls.squeeze()
            inv_ls = 1.0 / (ls + 1e-10)
            inv_ls = inv_ls / inv_ls.sum()
            return dict(zip(PARAM_NAMES, inv_ls.tolist()))
        except Exception:
            return {n: float("nan") for n in PARAM_NAMES}

    # ──────────────────────────────────────────────────────────────────────────
    # Safe objective evaluation (catches exceptions from FWI)
    # ──────────────────────────────────────────────────────────────────────────

    def _safe_evaluate(self, theta: np.ndarray) -> Dict:
        try:
            metrics = self.objective_fn(theta)
        except Exception as e:
            logger.warning(f"Objective evaluation failed for theta={theta}: {e}")
            # Return a high (bad) value so BO avoids this region
            metrics = {"J": 999.0, "rmse_ms": 9999.0, "r2": -9999.0,
                       "ssim": -1.0, "Ju": float("nan"), "error": str(e)}
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Logging & checkpointing
    # ──────────────────────────────────────────────────────────────────────────

    def _log_iteration(self, it: BOIteration) -> None:
        """Append iteration to a JSONL log file."""
        log_file = self.log_dir / "bo_iterations.jsonl"
        with log_file.open("a") as fh:
            fh.write(json.dumps(it.to_dict()) + "\n")

    def _checkpoint(
        self,
        iterations: List[BOIteration],
        train_X: Tensor,
        train_Y: Tensor,
    ) -> None:
        ckpt = {
            "n_evaluated":   len(iterations),
            "best_J":        min(it.J for it in iterations),
            "train_X":       train_X.tolist(),
            "train_Y":       train_Y.squeeze().tolist(),
        }
        ckpt_path = self.log_dir / "checkpoint.json"
        ckpt_path.write_text(json.dumps(ckpt, indent=2))
        logger.debug(f"Checkpoint saved → {ckpt_path}")

    @staticmethod
    def load_checkpoint(log_dir: str | Path) -> Optional[Dict]:
        """Load a saved checkpoint to resume a run."""
        p = Path(log_dir) / "checkpoint.json"
        if p.exists():
            return json.loads(p.read_text())
        return None
