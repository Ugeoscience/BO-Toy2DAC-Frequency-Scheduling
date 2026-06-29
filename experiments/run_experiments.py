"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

experiments/run_experiments.py

Execution (from bo_fwi root)
────────────────────────────
  python experiments/run_experiments.py --exp A            # real toy2dac run
  python experiments/run_experiments.py --exp A --mock     # pipeline test only
  python experiments/run_experiments.py --exp all          # full paper results

Experiment matrix
─────────────────
  A  BO vs expert schedule
  B  Sample efficiency: BO vs random vs grid
  C  Noise robustness: SNR = 10, 20, 40 dB
  D  Starting-model sensitivity: standard vs aggressively smoothed init
  E  Ablation: acquisition function (EI vs qEI) and Matérn ν
  F  Supervised vs unsupervised objective correlation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fwi.schedule        import make_schedule, expert_schedule, lhs_sample
from fwi.toy2dac_wrapper import (Toy2dacConfig, Toy2dacWrapper,
                                  sanity_check, _check_model_size)
from fwi.metrics         import (evaluate_result, smooth_model,
                                  add_noise_to_data)
from bo.bayesian_optimizer import BayesianOptimizer, BOConfig
from bo.baselines           import (RandomSearch, GridSearch, ExpertRunner,
                                     run_random_search_multi_seed)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments")

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration — all real paths filled in
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = ""
_BIN      = ""
_BO_ROOT  = ""

TOY2DAC_CFG = Toy2dacConfig(
    toy2dac_bin      = _BIN,
    base_data_dir    = _TEMPLATE,
    true_model_file  = f"{_TEMPLATE}/vp_Marmousi_exact",
    init_model_file  = f"{_TEMPLATE}/vp_Marmousi_init",
    density_file     = f"{_TEMPLATE}/rho",
    acq_file         = f"{_TEMPLATE}/acqui",
    nx               = 681,
    nz               = 141,
    dx               = 25.0,
    dz               = 25.0,
    mpi_np           = 4,          # ← change to match your cluster allocation
    mpi_launcher     = "mpirun",   # ← use "srun" on SLURM clusters
    n_threads        = 1,
)

BO_CFG = BOConfig(
    n_init           = 10,    # LHS warm-start evaluations
    n_iter           = 20,    # BO iterations after warm-start
    lhs_seed         = 42,
    acq_type         = "EI",
    q                = 1,
    acq_restarts     = 10,
    acq_samples      = 512,
    nu               = 2.5,
    log_dir          = f"{_BO_ROOT}/results/logs",
    checkpoint_every = 5,
)

RESULTS_DIR  = Path(f"{_BO_ROOT}/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_BUDGET  = BO_CFG.n_init + BO_CFG.n_iter   # 50 by default
SNR_LEVELS_DB = [10, 20, 40]
RS_SEEDS      = [0, 1, 2, 3, 4]

# Expert schedule: Marmousi @ 25 m grid, source dominant freq ~5–15 Hz
# Three groups that any experienced practitioner would choose.
EXPERT_GROUPS = [
    (3.0, 5.0),    # low   — builds background
    (4.0, 10.0),   # mid   — refines interfaces
    (8.0, 15.0),   # high  — sharpens details
]

# ─────────────────────────────────────────────────────────────────────────────
# Poor starting model for Experiment D
# ─────────────────────────────────────────────────────────────────────────────

_POOR_INIT_PATH = Path(f"{_BO_ROOT}/results/vp_Marmousi_init_poor")

def ensure_poor_model(cfg: Toy2dacConfig) -> Path:
    """
    Create an aggressively smoothed starting model for Experiment D.

    Standard init  : vp_Marmousi_init  (smooth_length ≈ 200–300 m)
    Poor init      : applies extra 500 m Gaussian smoothing on top —
                     represents a practitioner starting from a 1D gradient
                     or heavily over-smoothed tomography result.

    The poor model is created once and cached; subsequent calls reuse it.
    """
    if _POOR_INIT_PATH.exists():
        logger.info(f"Poor init model already exists: {_POOR_INIT_PATH}")
        return _POOR_INIT_PATH

    logger.info("Creating poor starting model (extra 500 m Gaussian smoothing)…")
    init_model = np.frombuffer(
        Path(cfg.init_model_file).read_bytes(), dtype=np.float32
    ).reshape((cfg.nz, cfg.nx), order="F")

    # Additional aggressive smoothing on top of whatever init already has
    poor = smooth_model(init_model, sigma_m=500.0, dx=cfg.dx, dz=cfg.dz)

    _POOR_INIT_PATH.write_bytes(
        poor.astype(np.float32).ravel(order="F").tobytes()
    )
    logger.info(f"Poor model written to {_POOR_INIT_PATH}  "
                f"(vel range: {poor.min():.0f}–{poor.max():.0f} m/s)")
    return _POOR_INIT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Objective function factory
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    wrapper:    Toy2dacWrapper,
    run_prefix: str            = "run",
    snr_db:     Optional[float] = None,
) -> Callable[[np.ndarray], Dict]:
    """
    Returns a black-box callable  theta → metrics_dict  for the BO loop.

    Each call:
      1. theta → FrequencySchedule
      2. wrapper.run(schedule, snr_db=snr_db) — runs toy2dac (MODELING + FWI)
      3. returns evaluate_result() metrics including J (normalised RMSE)

    The snr_db argument is forwarded to wrapper.run(), which injects
    Gaussian noise into data_modeling before the inversion step.
    """
    counter = [0]

    def objective(theta: np.ndarray) -> Dict:
        counter[0] += 1
        run_id   = f"{run_prefix}_{counter[0]:04d}"
        schedule = make_schedule(theta)
        logger.info(f"  [{run_id}] Evaluating: {schedule.summary_dict()}")

        result = wrapper.run(schedule, run_id=run_id, snr_db=snr_db)

        if not result.success:
            logger.warning(f"  [{run_id}] FWI failed: {result.error_msg[:200]}")
            return {
                "J": 999.0, "rmse_ms": 9999.0,
                "r2": -1.0, "ssim": -1.0,
                "Ju": float("nan"),
                "error": result.error_msg,
            }

        true_model = wrapper.load_true_model()
        metrics    = evaluate_result(m_est=result.final_model, m_true=true_model)
        metrics.update({
            "schedule":    schedule.summary_dict(),
            "run_id":      run_id,
            "wall_time_s": result.wall_time_s,
            "n_groups":    result.n_groups,
        })
        logger.info(
            f"  [{run_id}] J={metrics['J']:.5f}  "
            f"RMSE={metrics['rmse_ms']:.1f} m/s  "
            f"SSIM={metrics['ssim']:.4f}  "
            f"[{result.wall_time_s:.0f}s]"
        )
        return metrics

    return objective


def make_mock_objective(seed: int = 42) -> Callable[[np.ndarray], Dict]:
    """
    Synthetic landscape for pipeline testing — no toy2dac needed.

    J = 0.10 + 0.50 · exp(−‖θ_norm − θ*‖² / 0.20) + ε,  ε ~ N(0, 0.01)
    Optimum at θ* = [3.0, 22.0, 5.0, 1.8, 0.15] (natural units).
    """
    import time as _t
    from fwi.schedule import normalize as _norm

    rng  = np.random.default_rng(seed)
    tstar = _norm(np.array([3.0, 22.0, 5.0, 1.8, 0.15]))

    def mock(theta: np.ndarray) -> Dict:
        t    = _norm(theta)
        dist = np.sum((t - tstar) ** 2)
        J    = float(np.clip(0.10 + 0.50 * np.exp(-dist / 0.20) + rng.normal(0, 0.01), 0.01, 1.0))
        _t.sleep(0.005)   # simulate IO latency
        return {"J": J, "rmse_ms": J*1000, "r2": 1-J, "ssim": 1-J,
                "Ju": float(np.clip(J*1.05 + rng.normal(0, 0.005), 0.01, 1.1))}

    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A — BO vs expert
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_A(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 62)
    logger.info("EXPERIMENT A  —  BO vs expert schedule")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_A"
    exp_dir.mkdir(exist_ok=True)

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expA")

    # 1 — BO
    bo_cfg    = _bo_cfg(exp_dir / "logs_bo")
    bo_result = BayesianOptimizer(obj, bo_cfg).run()

    # 2 — Expert
    expert_result = ExpertRunner(obj, f_min=3.0, f_max=20.0,
                                  expert_groups=EXPERT_GROUPS).run()

    _dump(exp_dir / "bo_result.json", {
        "best_theta":       bo_result.best_theta.tolist(),
        "best_J":           bo_result.best_J,
        "best_metrics":     _clean(bo_result.best_metrics),
        "incumbent_trace":  bo_result.incumbent_trace.tolist(),
        "all_J":            bo_result.all_J.tolist(),
        "all_theta":        bo_result.all_theta.tolist(),
        "param_importances": BayesianOptimizer(obj, bo_cfg).parameter_importances(
                                bo_result.gp_model) if bo_result.gp_model else {},
    })
    _dump(exp_dir / "expert_result.json", {
        "theta":   expert_result.best_theta.tolist(),
        "J":       expert_result.best_J,
        "metrics": _clean(expert_result.best_metrics),
    })

    logger.info(f"Exp A done.  BO J={bo_result.best_J:.5f}  "
                f"Expert J={expert_result.best_J:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B — Sample efficiency
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_B(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 62)
    logger.info("EXPERIMENT B  —  Sample efficiency (BO vs random vs grid)")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_B"
    exp_dir.mkdir(exist_ok=True)

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expB")

    # BO
    bo_result = BayesianOptimizer(obj, _bo_cfg(exp_dir / "logs_bo")).run()

    # Random search (5 seeds → mean ± std for reviewer credibility)
    rs_agg = run_random_search_multi_seed(obj, n_eval=TOTAL_BUDGET, seeds=RS_SEEDS)

    # Grid search  (≈ TOTAL_BUDGET evaluations)
    gs_result = GridSearch(
        obj,
        f_min_values=[2.0, 3.0, 5.0, 7.0],
        f_max_values=[15.0, 20.0, 25.0, 30.0],
        K_values=[3, 4, 5],
    ).run()

    _dump(exp_dir / "summary.json", {
        "bo_trace":         bo_result.incumbent_trace.tolist(),
        "rs_mean":          rs_agg["incumbent_mean"].tolist(),
        "rs_std":           rs_agg["incumbent_std"].tolist(),
        "rs_best_per_seed": rs_agg["best_J_all_seeds"].tolist(),
        "gs_trace":         gs_result.incumbent_trace.tolist(),
        "bo_best_J":        bo_result.best_J,
        "rs_best_J_mean":   float(rs_agg["best_J_all_seeds"].mean()),
        "gs_best_J":        gs_result.best_J,
    })
    logger.info(f"Exp B done.  BO={bo_result.best_J:.5f}  "
                f"RS={rs_agg['best_J_all_seeds'].mean():.5f}  "
                f"GS={gs_result.best_J:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C — Noise robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_C(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 62)
    logger.info(f"EXPERIMENT C  —  Noise robustness  SNR={SNR_LEVELS_DB} dB")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_C"
    exp_dir.mkdir(exist_ok=True)

    for snr in SNR_LEVELS_DB:
        logger.info(f"  ── SNR = {snr} dB ──")
        snr_dir = exp_dir / f"snr_{snr}dB"
        snr_dir.mkdir(exist_ok=True)

        # snr_db is passed to wrapper.run() → injected after MODELING step
        obj = (make_mock_objective(seed=snr) if mock
               else make_objective(wrapper, run_prefix=f"expC_snr{snr}", snr_db=snr))

        bo_result     = BayesianOptimizer(obj, _bo_cfg(snr_dir / "logs_bo")).run()
        expert_result = ExpertRunner(obj, f_min=3.0, f_max=20.0,
                                      expert_groups=EXPERT_GROUPS).run()

        _dump(snr_dir / "results.json", {
            "snr_db":         snr,
            "bo_J":           bo_result.best_J,
            "expert_J":       expert_result.best_J,
            "bo_metrics":     _clean(bo_result.best_metrics),
            "expert_metrics": _clean(expert_result.best_metrics),
            "bo_trace":       bo_result.incumbent_trace.tolist(),
        })
        logger.info(f"  SNR={snr} dB  BO={bo_result.best_J:.5f}  "
                    f"Expert={expert_result.best_J:.5f}")

    logger.info("Exp C done.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D — Starting-model sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_D(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """
    Compare BO performance with two starting models:
      good : vp_Marmousi_init  (standard smoothed model from template)
      poor : extra 500 m Gaussian smoothing applied on top

    For the real run, we create two separate Toy2dacWrapper instances —
    one pointing to the standard init, one to the poor init.
    """
    logger.info("═" * 62)
    logger.info("EXPERIMENT D  —  Starting-model sensitivity")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_D"
    exp_dir.mkdir(exist_ok=True)

    configs = {
        "good": TOY2DAC_CFG,          # standard init (vp_Marmousi_init)
        "poor": None,                  # built below
    }

    if not mock:
        poor_path = ensure_poor_model(TOY2DAC_CFG)
        poor_cfg  = dataclasses.replace(TOY2DAC_CFG, init_model_file=str(poor_path))
        configs["poor"] = poor_cfg

    for quality, cfg in configs.items():
        logger.info(f"  ── Starting model: {quality} ──")
        subdir = exp_dir / quality
        subdir.mkdir(exist_ok=True)

        if mock:
            obj = make_mock_objective(seed=99 if quality == "poor" else 42)
        else:
            w   = Toy2dacWrapper(cfg, work_root=RESULTS_DIR / "runs" / f"expD_{quality}")
            obj = make_objective(w, run_prefix=f"expD_{quality}")

        bo_result = BayesianOptimizer(obj, _bo_cfg(subdir / "logs_bo")).run()

        _dump(subdir / "result.json", {
            "model_quality": quality,
            "best_J":        bo_result.best_J,
            "best_theta":    bo_result.best_theta.tolist(),
            "best_metrics":  _clean(bo_result.best_metrics),
            "incumbent_trace": bo_result.incumbent_trace.tolist(),
        })
        logger.info(f"  {quality}  BO J={bo_result.best_J:.5f}")

    logger.info("Exp D done.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment E — Ablation: acquisition function and Matérn ν
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_E(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 62)
    logger.info("EXPERIMENT E  —  Ablation (acq function + kernel)")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_E"
    exp_dir.mkdir(exist_ok=True)

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expE")

    variants = [
        {"acq_type": "EI",  "nu": 2.5, "label": "EI_Matern25"},
        {"acq_type": "EI",  "nu": 1.5, "label": "EI_Matern15"},
        {"acq_type": "qEI", "nu": 2.5, "label": "qEI_Matern25"},
    ]

    for var in variants:
        label = var["label"]
        logger.info(f"  ── Variant: {label} ──")
        overrides = {k: v for k, v in var.items() if k != "label"}
        cfg_v     = dataclasses.replace(BO_CFG, **overrides,
                                        log_dir=str(exp_dir / f"logs_{label}"))
        bo_result = BayesianOptimizer(obj, cfg_v).run()

        _dump(exp_dir / f"{label}.json", {
            "label":           label,
            "acq_type":        var["acq_type"],
            "nu":              var["nu"],
            "best_J":          bo_result.best_J,
            "incumbent_trace": bo_result.incumbent_trace.tolist(),
        })
        logger.info(f"  {label}  best J={bo_result.best_J:.5f}")

    logger.info("Exp E done.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment F — Supervised J vs unsupervised Ju correlation
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_F(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """
    Sample 30 schedules via LHS, run each FWI, record J (supervised RMSE)
    and Ju (held-out data residual).  Report Pearson r.

    NOTE: Ju requires running toy2dac in MODELING mode with the *inverted* model
    at a held-out frequency that was NOT in the schedule.  The implementation
    below computes Ju via the normalised residual in the inverted-model's
    data space.  You may extend this to true held-out frequencies by adding
    a third MODELING run after INVERSION — see the comment in the code.
    """
    logger.info("═" * 62)
    logger.info("EXPERIMENT F  —  J vs Ju correlation")
    logger.info("═" * 62)
    exp_dir = RESULTS_DIR / "exp_F"
    exp_dir.mkdir(exist_ok=True)

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expF")

    n_samp   = 30
    thetas   = lhs_sample(n_samp, seed=777)
    J_vals:  List[float] = []
    Ju_vals: List[float] = []

    for i, theta in enumerate(thetas):
        metrics = obj(theta)
        J_vals.append(metrics["J"])
        Ju_vals.append(metrics.get("Ju", float("nan")))
        logger.info(f"  F {i+1:02d}/{n_samp}  J={metrics['J']:.5f}  "
                    f"Ju={metrics.get('Ju', float('nan')):.5f}")

    J  = np.array(J_vals)
    Ju = np.array(Ju_vals)
    valid = np.isfinite(J) & np.isfinite(Ju)
    r = float(np.corrcoef(J[valid], Ju[valid])[0, 1]) if valid.sum() > 2 else float("nan")

    _dump(exp_dir / "correlation.json", {
        "J_values":  J.tolist(),
        "Ju_values": Ju.tolist(),
        "pearson_r": r,
        "n_valid":   int(valid.sum()),
    })
    logger.info(f"Exp F done.  Pearson r(J, Ju) = {r:.4f}  (n_valid={valid.sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bo_cfg(log_dir: Path) -> BOConfig:
    """Clone the global BOConfig with a different log_dir."""
    return dataclasses.replace(BO_CFG, log_dir=str(log_dir))


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _clean(d: Dict) -> Dict:
    """Remove non-serialisable keys (numpy arrays nested in metrics)."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in d.items() if not isinstance(v, np.ndarray) or v.ndim == 0}


def _build_wrapper() -> Toy2dacWrapper:
    """Pre-flight check + wrapper construction."""
    ok = sanity_check(TOY2DAC_CFG)
    if not ok:
        raise RuntimeError(
            "Sanity check failed — fix the paths in TOY2DAC_CFG at the top of "
            "run_experiments.py, then re-run."
        )
    return Toy2dacWrapper(TOY2DAC_CFG, work_root=RESULTS_DIR / "runs")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A": run_experiment_A,
    "B": run_experiment_B,
    "C": run_experiment_C,
    "D": run_experiment_D,
    "E": run_experiment_E,
    "F": run_experiment_F,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="BO-FWI manuscript experiments")
    parser.add_argument("--exp",  default="A",
                        choices=list(EXPERIMENTS) + ["all"],
                        help="Experiment to run (default: A)")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic objective — no toy2dac needed")
    parser.add_argument("--mpi-np", type=int, default=None,
                        help="Override number of MPI ranks (default: from TOY2DAC_CFG)")
    args = parser.parse_args()

    # Optional MPI override from command line
    if args.mpi_np is not None:
        global TOY2DAC_CFG
        TOY2DAC_CFG = dataclasses.replace(TOY2DAC_CFG, mpi_np=args.mpi_np)
        logger.info(f"MPI ranks overridden: mpi_np={args.mpi_np}")

    wrapper = None
    if not args.mock:
        wrapper = _build_wrapper()
        logger.info(f"toy2dac wrapper ready  "
                    f"(mpirun -n {TOY2DAC_CFG.mpi_np}  "
                    f"grid={TOY2DAC_CFG.nx}×{TOY2DAC_CFG.nz})")

    t0 = time.perf_counter()
    if args.exp == "all":
        for name, fn in EXPERIMENTS.items():
            fn(wrapper, mock=args.mock)
    else:
        EXPERIMENTS[args.exp](wrapper, mock=args.mock)

    logger.info(f"Total wall time: {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
