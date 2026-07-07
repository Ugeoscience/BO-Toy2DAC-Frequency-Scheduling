"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : BSD 3-Clause License. Copyright (c) 2026, Ugeoscience. Redistribution and use in source and binary forms, with or without modification, are permitted provided that copyright notice, license conditions, and disclaimer are retained. Neither Ugeoscience nor contributor names may be used to endorse or promote derived products without prior written permission. This software is provided "AS IS", without warranties or liability.

experiments/run_experiments.py
───────────────────────────────
Authoritative experiment runner for the BO-FWI manuscript.

This driver reproduces the manuscript's experiment matrix at the budgets below
(239 FWI evaluations total). BO is run ONCE (Experiment A) and reused by the
other parts, so it is never double-counted.

    Part          Method                        Budget
    ────────────  ───────────────────────────   ───────────────────────────────
    Experiment A  Bayesian optimization         30 evaluations
    Experiment A  Expert-style heuristic         1 evaluation
    Experiment B  Random search                 30 × 5 seeds = 150 runs
    Experiment B  Grid search                   30 evaluations
    Experiment C  BO-best vs expert + noise     3 SNR × 2 sched × 3 seeds = 18
    Experiment D  Poor-start LHS screening      10 evaluations
    ────────────  ───────────────────────────   ───────────────────────────────
    Total                                       239 FWI runs

Paths (confirmed)
─────────────────
  toy2dac binary : /home/mau0009/softwares/TOY2DAC_V2.6_BO/bin/toy2dac
  template dir   : /home/mau0009/softwares/TOY2DAC_V2.6_BO/run_marmousi_template
  bo_fwi root    : /Data2/Ali/NewStudy_Bayesian/bo_fwi_Claude

Execution (from bo_fwi root)
────────────────────────────
  python experiments/run_experiments.py --exp A      # BO + expert      (31 runs)
  python experiments/run_experiments.py --exp B      # RS 150 + grid 30 (180 runs, needs A)
  python experiments/run_experiments.py --exp C      # noise            (18 runs, needs A)
  python experiments/run_experiments.py --exp D      # poor-start LHS   (10 runs, needs A)
  python experiments/run_experiments.py --exp all    # A → B → C → D    (239 runs)
  python experiments/run_experiments.py --exp all --mock   # pipeline test, no toy2dac

Notes
─────
  * "all" runs ONLY the paper experiments A, B, C, D. The optional ablations
    E (acquisition/kernel) and F (J vs Ju) are still runnable individually.
  * Experiments B, C, D LOAD the BO result from Experiment A; run A first
    (or use --exp all, which runs A before the others).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
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
# Global configuration — real paths filled in
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = "/home/mau0009/softwares/TOY2DAC_V2.6_BO/run_marmousi_template"
_BIN      = "/home/mau0009/softwares/TOY2DAC_V2.6_BO/bin/toy2dac"
_BO_ROOT  = "/Data2/Ali/NewStudy_Bayesian/bo_fwi_Claude"

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
    n_iter           = 20,    # BO iterations after warm-start  →  30 total
    lhs_seed         = 42,
    acq_type         = "EI",
    q                = 1,
    acq_restarts     = 10,
    acq_samples      = 512,
    nu               = 2.5,
    ei_threshold     = 1e-6,  # conservative; the validated run used the full 30
    log_dir          = f"{_BO_ROOT}/results/logs",
    checkpoint_every = 5,
)

RESULTS_DIR  = Path(f"{_BO_ROOT}/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Budgets that define the 239-run plan ─────────────────────────────────────
BO_BUDGET      = BO_CFG.n_init + BO_CFG.n_iter   # = 30  (Experiment A)
RS_BUDGET      = BO_BUDGET                        # equal-budget random search
RS_SEEDS       = [0, 1, 2, 3, 4]                  # 30 × 5 = 150 runs (Experiment B)
SNR_LEVELS_DB  = [10, 20, 40]                     # Experiment C
NOISE_SEEDS    = [0, 1, 2]                        # 3 noise realizations per SNR
D_LHS_BUDGET   = 10                               # Experiment D screening

# ── Canonical expert-style heuristic schedule (Table 3) ──────────────────────
# Linearly-spaced K=3 continuation: centers 3.0 / 11.5 / 20.0 Hz, group
# boundaries at the midpoints 7.25 / 15.75 Hz. Decodes through make_schedule()
# to exactly the groups reported in Table 3, so the repo reproduces J_expert.
EXPERT_THETA  = np.array([3.0, 20.0, 3.0, 1.0, 0.0])

# Deprecated: kept only for backward-compatible imports. NOT used as the
# baseline anymore (it produced overlapping thirds that do not match Table 3).
EXPERT_GROUPS = [(3.0, 5.0), (4.0, 10.0), (8.0, 15.0)]

# ── Grid for Experiment B: exactly 30 points, no f_min ≥ f_max skips ─────────
# Coarse, physically-motivated sweep over the interpretable axes, gamma/beta
# fixed (the manuscript's "practical low-budget manual search"). 5 × 3 × 2 = 30.
GRID_FMIN   = [2.0, 3.0, 4.0, 5.0, 7.0]
GRID_FMAX   = [15.0, 20.0, 25.0]
GRID_K      = [3, 5]
GRID_GAMMA  = 1.5
GRID_BETA   = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Small NaN-safe JSON loader (Exp B/C/D read Exp A's outputs)
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(re.sub(r"\bNaN\b", "null", path.read_text()))


# ─────────────────────────────────────────────────────────────────────────────
# Poor starting model for Experiment D
# ─────────────────────────────────────────────────────────────────────────────

_POOR_INIT_PATH = Path(f"{_BO_ROOT}/results/vp_Marmousi_init_poor")

def ensure_poor_model(cfg: Toy2dacConfig) -> Path:
    """
    Create an aggressively smoothed starting model for Experiment D.

    Standard init : vp_Marmousi_init  (≈ 300 m smoothing of the true model)
    Poor init     : extra 500 m Gaussian smoothing applied on top.
    Created once and cached.
    """
    if _POOR_INIT_PATH.exists():
        logger.info(f"Poor init model already exists: {_POOR_INIT_PATH}")
        return _POOR_INIT_PATH

    logger.info("Creating poor starting model (extra 500 m Gaussian smoothing)…")
    init_model = np.frombuffer(
        Path(cfg.init_model_file).read_bytes(), dtype=np.float32
    ).reshape((cfg.nz, cfg.nx), order="F")

    poor = smooth_model(init_model, sigma_m=500.0, dx=cfg.dx, dz=cfg.dz)

    _POOR_INIT_PATH.write_bytes(poor.astype(np.float32).ravel(order="F").tobytes())
    logger.info(f"Poor model written to {_POOR_INIT_PATH}  "
                f"(vel range: {poor.min():.0f}–{poor.max():.0f} m/s)")
    return _POOR_INIT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Objective function factory
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    wrapper:    Toy2dacWrapper,
    run_prefix: str             = "run",
    snr_db:     Optional[float] = None,
    noise_seed: int             = 42,
) -> Callable[[np.ndarray], Dict]:
    """
    Black-box callable  theta → metrics_dict  for the BO loop.

    snr_db     : if set, Gaussian noise at that SNR is injected into
                 data_modeling before inversion (Experiment C).
    noise_seed : selects the noise realization (Experiment C); ignored when
                 snr_db is None.
    """
    counter = [0]

    def objective(theta: np.ndarray) -> Dict:
        counter[0] += 1
        run_id   = f"{run_prefix}_{counter[0]:04d}"
        schedule = make_schedule(theta)
        logger.info(f"  [{run_id}] Evaluating: {schedule.summary_dict()}")

        result = wrapper.run(schedule, run_id=run_id,
                             snr_db=snr_db, noise_seed=noise_seed)

        if not result.success:
            logger.warning(f"  [{run_id}] FWI failed: {result.error_msg[:200]}")
            return {"J": 999.0, "rmse_ms": 9999.0, "r2": -1.0, "ssim": -1.0,
                    "Ju": float("nan"), "error": result.error_msg}

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


def make_mock_objective(seed: int = 42, snr_db: Optional[float] = None
                        ) -> Callable[[np.ndarray], Dict]:
    """
    Synthetic landscape for pipeline testing — no toy2dac needed.

    J = 0.10 + 0.50·exp(−‖θ_norm − θ*‖²/0.20) + ε   (ε ~ N(0, 0.01))
    A small SNR-dependent penalty is added when snr_db is set so that the
    Experiment-C control flow can be exercised in --mock runs.
    """
    import time as _t
    from fwi.schedule import normalize as _norm

    rng   = np.random.default_rng(seed)
    tstar = _norm(np.array([3.0, 22.0, 5.0, 1.8, 0.15]))
    penalty = 0.0 if snr_db is None else 0.02 * (40.0 - float(snr_db)) / 30.0

    def mock(theta: np.ndarray) -> Dict:
        t    = _norm(theta)
        dist = np.sum((t - tstar) ** 2)
        J    = float(np.clip(0.10 + 0.50 * np.exp(-dist / 0.20)
                             + penalty + rng.normal(0, 0.01), 0.01, 1.0))
        _t.sleep(0.005)
        return {"J": J, "rmse_ms": J * 1000, "r2": 1 - J, "ssim": 1 - J,
                "Ju": float(np.clip(J * 1.05 + rng.normal(0, 0.005), 0.01, 1.1))}

    return mock


# ═════════════════════════════════════════════════════════════════════════════
# Experiment A — BO (30) vs expert-style heuristic (1)
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment_A(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 64)
    logger.info("EXPERIMENT A  —  BO (30 evals) vs expert-style heuristic (1 eval)")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_A"
    exp_dir.mkdir(exist_ok=True)

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expA")

    # 1 — Bayesian optimization (10 LHS warm-start + 20 GP-guided = 30 evals)
    bo_cfg    = _bo_cfg(exp_dir / "logs_bo")
    bo_result = BayesianOptimizer(obj, bo_cfg).run()

    # 2 — Expert-style heuristic baseline (Table 3 schedule), single evaluation
    logger.info("  Evaluating expert-style heuristic schedule (Table 3) …")
    expert_metrics = obj(EXPERT_THETA)

    importances = {}
    if getattr(bo_result, "gp_model", None) is not None:
        try:
            importances = BayesianOptimizer(obj, bo_cfg).parameter_importances(
                bo_result.gp_model)
        except Exception as exc:
            logger.warning(f"  parameter_importances failed: {exc}")

    _dump(exp_dir / "bo_result.json", {
        "best_theta":        bo_result.best_theta.tolist(),
        "best_J":            bo_result.best_J,
        "best_metrics":      _clean(bo_result.best_metrics),
        "incumbent_trace":   bo_result.incumbent_trace.tolist(),
        "all_J":             bo_result.all_J.tolist(),
        "all_theta":         bo_result.all_theta.tolist(),
        "n_init":            BO_CFG.n_init,
        "param_importances": importances,
    })
    _dump(exp_dir / "expert_result.json", {
        "theta":   EXPERT_THETA.tolist(),
        "J":       expert_metrics["J"],
        "metrics": _clean(expert_metrics),
    })

    logger.info(f"Exp A done.  BO J={bo_result.best_J:.5f}  "
                f"Expert J={expert_metrics['J']:.5f}  "
                f"(LHS warm-start best J={min(bo_result.all_J[:BO_CFG.n_init]):.5f})")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment B — Sample efficiency: RS (150) + grid (30); BO reused from A
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment_B(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 64)
    logger.info("EXPERIMENT B  —  RS (30×5=150) + grid (30); BO reused from Exp A")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_B"
    exp_dir.mkdir(exist_ok=True)

    bo_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if bo_data is None:
        logger.error("Exp A bo_result.json not found — run Exp A first.")
        return
    bo_trace  = np.array(bo_data["incumbent_trace"])
    bo_best_J = bo_data["best_J"]
    ex_data   = load_json(RESULTS_DIR / "exp_A" / "expert_result.json")
    expert_J  = ex_data["J"] if ex_data else None
    logger.info(f"  BO (Exp A):  best_J={bo_best_J:.5f} in {len(bo_trace)} evals — no new BO runs")
    logger.info(f"  Expert (Exp A):  J={expert_J}")

    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expB")

    # Random search — equal budget (30) × 5 seeds = 150 new FWI runs
    logger.info(f"  Random search: {len(RS_SEEDS)} seeds × {RS_BUDGET} evals = "
                f"{len(RS_SEEDS) * RS_BUDGET} runs")
    rs_agg = run_random_search_multi_seed(obj, n_eval=RS_BUDGET, seeds=RS_SEEDS)

    # Grid search — exactly 30 points
    gs = GridSearch(obj,
                    f_min_values=GRID_FMIN, f_max_values=GRID_FMAX, K_values=GRID_K,
                    gamma_fixed=GRID_GAMMA, beta_fixed=GRID_BETA)
    assert gs.grid_size == 30, f"grid has {gs.grid_size} points, expected 30"
    logger.info(f"  Grid search: {gs.grid_size} points "
                f"(gamma={GRID_GAMMA}, beta={GRID_BETA})")
    gs_result = gs.run()

    rs_best = rs_agg["best_J_all_seeds"]
    _dump(exp_dir / "summary.json", {
        "budget":           RS_BUDGET,
        "bo_trace":         bo_trace.tolist(),
        "bo_best_J":        bo_best_J,
        "rs_mean":          rs_agg["incumbent_mean"].tolist(),
        "rs_std":           rs_agg["incumbent_std"].tolist(),
        "rs_best_per_seed": rs_best.tolist(),
        "rs_seeds":         RS_SEEDS,
        "rs_best_J_mean":   float(rs_best.mean()),
        "rs_best_J_std":    float(rs_best.std()),
        "gs_trace":         gs_result.incumbent_trace.tolist(),
        "gs_best_J":        gs_result.best_J,
        "expert_J":         expert_J,
    })

    logger.info("Exp B summary:")
    logger.info(f"  BO   J = {bo_best_J:.5f}   (30 evals, from Exp A)")
    logger.info(f"  RS   J = {rs_best.mean():.5f} ± {rs_best.std():.5f}  "
                f"({len(RS_SEEDS)} seeds × {RS_BUDGET})")
    logger.info(f"  GS   J = {gs_result.best_J:.5f}  ({gs_result.n_eval} pts)")
    if expert_J:
        logger.info(f"  Expert J = {expert_J:.5f}")
    logger.info(f"  BO vs RS-mean: "
                f"{(rs_best.mean() - bo_best_J) / rs_best.mean() * 100:+.1f}%   "
                f"BO vs GS: {(gs_result.best_J - bo_best_J) / gs_result.best_J * 100:+.1f}%")
    logger.info("Exp B done.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment C — noise transfer: fixed BO-best vs expert, 3 SNR × 3 seeds × 2
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment_C(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 64)
    logger.info(f"EXPERIMENT C  —  Noise transfer  SNR={SNR_LEVELS_DB} dB × "
                f"{len(NOISE_SEEDS)} realizations × 2 schedules "
                f"= {len(SNR_LEVELS_DB)*len(NOISE_SEEDS)*2} runs")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_C"
    exp_dir.mkdir(exist_ok=True)

    bo_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if bo_data is None:
        logger.error("Exp A bo_result.json not found — run Exp A first.")
        return
    best_theta = np.array(bo_data["best_theta"])
    logger.info(f"  BO best theta (fixed, from Exp A): "
                f"{[round(x, 3) for x in best_theta.tolist()]}")
    logger.info(f"  Expert theta (fixed): {EXPERT_THETA.tolist()}")

    per_run: List[Dict] = []
    per_snr: List[Dict] = []

    for snr in SNR_LEVELS_DB:
        bo_Js, ex_Js = [], []
        logger.info(f"\n  ── SNR = {snr} dB ──")
        for seed in NOISE_SEEDS:
            if mock:
                obj = make_mock_objective(seed=1000 * snr + seed, snr_db=snr)
            else:
                obj = make_objective(wrapper,
                                     run_prefix=f"expC_snr{snr}_s{seed}",
                                     snr_db=float(snr), noise_seed=seed)
            m_bo = obj(best_theta)
            m_ex = obj(EXPERT_THETA)
            bo_Js.append(m_bo["J"]); ex_Js.append(m_ex["J"])
            row = {"snr_db": snr, "noise_seed": seed,
                   "bo_J": m_bo["J"], "expert_J": m_ex["J"],
                   "bo_rmse_ms": m_bo.get("rmse_ms"),
                   "expert_rmse_ms": m_ex.get("rmse_ms"),
                   "bo_ssim": m_bo.get("ssim"), "expert_ssim": m_ex.get("ssim"),
                   "improvement_pct": (m_ex["J"] - m_bo["J"]) / m_ex["J"] * 100}
            per_run.append(row)
            logger.info(f"    seed={seed}:  BO={m_bo['J']:.5f}  "
                        f"Expert={m_ex['J']:.5f}  Δ={row['improvement_pct']:+.1f}%")

        bo_Js, ex_Js = np.array(bo_Js), np.array(ex_Js)
        impr = (ex_Js - bo_Js) / ex_Js * 100.0
        summary = {"snr_db": snr,
                   "bo_J_mean": float(bo_Js.mean()), "bo_J_std": float(bo_Js.std()),
                   "expert_J_mean": float(ex_Js.mean()), "expert_J_std": float(ex_Js.std()),
                   "improvement_pct_mean": float(impr.mean()),
                   "improvement_pct_std": float(impr.std()),
                   "n_seeds": len(NOISE_SEEDS)}
        per_snr.append(summary)
        logger.info(f"    SNR={snr} dB:  BO={bo_Js.mean():.5f}±{bo_Js.std():.5f}  "
                    f"Expert={ex_Js.mean():.5f}±{ex_Js.std():.5f}  "
                    f"Δ={impr.mean():.1f}±{impr.std():.1f}%")

    _dump(exp_dir / "all_runs.json", per_run)
    _dump(exp_dir / "summary.json", per_snr)
    logger.info("\nExp C summary (mean ± SD over noise realizations):")
    for s in per_snr:
        logger.info(f"  SNR={s['snr_db']:2d} dB:  "
                    f"BO={s['bo_J_mean']:.5f}±{s['bo_J_std']:.5f}  "
                    f"Expert={s['expert_J_mean']:.5f}±{s['expert_J_std']:.5f}  "
                    f"Δ={s['improvement_pct_mean']:.1f}±{s['improvement_pct_std']:.1f}%")
    logger.info("Exp C done.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment D — starting-model sensitivity: 10-LHS screening (poor init)
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment_D(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    logger.info("═" * 64)
    logger.info(f"EXPERIMENT D  —  Poor-start LHS screening ({D_LHS_BUDGET} evals)")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_D"
    exp_dir.mkdir(exist_ok=True)

    # Good-model baseline = best of Exp A's LHS warm-start (matches the paper's
    # "best warm-start-stage result from the standard starting-model condition").
    bo_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if bo_data is None:
        logger.error("Exp A bo_result.json not found — run Exp A first.")
        return
    n_init     = int(bo_data.get("n_init", BO_CFG.n_init))
    all_J      = np.array(bo_data["all_J"])
    good_lhs_J = float(np.min(all_J[:n_init]))
    logger.info(f"  Good-model LHS-screening baseline (first {n_init} Exp-A evals): "
                f"J={good_lhs_J:.5f}")

    # Poor model: extra 500 m smoothing, then 10 LHS schedules (same bounds/seed).
    if mock:
        poor_obj = make_mock_objective(seed=99)
    else:
        poor_path    = ensure_poor_model(TOY2DAC_CFG)
        poor_cfg     = dataclasses.replace(TOY2DAC_CFG, init_model_file=str(poor_path))
        poor_wrapper = Toy2dacWrapper(poor_cfg, work_root=RESULTS_DIR / "runs" / "expD_poor")
        poor_obj     = make_objective(poor_wrapper, run_prefix="expD_poor")

    thetas = lhs_sample(D_LHS_BUDGET, seed=BO_CFG.lhs_seed)
    poor_Js, poor_rows = [], []
    for i, theta in enumerate(thetas):
        m = poor_obj(theta)
        poor_Js.append(m["J"])
        poor_rows.append({"idx": i, "theta": np.asarray(theta).tolist(),
                          "J": m["J"], "rmse_ms": m.get("rmse_ms"),
                          "ssim": m.get("ssim")})
        logger.info(f"  LHS {i+1:02d}/{D_LHS_BUDGET}  J={m['J']:.5f}  "
                    f"best={min(poor_Js):.5f}")

    poor_Js   = np.array(poor_Js)
    poor_best = float(poor_Js.min())
    best_idx  = int(poor_Js.argmin())
    degradation = (poor_best - good_lhs_J) / good_lhs_J * 100.0

    _dump(exp_dir / "summary.json", {
        "baseline_source":  f"Exp A first {n_init} LHS evals (min)",
        "good_lhs_best_J":  good_lhs_J,
        "poor_best_J":      poor_best,
        "poor_best_theta":  poor_rows[best_idx]["theta"],
        "degradation_pct":  degradation,
        "n_lhs":            D_LHS_BUDGET,
        "poor_all":         poor_rows,
    })

    logger.info("Exp D summary:")
    logger.info(f"  Good (LHS screening): J={good_lhs_J:.5f}")
    logger.info(f"  Poor (LHS screening): J={poor_best:.5f}  "
                f"theta={poor_rows[best_idx]['theta']}")
    logger.info(f"  Degradation from poor init: {degradation:+.1f}%")
    logger.info("Exp D done.")


# ═════════════════════════════════════════════════════════════════════════════
# Optional ablations (NOT part of --exp all) -----------------------------------
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment_E(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """Ablation: acquisition function (EI vs qEI) and Matérn ν (2.5 vs 1.5)."""
    logger.info("═" * 64)
    logger.info("EXPERIMENT E  —  Ablation (acquisition + kernel)   [optional]")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_E"; exp_dir.mkdir(exist_ok=True)
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
        cfg_v = dataclasses.replace(BO_CFG, **overrides,
                                    log_dir=str(exp_dir / f"logs_{label}"))
        bo_result = BayesianOptimizer(obj, cfg_v).run()
        _dump(exp_dir / f"{label}.json", {
            "label": label, "acq_type": var["acq_type"], "nu": var["nu"],
            "best_J": bo_result.best_J,
            "incumbent_trace": bo_result.incumbent_trace.tolist(),
        })
        logger.info(f"  {label}  best J={bo_result.best_J:.5f}")
    logger.info("Exp E done.")


def run_experiment_F(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """Supervised J vs unsupervised Ju correlation over 30 LHS schedules."""
    logger.info("═" * 64)
    logger.info("EXPERIMENT F  —  J vs Ju correlation   [optional]")
    logger.info("═" * 64)
    exp_dir = RESULTS_DIR / "exp_F"; exp_dir.mkdir(exist_ok=True)
    obj = make_mock_objective() if mock else make_objective(wrapper, run_prefix="expF")
    thetas = lhs_sample(30, seed=777)
    J_vals, Ju_vals = [], []
    for i, theta in enumerate(thetas):
        m = obj(theta)
        J_vals.append(m["J"]); Ju_vals.append(m.get("Ju", float("nan")))
        logger.info(f"  F {i+1:02d}/30  J={m['J']:.5f}  Ju={m.get('Ju', float('nan')):.5f}")
    J, Ju = np.array(J_vals), np.array(Ju_vals)
    valid = np.isfinite(J) & np.isfinite(Ju)
    r = float(np.corrcoef(J[valid], Ju[valid])[0, 1]) if valid.sum() > 2 else float("nan")
    _dump(exp_dir / "correlation.json",
          {"J_values": J.tolist(), "Ju_values": Ju.tolist(),
           "pearson_r": r, "n_valid": int(valid.sum())})
    logger.info(f"Exp F done.  Pearson r(J, Ju) = {r:.4f}  (n_valid={valid.sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bo_cfg(log_dir: Path) -> BOConfig:
    return dataclasses.replace(BO_CFG, log_dir=str(log_dir))

def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))

def _clean(d: Dict) -> Dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in d.items() if not isinstance(v, np.ndarray) or v.ndim == 0}

def _build_wrapper() -> Toy2dacWrapper:
    if not sanity_check(TOY2DAC_CFG):
        raise RuntimeError("Sanity check failed — fix TOY2DAC_CFG paths first.")
    return Toy2dacWrapper(TOY2DAC_CFG, work_root=RESULTS_DIR / "runs")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A": run_experiment_A, "B": run_experiment_B,
    "C": run_experiment_C, "D": run_experiment_D,
    "E": run_experiment_E, "F": run_experiment_F,
}
PAPER_EXPERIMENTS = ["A", "B", "C", "D"]   # what --exp all runs (239 FWI runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="BO-FWI manuscript experiments")
    parser.add_argument("--exp", default="A",
                        choices=list(EXPERIMENTS) + ["all"],
                        help="Experiment to run (default: A). 'all' = A,B,C,D.")
    parser.add_argument("--mock", action="store_true",
                        help="Synthetic objective — no toy2dac needed")
    parser.add_argument("--mpi-np", type=int, default=None,
                        help="Override number of MPI ranks")
    args = parser.parse_args()

    if args.mpi_np is not None:
        global TOY2DAC_CFG
        TOY2DAC_CFG = dataclasses.replace(TOY2DAC_CFG, mpi_np=args.mpi_np)
        logger.info(f"MPI ranks overridden: mpi_np={args.mpi_np}")

    wrapper = None
    if not args.mock:
        wrapper = _build_wrapper()
        logger.info(f"toy2dac wrapper ready  (mpirun -n {TOY2DAC_CFG.mpi_np}  "
                    f"grid={TOY2DAC_CFG.nx}×{TOY2DAC_CFG.nz})")

    t0 = time.perf_counter()
    to_run = PAPER_EXPERIMENTS if args.exp == "all" else [args.exp]
    for name in to_run:
        EXPERIMENTS[name](wrapper, mock=args.mock)
    logger.info(f"Total wall time: {(time.perf_counter()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
