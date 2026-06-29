#!/usr/bin/env python3
"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

run_BCD.py
══════════
Optimised runner for Experiments B, C, and D.

Runs from the bo_fwi root:
    cd 
    python3 run_BCD.py --exp B          # sample efficiency
    python3 run_BCD.py --exp C          # noise robustness  (6 runs, ~2 h)
    python3 run_BCD.py --exp D          # start-model sens. (10 runs, ~3 h)
    python3 run_BCD.py --exp all        # all three
    python3 run_BCD.py --exp B --mock   # pipeline test (no toy2dac)
Parallel tip ───────────────────────────────────────────────────────
  Open 3 tmux panes and run each in its own session:
    nohup python3 run_BCD.py --exp B > logs/expB.log 2>&1 &
    nohup python3 run_BCD.py --exp C > logs/expC.log 2>&1 &
    nohup python3 run_BCD.py --exp D > logs/expD.log 2>&1 &
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
sys.path.insert(0, str(Path(__file__).parent))

from fwi.schedule        import make_schedule, expert_schedule, lhs_sample, PARAM_NAMES
from fwi.toy2dac_wrapper import (Toy2dacConfig, Toy2dacWrapper,
                                  sanity_check)
from fwi.metrics         import evaluate_result, smooth_model

from bo.baselines         import (RandomSearch, GridSearch, ExpertRunner,
                                   run_random_search_multi_seed)
from experiments.run_experiments import (
    TOY2DAC_CFG, RESULTS_DIR,
    make_objective, make_mock_objective,
    ensure_poor_model, EXPERT_GROUPS,
    _dump, _clean, _bo_cfg,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BCD")

# ── Experiment-specific settings (trimmed from the full 50-eval plan) ─────────
# Exp B: equal budget = Exp A's 11 evaluations (10 LHS + 1 BO)
BUDGET_B   = 11       # RS and GS get the same number of evaluations as BO had
RS_SEEDS_B = [0, 1, 2]   # 3 seeds  ×  11 evals = 33 RS runs

# Exp C: SNR levels to evaluate
SNR_LEVELS = [10, 20, 40]   # dB

# ── NaN-safe loader ───────────────────────────────────────────────────────────
def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(re.sub(r'\bNaN\b', 'null', path.read_text()))


# ══════════════════════════════════════════════════════════════════════════════
# EXP B — Sample efficiency
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment_B(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """
    Compare BO vs random search vs grid search at equal evaluation budget.

    Key design: BO result is LOADED from Exp A (already run, 11 evaluations).
    Only RS and GS require new FWI calls.

    Equal budget = 11 evaluations (the actual budget BO used in Exp A).
    RS: 3 seeds × 11 evaluations = 33 new FWI runs
    GS: coarse 3×2×2 = 12 grid points
    Total new runs: 45
    """
    logger.info("═" * 64)
    logger.info("EXPERIMENT B  —  Sample efficiency (BO reused from Exp A)")
    logger.info("═" * 64)

    exp_dir = RESULTS_DIR / "exp_B"
    exp_dir.mkdir(exist_ok=True)

    # ── Load BO results from Exp A (no new BO runs needed) ───────────────────
    bo_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if bo_data is None:
        logger.error("Exp A bo_result.json not found. Run Exp A first.")
        return

    bo_trace  = np.array(bo_data["incumbent_trace"])
    bo_best_J = bo_data["best_J"]
    logger.info(f"  BO  (Exp A):  best_J={bo_best_J:.5f}  "
                f"in {len(bo_trace)} evaluations — no new FWI needed")

    # ── Expert (already done in Exp A) ───────────────────────────────────────
    ex_data  = load_json(RESULTS_DIR / "exp_A" / "expert_result.json")
    expert_J = ex_data["J"] if ex_data else None
    logger.info(f"  Expert:       J={expert_J:.5f}  (from Exp A)")

    # ── Objective function ────────────────────────────────────────────────────
    obj = make_mock_objective() if mock else make_objective(
        wrapper, run_prefix="expB"
    )

    # ── Random search  (3 seeds × 11 evaluations = 33 new FWI runs) ──────────
    logger.info(f"  Starting Random Search: {len(RS_SEEDS_B)} seeds × {BUDGET_B} evals")
    rs_agg = run_random_search_multi_seed(obj, n_eval=BUDGET_B, seeds=RS_SEEDS_B)
    logger.info(f"  RS done.  mean_best_J={rs_agg['best_J_all_seeds'].mean():.5f}  "
                f"±{rs_agg['best_J_all_seeds'].std():.5f}")

    # ── Grid search  (~12 grid points) ───────────────────────────────────────
    # Coarse grid on [f_min × f_max × K] with gamma and beta fixed
    # (from Exp A we know f_min/f_max dominate: 99.9% of importance)
    logger.info("  Starting Grid Search (12 points, fixed gamma=1.5 beta=0.1)")
    gs = GridSearch(
        obj,
        f_min_values = [3.0, 5.0, 7.0],    # 3 f_min values
        f_max_values = [13.0, 18.0],         # 2 f_max values
        K_values     = [3, 5],               # 2 K values
        gamma_fixed  = 1.5,
        beta_fixed   = 0.10,
    )
    gs_result = gs.run()
    logger.info(f"  GS done.  best_J={gs_result.best_J:.5f}  n_evals={gs_result.n_eval}")

    # ── Save all results ──────────────────────────────────────────────────────
    _dump(exp_dir / "summary.json", {
        "budget":           BUDGET_B,
        "bo_trace":         bo_trace.tolist(),
        "bo_best_J":        bo_best_J,
        "rs_mean":          rs_agg["incumbent_mean"].tolist(),
        "rs_std":           rs_agg["incumbent_std"].tolist(),
        "rs_best_per_seed": rs_agg["best_J_all_seeds"].tolist(),
        "rs_seeds":         RS_SEEDS_B,
        "gs_trace":         gs_result.incumbent_trace.tolist(),
        "gs_best_J":        gs_result.best_J,
        "expert_J":         expert_J,
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("Exp B summary:")
    logger.info(f"  BO      J = {bo_best_J:.5f}  (best in {len(bo_trace)} evals)")
    logger.info(f"  RS mean J = {rs_agg['best_J_all_seeds'].mean():.5f} ± "
                f"{rs_agg['best_J_all_seeds'].std():.5f}  ({len(RS_SEEDS_B)} seeds)")
    logger.info(f"  GS      J = {gs_result.best_J:.5f}  ({gs_result.n_eval} grid pts)")
    logger.info(f"  Expert  J = {expert_J:.5f}")
    logger.info(f"  BO vs RS mean improvement: "
                f"{(rs_agg['best_J_all_seeds'].mean() - bo_best_J) / rs_agg['best_J_all_seeds'].mean() * 100:.1f}%")
    logger.info("Exp B done.")


# ══════════════════════════════════════════════════════════════════════════════
# EXP C — Noise robustness (lite: 6 FWI runs)
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment_C(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """
    Noise robustness — lite version.

    We evaluate the FIXED best schedule from Exp A and the expert schedule
    at each of 3 SNR levels.  This requires only 6 new FWI calls:
        3 SNR × (BO best schedule + expert schedule) = 6 runs

    This answers: "Does the noise-free optimal schedule transfer to noisy data?"
    A positive result (BO still outperforms expert under noise) strengthens
    the robustness claim without requiring full BO re-optimisation per SNR level.
    """
    logger.info("═" * 64)
    logger.info(f"EXPERIMENT C  —  Noise robustness  SNR={SNR_LEVELS} dB  (6 runs)")
    logger.info("═" * 64)

    # ── Load best theta from Exp A ────────────────────────────────────────────
    bo_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if bo_data is None:
        logger.error("Exp A bo_result.json not found. Run Exp A first.")
        return

    best_theta  = np.array(bo_data["best_theta"])
    best_sched  = make_schedule(best_theta)
    expert_thta = np.array([3.0, 20.0, 3.0, 1.0, 0.0])

    logger.info(f"  BO best theta: {[round(x,2) for x in best_theta.tolist()]}")
    logger.info(f"  Expert theta:  [3.0, 20.0, 3.0, 1.0, 0.0]")

    exp_dir = RESULTS_DIR / "exp_C"
    exp_dir.mkdir(exist_ok=True)
    all_results = []

    for snr in SNR_LEVELS:
        logger.info(f"\n  ── SNR = {snr} dB ──")
        snr_dir = exp_dir / f"snr_{snr}dB"
        snr_dir.mkdir(exist_ok=True)

        # Build objective with noise injection
        if mock:
            import time as _t
            rng = np.random.default_rng(snr)
            def obj_mock(theta):
                _t.sleep(0.005)
                n = rng.normal(0, 0.005)
                # Simulate noise degrading J: higher SNR → closer to noiseless
                noise_penalty = 0.02 * (40 - snr) / 30   # 0 at 40dB, 0.02 at 10dB
                from fwi.schedule import normalize as _n
                from fwi.schedule import BOUNDS_LOWER, BOUNDS_UPPER
                t   = _n(theta)
                ts  = _n(best_theta)
                d   = np.sum((t - ts) ** 2)
                J   = float(np.clip(.05 + .45 * np.exp(-d/.2) + noise_penalty + n, .02, 1.0))
                return {"J": J, "rmse_ms": J*1000, "r2": 1-J, "ssim": 1-J}
            obj = obj_mock
        else:
            obj = make_objective(
                wrapper,
                run_prefix  = f"expC_snr{snr}",
                snr_db      = float(snr),
            )

        # Evaluate BO best schedule
        logger.info(f"    Evaluating BO best schedule ...")
        m_bo = obj(best_theta)
        logger.info(f"    BO J={m_bo['J']:.5f}  RMSE={m_bo.get('rmse_ms',0):.1f} m/s")

        # Evaluate expert schedule
        logger.info(f"    Evaluating expert schedule ...")
        m_ex = obj(expert_thta)
        logger.info(f"    Expert J={m_ex['J']:.5f}  RMSE={m_ex.get('rmse_ms',0):.1f} m/s")

        row = {
            "snr_db":         snr,
            "bo_J":           m_bo["J"],
            "expert_J":       m_ex["J"],
            "bo_rmse_ms":     m_bo.get("rmse_ms"),
            "expert_rmse_ms": m_ex.get("rmse_ms"),
            "bo_ssim":        m_bo.get("ssim"),
            "expert_ssim":    m_ex.get("ssim"),
            "bo_improvement_pct": (m_ex["J"] - m_bo["J"]) / m_ex["J"] * 100,
        }
        all_results.append(row)
        _dump(snr_dir / "results.json", row)

        logger.info(
            f"    BO improvement at {snr} dB: "
            f"{row['bo_improvement_pct']:.1f}%  "
            f"({m_ex['J']:.5f} → {m_bo['J']:.5f})"
        )

    _dump(exp_dir / "all_results.json", all_results)
    logger.info("\nExp C summary:")
    for r in all_results:
        logger.info(f"  SNR={r['snr_db']:2d} dB:  BO={r['bo_J']:.5f}  "
                    f"Expert={r['expert_J']:.5f}  "
                    f"Δ={r['bo_improvement_pct']:+.1f}%")
    logger.info("Exp C done.")


# ══════════════════════════════════════════════════════════════════════════════
# EXP D — Starting-model sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment_D(wrapper: Optional[Toy2dacWrapper], mock: bool = False) -> None:
    """
    Starting-model sensitivity.

    Good model:  vp_Marmousi_init  (standard, from Exp A — no new runs needed)
    Poor model:  extra 500 m Gaussian smoothing applied on top
                 → 10 new LHS evaluations with the poor init model

    The comparison is: does the BO framework still find a schedule that beats
    the expert when starting from an inferior velocity model?
    """
    logger.info("═" * 64)
    logger.info("EXPERIMENT D  —  Starting-model sensitivity  (10 new runs)")
    logger.info("═" * 64)

    exp_dir = RESULTS_DIR / "exp_D"
    exp_dir.mkdir(exist_ok=True)

    # ── Good model result: load from Exp A ────────────────────────────────────
    good_data = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
    if good_data:
        good_J     = good_data["best_J"]
        good_trace = good_data["incumbent_trace"]
        logger.info(f"  Good model (Exp A):  best_J={good_J:.5f}  "
                    f"({len(good_trace)} evals)")
    else:
        good_J, good_trace = None, []
        logger.warning("  Exp A result not found — good-model data will be missing")

    # ── Poor model: create + run LHS ──────────────────────────────────────────
    if not mock:
        poor_path = ensure_poor_model(TOY2DAC_CFG)
        poor_cfg  = dataclasses.replace(
            TOY2DAC_CFG, init_model_file=str(poor_path)
        )
        poor_wrapper = Toy2dacWrapper(
            poor_cfg, work_root=RESULTS_DIR / "runs" / "expD_poor"
        )
        obj = make_objective(poor_wrapper, run_prefix="expD_poor")
    else:
        rng = np.random.default_rng(99)
        def obj(theta):
            import time as _t; _t.sleep(0.005)
            from fwi.schedule import normalize as _n
            t = _n(theta)
            # Simulated poor model: 10-15% worse than good model
            J = float(np.clip(.12 + .5*np.exp(-np.sum((t-.5)**2)/.2)
                              + rng.normal(0,.01), .05, 1.0))
            return {"J": J, "rmse_ms": J*1000, "r2": 1-J, "ssim": 1-J}

    # LHS only (no BO — we just want to know if the landscape shifts
    # compared to the good starting model; n_eval = n_init from BOConfig)
    n_lhs = 10
    logger.info(f"  Running {n_lhs} LHS evaluations with poor starting model ...")

    from bo.baselines import RandomSearch
    rs = RandomSearch(obj, n_eval=n_lhs, seed=42)
    rs_result = rs.run()

    poor_J     = rs_result.best_J
    poor_trace = rs_result.incumbent_trace.tolist()

    logger.info(f"  Poor model LHS: best_J={poor_J:.5f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    (exp_dir / "good").mkdir(exist_ok=True)
    (exp_dir / "poor").mkdir(exist_ok=True)

    if good_J is not None:
        _dump(exp_dir / "good" / "result.json", {
            "model_quality":   "good",
            "best_J":          good_J,
            "best_theta":      good_data.get("best_theta", []),
            "incumbent_trace": good_trace,
            "source":          "Exp A",
        })

    _dump(exp_dir / "poor" / "result.json", {
        "model_quality":   "poor",
        "best_J":          poor_J,
        "best_theta":      rs_result.best_theta.tolist(),
        "incumbent_trace": poor_trace,
        "n_eval":          n_lhs,
    })

    logger.info("Exp D summary:")
    if good_J:
        logger.info(f"  Good model:  best_J={good_J:.5f}")
    logger.info(f"  Poor model:  best_J={poor_J:.5f}")
    if good_J:
        delta = (poor_J - good_J) / good_J * 100
        logger.info(f"  Degradation from poor init: {delta:+.1f}%")
    logger.info("Exp D done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = {"B": run_experiment_B, "C": run_experiment_C, "D": run_experiment_D}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiments B, C, D for the BO-FWI manuscript."
    )
    parser.add_argument(
        "--exp", required=True,
        choices=list(EXPERIMENTS) + ["all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Synthetic objective — no toy2dac needed (pipeline test)",
    )
    parser.add_argument(
        "--mpi-np", type=int, default=None,
        help="Override MPI rank count",
    )
    args = parser.parse_args()

    global TOY2DAC_CFG
    if args.mpi_np:
        from experiments.run_experiments import TOY2DAC_CFG as _base
        import experiments.run_experiments as _re
        _re.TOY2DAC_CFG = dataclasses.replace(_base, mpi_np=args.mpi_np)
        TOY2DAC_CFG     = _re.TOY2DAC_CFG

    wrapper = None
    if not args.mock:
        ok = sanity_check(TOY2DAC_CFG)
        if not ok:
            logger.error("Sanity check failed. Fix TOY2DAC_CFG paths first.")
            sys.exit(1)
        wrapper = Toy2dacWrapper(TOY2DAC_CFG, work_root=RESULTS_DIR / "runs")

    t0 = time.perf_counter()

    if args.exp == "all":
        for name, fn in EXPERIMENTS.items():
            fn(wrapper, mock=args.mock)
    else:
        EXPERIMENTS[args.exp](wrapper, mock=args.mock)

    logger.info(f"\nTotal wall time: {(time.perf_counter()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
