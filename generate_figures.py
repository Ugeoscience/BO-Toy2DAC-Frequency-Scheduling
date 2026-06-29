#!/usr/bin/env python3
"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

generate_figures.py



Figures produced
────────────────
  fig01_workflow.{pdf,png}           Method schematic (BO outer + toy2dac inner)
  fig02_marmousi.{pdf,png}           True / init / poor starting model panels
  fig03_schedule_params.{pdf,png}    Effect of γ on frequency-group layout
  fig04_gp_surrogate.{pdf,png}       GP posterior mean & σ over iterations
  fig05_ei_surface.{pdf,png}         EI acquisition surface + next candidate
  fig06_convergence.{pdf,png}        Incumbent J vs evaluations  ← headline figure
  fig07_schedule_compare.{pdf,png}   BO-optimal vs expert frequency schedule
  fig08_velocity_models.{pdf,png}    True / Init / Expert / BO model + diff maps
  fig09_profiles.{pdf,png}           Vertical velocity profiles at 3 locations
  fig10_noise.{pdf,png}              J vs SNR (noise robustness, Exp C)
  fig11_startmodel.{pdf,png}         Good vs poor starting model sensitivity (Exp D)
  fig12_importance.{pdf,png}         GP-derived parameter importance (Exp A)

Usage
─────
  python3 generate_figures.py                     # all figures
  python3 generate_figures.py --figs 6 7 8 9      # specific figures
  python3 generate_figures.py --figs 1 2 3        # schematics only (no run data)
  python3 generate_figures.py --list              # list status of all figures
  python3 generate_figures.py --outdir /my/dir    # custom output directory
"""

from __future__ import annotations

import argparse
import json
import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("figures")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit paths here if needed
# ──────────────────────────────────────────────────────────────────────────────

TEMPLATE_DIR = Path("")
BO_FWI_ROOT  = Path(__file__).parent          
RESULTS_DIR  = BO_FWI_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
RUNS_DIR     = RESULTS_DIR / "runs"

NX, NZ, DX, DZ = 681, 141, 25.0, 25.0   # Marmousi @ 25 m grid

# ── Figure dimensions ─────────────────
W1 = 3.35     # single-column width  (8.5 cm)
W2 = 6.89     # double-column width  (17.5 cm)

# ── Consistent colour palette ─────────────────────────────────────────────────
C = dict(
    bo       = "#2166ac",   # blue   — BO
    rs       = "#d6604d",   # red    — random search
    gs       = "#4dac26",   # green  — grid search
    expert   = "#8073ac",   # purple — expert schedule
    good     = "#2166ac",   # alias  — good start model
    poor     = "#d6604d",   # alias  — poor start model
    ref      = "#1a1a1a",   # near-black — true model line
    init     = "#888888",   # gray   — initial model
    g0       = "#7F77DD",   # group 0 colour
    g1       = "#1D9E75",   # group 1
    g2       = "#EF9F27",   # group 2
    g3       = "#D85A30",   # group 3
)
CMAP_VEL  = "jet"           # velocity model colourmap (industry standard)
CMAP_DIFF = "RdBu_r"        # difference maps

# ── Matplotlib global style ───────────────────────────────────────────────────
RC = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
    "patch.linewidth":   0.7,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
}
plt.rcParams.update(RC)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_binary_model(path: str | Path,
                      nx: int = NX, nz: int = NZ) -> Optional[np.ndarray]:
    """Load a raw float32 Fortran-order binary model file → (nz, nx) array."""
    p = Path(path)
    if not p.exists():
        log.warning(f"Model file not found: {p}")
        return None
    raw = np.frombuffer(p.read_bytes(), dtype=np.float32)
    if raw.size != nx * nz:
        log.warning(f"{p.name}: expected {nx*nz} floats, got {raw.size}")
        return None
    return raw.reshape((nz, nx), order="F")


def load_json(path: str | Path) -> Optional[Dict]:
    """JSON loader tolerant of bare NaN (Python/Fortran) -> converts to null."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(re.sub(r'\bNaN\b', 'null', p.read_text()))


def load_jsonl(path: str | Path) -> List[Dict]:
    """Load a .jsonl file line-by-line, NaN-safe."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(re.sub(r'\bNaN\b', 'null', line)))
            except json.JSONDecodeError:
                pass
    return rows


def find_jsonl() -> Path:
    """Return the bo_iterations.jsonl path, searching common locations."""
    candidates = [
        RESULTS_DIR / "exp_A" / "logs_bo" / "bo_iterations.jsonl",
        RESULTS_DIR / "logs" / "bo_iterations.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return primary even if missing (will warn)


def find_fwi_model(run_id: str) -> Optional[np.ndarray]:
    """
    Find and load the final velocity model for a given run_id.
    Checks (in order):
      1. runs/{run_id}/vp_final            (saved by Toy2dacWrapper.run)
      2. runs/{run_id}/group_NN/param_vp_final  (last group output)
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        log.warning(f"Run directory not found: {run_dir}")
        return None
    # Preferred: top-level vp_final
    top = run_dir / "vp_final"
    if top.exists():
        return load_binary_model(top)
    # Fallback: last group's param_vp_final
    grp_dirs = sorted(run_dir.glob("group_*"), reverse=True)
    for gd in grp_dirs:
        cand = gd / "param_vp_final"
        if cand.exists():
            return load_binary_model(cand)
    log.warning(f"No final model found in {run_dir}")
    return None


def save_fig(fig: plt.Figure, name: str, outdir: Path = FIGURES_DIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
    log.info(f"  ✓  {name}.pdf  +  .png")
    plt.close(fig)


def model_extent_km() -> List[float]:
    """[x_min, x_max, z_max, z_min] in km for imshow extent."""
    return [0, NX * DX / 1_000, NZ * DZ / 1_000, 0]


def vel_limits(m_true: np.ndarray) -> Tuple[float, float]:
    vmin = float(m_true.min()) / 1_000
    vmax = float(m_true.max()) / 1_000
    return vmin, vmax


# ──────────────────────────────────────────────────────────────────────────────
# DATA CONTEXT  — loads all results once, shared across figure functions
# ──────────────────────────────────────────────────────────────────────────────

class FigCtx:
    """Lazy-loaded data container."""

    def __init__(self):
        self._true   = None
        self._init   = None
        self._poor   = None
        self._expA   = None
        self._expB   = None
        self._expC   = None
        self._expD_g = None
        self._expD_p = None

    # ── Model files ───────────────────────────────────────────────────────────
    @property
    def m_true(self):
        if self._true is None:
            self._true = load_binary_model(TEMPLATE_DIR / "vp_Marmousi_exact")
        return self._true

    @property
    def m_init(self):
        if self._init is None:
            self._init = load_binary_model(TEMPLATE_DIR / "vp_Marmousi_init")
        return self._init

    @property
    def m_poor(self):
        if self._poor is None:
            self._poor = load_binary_model(RESULTS_DIR / "vp_Marmousi_init_poor")
        return self._poor

    def m_expert(self):
        d = load_json(RESULTS_DIR / "exp_A" / "expert_result.json")
        if d and "metrics" in d and "run_id" in d["metrics"]:
            return find_fwi_model(d["metrics"]["run_id"])
        return None

    def m_bo_best(self):
        d = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
        if d and "best_metrics" in d and "run_id" in d["best_metrics"]:
            return find_fwi_model(d["best_metrics"]["run_id"])
        return None

    # ── Experiment JSON results ───────────────────────────────────────────────
    @property
    def expA_bo(self):
        if self._expA is None:
            self._expA = load_json(RESULTS_DIR / "exp_A" / "bo_result.json")
        return self._expA

    @property
    def expA_expert(self):
        return load_json(RESULTS_DIR / "exp_A" / "expert_result.json")

    @property
    def expB(self):
        if self._expB is None:
            self._expB = load_json(RESULTS_DIR / "exp_B" / "summary.json")
        return self._expB

    @property
    def expC(self):
        if self._expC is None:
            rows = []
            for snr_dir in sorted((RESULTS_DIR / "exp_C").glob("snr_*")):
                r = load_json(snr_dir / "results.json")
                if r:
                    rows.append(r)
            self._expC = rows or None
        return self._expC

    @property
    def expD_good(self):
        if self._expD_g is None:
            self._expD_g = load_json(RESULTS_DIR / "exp_D" / "good" / "result.json")
        return self._expD_g

    @property
    def expD_poor(self):
        if self._expD_p is None:
            self._expD_p = load_json(RESULTS_DIR / "exp_D" / "poor" / "result.json")
        return self._expD_p


CTX = FigCtx()


# ──────────────────────────────────────────────────────────────────────────────
# FIG 1 — WORKFLOW SCHEMATIC
# ──────────────────────────────────────────────────────────────────────────────

def fig01_workflow():
    """BO outer loop wrapping toy2dac multi-scale FWI inner loop."""
    fig, ax = plt.subplots(figsize=(W2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, label, sublabel="", fc="#EEEDFE", ec="#534AB7", fs=8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               fc=fc, ec=ec, lw=0.8, zorder=2)
        ax.add_patch(rect)
        cy = y + h / 2 + (0.12 if sublabel else 0)
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=ec, zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                    fontsize=6.5, color="#666", zorder=3, style="italic")

    def arr(x1, y1, x2, y2, col="#534AB7"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9))

    # ── BO outer loop box ─────────────────────────────────────────────────────
    bo_rect = FancyBboxPatch((0.2, 0.3), 9.6, 5.5, boxstyle="round,pad=0.12",
                              fc="#F8F8FF", ec="#534AB7", lw=1.2, ls="--", zorder=0)
    ax.add_patch(bo_rect)
    ax.text(0.55, 5.6, "Bayesian optimisation outer loop", fontsize=7.5,
            color="#534AB7", va="center", fontweight="bold")

    # Warm-start block
    box(0.4, 3.8, 2.2, 1.0,   "LHS warm-start",  f"n₀ = 10 schedules", "#E1F5EE","#0F6E56")

    # GP block
    box(0.4, 1.8, 2.2, 1.2,   "GP surrogate",    "Matérn-5/2 kernel", "#EEEDFE","#534AB7")

    # EI block
    box(0.4, 0.5, 2.2, 1.0,   "EI acquisition",  "next θ candidate",   "#FAEEDA","#854F0B")

    # Arrows: LHS → GP → EI → loop back
    arr(1.5, 3.8, 1.5, 3.0)
    arr(1.5, 1.8, 1.5, 1.5)
    ax.annotate("", xy=(2.62, 0.95), xytext=(2.62, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#534AB7", lw=0.9))

    # θ proposal label
    ax.text(3.05, 2.9, r"$\boldsymbol{\theta}$", fontsize=9, color="#534AB7",
            ha="center", va="center")
    arr(2.62, 0.95, 3.5, 2.2)

    # ── toy2dac inner loop (freq-continuation) ────────────────────────────────
    inner = FancyBboxPatch((3.5, 0.4), 5.2, 5.2, boxstyle="round,pad=0.1",
                            fc="#FFF8F5", ec="#D85A30", lw=0.9, ls="--", zorder=1)
    ax.add_patch(inner)
    ax.text(3.75, 5.45, "toy2dac multi-scale FWI inner loop", fontsize=7.5,
            color="#D85A30", fontweight="bold")

    # Frequency groups
    grp_cols = ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30"]
    grp_labels = ["Group 0\nlow Hz", "Group 1\nmid Hz", "Group 2\nhigh Hz", "···"]
    xs = [3.7, 4.75, 5.8, 6.85]
    for i, (xg, lbl, col) in enumerate(zip(xs, grp_labels, grp_cols)):
        box(xg, 3.2, 0.9, 1.5, lbl, "", fc=col+"22", ec=col, fs=7)
        if i < 3:
            box(xg, 1.5, 0.9, 1.4, "MODEL\n+\nINVERT", "", fc="#FFF",
                ec=col, fs=6.5)

    # Arrows between groups
    for x_from in [4.6, 5.65, 6.7]:
        arr(x_from, 3.95, x_from + 0.15, 3.95, "#666")
        arr(x_from, 2.20, x_from + 0.15, 2.20, "#666")

    # Band → model arrows
    for xg in xs[:3]:
        arr(xg + 0.45, 3.2, xg + 0.45, 2.9, "#555")

    # J metric
    box(7.9, 1.9, 1.7, 0.85, "J = RMSE\n(m_est, m_true)", "", "#FCEBEB","#A32D2D", 7)
    arr(7.7, 2.2, 7.9, 2.2, "#A32D2D")
    # J → GP update
    ax.annotate("", xy=(2.62, 1.8), xytext=(2.62, 0.6),
                arrowprops=dict(arrowstyle="-|>", color="#A32D2D",
                                connectionstyle="arc3,rad=0", lw=0.9))
    ax.plot([7.9 + 0.85, 8.75, 8.75, 2.62], [2.32, 2.32, 0.6, 0.6],
            color="#A32D2D", lw=0.9, ls="-")
    ax.text(5.6, 0.25, "J  →  update GP  →  next θ", fontsize=7,
            color="#A32D2D", ha="center")

    # Label
    ax.text(5.0, 5.75, "Fig. 1 — BO-FWI method overview", fontsize=8,
            ha="center", fontweight="bold", color="#333")

    save_fig(fig, "fig01_workflow")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 2 — MARMOUSI MODEL PANELS
# ──────────────────────────────────────────────────────────────────────────────

def fig02_marmousi():
    m_true = CTX.m_true
    m_init = CTX.m_init
    if m_true is None:
        log.warning("Fig 2 skipped: vp_Marmousi_exact not found"); return

    has_poor = CTX.m_poor is not None
    n = 3 if has_poor else 2
    fig, axes = plt.subplots(1, n, figsize=(W2, 2.8), sharey=True)
    if n == 2:
        axes = list(axes)

    ext = model_extent_km()
    vmin, vmax = vel_limits(m_true)

    panels = [(m_true, "(a) True Marmousi model")]
    if m_init is not None:
        panels.append((m_init, "(b) Standard starting model"))
    if has_poor:
        panels.append((CTX.m_poor, "(c) Poor starting model (Exp D)"))

    for ax, (m, title) in zip(axes, panels):
        im = ax.imshow(m / 1_000, extent=ext, aspect="auto",
                       cmap=CMAP_VEL, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("Distance (km)", fontsize=8)
    axes[0].set_ylabel("Depth (km)", fontsize=8)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("P-wave velocity (km/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Marmousi benchmark model", fontsize=9, y=1.01)
    save_fig(fig, "fig02_marmousi")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 3 — SCHEDULE PARAMETERISATION (gamma effect)
# ──────────────────────────────────────────────────────────────────────────────

def fig03_schedule_params():
    from fwi.schedule import make_schedule

    gammas  = [0.5, 1.0, 1.5, 2.5]
    colors  = [C["g0"], C["g1"], C["g2"], C["g3"]]
    theta_base = [3.0, 20.0, 5.0, None, 0.1]   # gamma filled in below

    fig, axes = plt.subplots(len(gammas), 1, figsize=(W1 + 0.5, 4.0), sharex=True)

    for ax, g, col in zip(axes, gammas, colors):
        theta = theta_base[:3] + [g] + theta_base[4:]
        sched = make_schedule(theta, f_sample=0.5)
        for grp in sched.groups:
            ax.barh(0, grp.f_hi - grp.f_lo, left=grp.f_lo,
                    height=0.55, color=col, alpha=0.78,
                    edgecolor="white", linewidth=0.6)
            ax.plot(grp.f_center, 0, "|", color="white",
                    markersize=6, markeredgewidth=1.2)
        ax.set_ylabel(f"γ = {g}", rotation=0, labelpad=40,
                      va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(1.5, 21.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", lw=0.4)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=8)
    fig.suptitle(
        f"Effect of spacing exponent γ on frequency-group layout\n"
        f"(f_min=3 Hz, f_max=20 Hz, K=5, β=0.10)",
        fontsize=8, y=1.0,
    )
    fig.tight_layout()
    save_fig(fig, "fig03_schedule_params")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 4 — GP SURROGATE EVOLUTION (optional — requires BoTorch)
# ──────────────────────────────────────────────────────────────────────────────

def fig04_gp_surrogate():
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError:
        log.warning("Fig 4 skipped: BoTorch not available"); return

    log_path = RESULTS_DIR / "logs" / "bo_iterations.jsonl"
    if not log_path.exists():
        log.warning("Fig 4 skipped: bo_iterations.jsonl not found"); return

    from fwi.schedule import DIM, PARAM_NAMES, normalize, denormalize

    records = load_jsonl(log_path)
    if len(records) < 5:
        log.warning("Fig 4 skipped: too few BO iterations logged"); return

    theta_all = np.array([r["theta"] for r in records])
    J_all     = np.array([r["J"]     for r in records])
    X_all     = torch.tensor(np.apply_along_axis(normalize, 1, theta_all))
    Y_all     = torch.tensor(J_all).unsqueeze(-1)

    # Snapshots at 10%, 30%, 60%, 100% of budget
    n_total = len(records)
    snaps   = [max(5, int(n_total * f)) for f in (0.1, 0.3, 0.6, 1.0)]
    snaps   = sorted(set(snaps))

    # Show 2D slice: f_min (idx=0) vs gamma (idx=3)
    pi, pj = 0, 3
    xi_n = np.linspace(0, 1, 40)
    xj_n = np.linspace(0, 1, 40)
    XI, XJ = np.meshgrid(xi_n, xj_n)
    fixed  = np.full(DIM, 0.5)

    fig, axes = plt.subplots(1, len(snaps), figsize=(W2, 2.6), sharey=True)

    for ax, n in zip(axes if len(snaps) > 1 else [axes], snaps):
        Xn = X_all[:n]; Yn = Y_all[:n]
        gp = SingleTaskGP(Xn, Yn,
                          covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=DIM)),
                          input_transform=Normalize(d=DIM),
                          outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll); gp.eval()

        n_pts   = 40 * 40
        X_flat  = np.tile(fixed, (n_pts, 1))
        X_flat[:, pi] = XI.ravel()
        X_flat[:, pj] = XJ.ravel()
        with torch.no_grad():
            post = gp.posterior(torch.tensor(X_flat))
            mu   = post.mean.numpy().reshape(40, 40)

        xi_nat = denormalize(
            np.column_stack([xi_n] + [np.zeros_like(xi_n)] * (DIM-1)))[:, pi]
        xj_nat = denormalize(
            np.column_stack([np.zeros_like(xj_n), np.zeros_like(xj_n),
                             np.zeros_like(xj_n), xj_n,
                             np.zeros_like(xj_n)]))[:, pj]

        ctr = ax.contourf(xi_nat, xj_nat, mu, levels=15, cmap="viridis_r", alpha=0.85)
        # Scatter observed points
        xi_obs = denormalize(Xn.numpy())[:, pi]
        xj_obs = denormalize(Xn.numpy())[:, pj]
        ax.scatter(xi_obs, xj_obs, s=8, c="white", edgecolors="#333",
                   linewidths=0.4, zorder=5)
        ax.set_title(f"n = {n}", fontsize=8)
        ax.set_xlabel(PARAM_NAMES[pi], fontsize=8)

    axes[0].set_ylabel(PARAM_NAMES[pj], fontsize=8)
    cbar = fig.colorbar(ctr, ax=axes if len(snaps) > 1 else axes[0],
                        fraction=0.03, pad=0.01)
    cbar.set_label("GP posterior mean J", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("GP surrogate evolution (2-D slice: f_min vs γ)", fontsize=9, y=1.01)
    save_fig(fig, "fig04_gp_surrogate")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 5 — EI ACQUISITION SURFACE
# ──────────────────────────────────────────────────────────────────────────────

def fig05_ei_surface():
    log_path = find_jsonl()
    if not log_path.exists():
        log.warning("Fig 5 skipped: no iteration log"); return
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError:
        log.warning("Fig 5 skipped: BoTorch not available"); return

    from fwi.schedule import DIM, PARAM_NAMES, normalize, denormalize

    records = load_jsonl(log_path)
    if len(records) < 10:
        log.warning("Fig 5 skipped: not enough data"); return

    theta_arr = np.array([r["theta"] for r in records])
    J_arr     = np.array([r["J"]     for r in records])
    X = torch.tensor(np.apply_along_axis(normalize, 1, theta_arr))
    Y = torch.tensor(J_arr).unsqueeze(-1)

    gp = SingleTaskGP(X, Y,
                      covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=DIM)),
                      input_transform=Normalize(d=DIM),
                      outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp)); gp.eval()

    best_f = Y.min().item()
    EI = LogExpectedImprovement(model=gp, best_f=best_f, maximize=False)

    pi, pj = 0, 3  # f_min vs gamma
    xi_n = np.linspace(0, 1, 50)
    xj_n = np.linspace(0, 1, 50)
    XI, XJ = np.meshgrid(xi_n, xj_n)
    fixed   = np.full(DIM, 0.5)
    n_pts   = 50 * 50
    X_flat  = np.tile(fixed, (n_pts, 1))
    X_flat[:, pi] = XI.ravel()
    X_flat[:, pj] = XJ.ravel()
    with torch.no_grad():
        log_ei = EI(torch.tensor(X_flat).unsqueeze(1))
        ei_map = torch.exp(log_ei).numpy().reshape(50, 50)

    xi_nat = denormalize(np.column_stack([xi_n] + [np.zeros_like(xi_n)] * (DIM-1)))[:, pi]
    xj_nat = denormalize(np.column_stack([np.zeros_like(xj_n), np.zeros_like(xj_n),
                         np.zeros_like(xj_n), xj_n, np.zeros_like(xj_n)]))[:, pj]
    best_idx = np.unravel_index(np.argmax(ei_map), ei_map.shape)

    fig, ax = plt.subplots(figsize=(W1 + 0.4, 3.0))
    ct = ax.contourf(xi_nat, xj_nat, ei_map, levels=20, cmap="YlOrRd")
    ax.contour(xi_nat, xj_nat, ei_map, levels=8, colors="white",
               linewidths=0.3, alpha=0.5)
    ax.plot(xi_nat[best_idx[1]], xj_nat[best_idx[0]], "*", color="#2166ac",
            markersize=10, label="Next candidate", zorder=6)
    xi_obs = denormalize(X.numpy())[:, pi]
    xj_obs = denormalize(X.numpy())[:, pj]
    ax.scatter(xi_obs, xj_obs, s=7, c="white", edgecolors="#333",
               linewidths=0.4, zorder=5, label="Previous evals")
    ax.set_xlabel(f"$f_{{\\min}}$ (Hz)", fontsize=8)
    ax.set_ylabel("$\\gamma$", fontsize=8)
    ax.set_title("Expected Improvement surface\n(2-D slice: $f_{\\min}$ vs $\\gamma$)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.colorbar(ct, ax=ax, fraction=0.04, pad=0.02).set_label("EI", fontsize=8)
    save_fig(fig, "fig05_ei_surface")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 6 — CONVERGENCE  ← headline figure
# ──────────────────────────────────────────────────────────────────────────────

def fig06_convergence():
    d = CTX.expB
    d_a = CTX.expA_bo
    if d is None and d_a is None:
        log.warning("Fig 6 skipped: no convergence data"); return
    if d is None:
        d = {"bo_trace": d_a.get("incumbent_trace", []),
             "rs_mean": [], "rs_std": [], "gs_trace": []}
        log.info("Fig 6: using Exp A trace only (Exp B not yet run)")

    bo_trace  = np.array(d.get("bo_trace",  []))
    rs_mean   = np.array(d.get("rs_mean",   []))
    rs_std    = np.array(d.get("rs_std",    []))
    gs_trace  = np.array(d.get("gs_trace",  []))
    _ex_d     = CTX.expA_expert or {}
    expert_J  = _ex_d.get("J", None)   # top-level "J" in expert_result.json

    n_init = 10   # must match BOConfig.n_init

    fig, ax = plt.subplots(figsize=(W2 * 0.72, 3.2))

    if len(bo_trace):
        ev = np.arange(1, len(bo_trace) + 1)
        ax.plot(ev, bo_trace, color=C["bo"], lw=1.8, label="BO (EI + GP)", zorder=5)
        ax.axvline(n_init + 0.5, color="grey", lw=0.8, ls="--", alpha=0.55)
        ax.text(n_init + 0.9, ax.get_ylim()[1] if ax.get_ylim()[1] < 1 else 0.95,
                "BO starts", fontsize=6.5, color="grey", va="top")

    if len(rs_mean):
        ev_rs = np.arange(1, len(rs_mean) + 1)
        ax.plot(ev_rs, rs_mean, color=C["rs"], lw=1.4, label="Random search (mean)")
        if len(rs_std):
            ax.fill_between(ev_rs, rs_mean - rs_std, rs_mean + rs_std,
                            color=C["rs"], alpha=0.15, label="Random (±1σ)")

    if len(gs_trace):
        ev_gs = np.arange(1, len(gs_trace) + 1)
        ax.step(ev_gs, gs_trace, where="post",
                color=C["gs"], lw=1.4, ls="-.", label="Grid search")

    if expert_J is not None:
        ax.axhline(expert_J, color=C["expert"], lw=1.2, ls=":",
                   label=f"Expert schedule  (J = {expert_J:.4f})")

    ax.set_xlabel("Number of FWI evaluations", fontsize=9)
    ax.set_ylabel("Incumbent normalised RMSE  $J(\\boldsymbol{\\theta})$", fontsize=9)
    ax.set_title("Sample efficiency: BO vs baselines", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, "fig06_convergence")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 7 — FREQUENCY SCHEDULE COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def fig07_schedule_compare():
    from fwi.schedule import make_schedule, expert_schedule

    d = CTX.expA_bo
    if d is None:
        log.warning("Fig 7 skipped: Exp A not done"); return

    best_theta = d["best_theta"]
    bo_sched   = make_schedule(best_theta, f_sample=0.5)
    _ex_raw = ((CTX.expA_expert or {}).get("metrics", {})
               .get("schedule", {}).get("groups", []))
    if _ex_raw:
        ex_grps = [(_g["f_lo"], _g["f_hi"]) for _g in _ex_raw]
        ex_fmin, ex_fmax = ex_grps[0][0], ex_grps[-1][1]
    else:
        ex_grps, ex_fmin, ex_fmax = [(3.0,5.0),(4.0,10.0),(8.0,15.0)], 3.0, 20.0
    ex_sched = expert_schedule(f_min=ex_fmin, f_max=ex_fmax,
                                groups=ex_grps, f_sample=0.5)

    fig, axes = plt.subplots(2, 1, figsize=(W2 * 0.7, 3.0), sharex=True)
    grp_colors = [C["g0"], C["g1"], C["g2"], C["g3"],
                  "#9B7FDD", "#5CB8E4", "#E06C75"]

    for ax, (sched, title, col) in zip(axes, [
        (ex_sched, "(a) Expert schedule  (K = 3)", C["expert"]),
        (bo_sched,
         f"(b) BO-optimal schedule  (K = {len(bo_sched)},  "
         f"γ = {best_theta[3]:.2f},  β = {best_theta[4]:.2f})",
         C["bo"]),
    ]):
        for i, grp in enumerate(sched.groups):
            gc = grp_colors[i % len(grp_colors)]
            ax.barh(0, grp.f_hi - grp.f_lo, left=grp.f_lo,
                    height=0.55, color=gc, alpha=0.78,
                    edgecolor="white", linewidth=0.6)
            ax.plot(grp.f_center, 0, "|", color="white",
                    markersize=7, markeredgewidth=1.3)
            ax.text(grp.f_center, 0.37,
                    f"G{i+1}\n{grp.f_lo:.1f}–{grp.f_hi:.1f} Hz",
                    ha="center", va="bottom", fontsize=6.2, color=col)
        ax.set_ylabel(title, rotation=0, labelpad=6, va="center",
                      ha="left", fontsize=8, x=-0.01)
        ax.set_yticks([])
        ax.set_xlim(1.5, 22.0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", lw=0.4)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=9)
    axes[0].set_title("Frequency schedule comparison", fontsize=9, pad=4)
    fig.tight_layout(h_pad=1.2)
    save_fig(fig, "fig07_schedule_compare")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 8 — RECOVERED VELOCITY MODELS + DIFFERENCE MAPS
# ──────────────────────────────────────────────────────────────────────────────

def fig08_velocity_models():
    m_true = CTX.m_true
    if m_true is None:
        log.warning("Fig 8 skipped: true model not found"); return

    m_expert = CTX.m_expert()
    m_bo     = CTX.m_bo_best()
    m_init   = CTX.m_init

    # Build available panels dynamically
    models_top = [(m_true, "(a) True model"), (m_init, "(b) Starting model")]
    if m_expert is not None: models_top.append((m_expert, "(c) Expert FWI"))
    if m_bo     is not None: models_top.append((m_bo,    "(d) BO-optimal FWI"))

    ext              = model_extent_km()
    vmin, vmax       = vel_limits(m_true)
    n_top            = len(models_top)

    # Decide layout: top row = models, bottom row = difference maps
    has_diff = m_expert is not None or m_bo is not None
    nrows    = 2 if has_diff else 1

    fig = plt.figure(figsize=(W2, 2.5 * nrows + 0.5))
    gs  = gridspec.GridSpec(nrows, n_top, figure=fig, hspace=0.35, wspace=0.06)

    axes_top  = [fig.add_subplot(gs[0, i]) for i in range(n_top)]
    if has_diff:
        axes_bot = [fig.add_subplot(gs[1, i]) for i in range(n_top)]

    # ── Top row — raw velocity models ────────────────────────────────────────
    im_v = None
    for ax, (m, title) in zip(axes_top, models_top):
        if m is None:
            ax.text(0.5, 0.5, "Not yet\navailable",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=8, pad=3); continue
        im_v = ax.imshow(m / 1_000, extent=ext, aspect="auto",
                         cmap=CMAP_VEL, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("Distance (km)", fontsize=7)
    axes_top[0].set_ylabel("Depth (km)", fontsize=7)
    if im_v:
        cb1 = fig.colorbar(im_v, ax=axes_top, fraction=0.012, pad=0.01)
        cb1.set_label("Velocity (km/s)", fontsize=7)
        cb1.ax.tick_params(labelsize=6)

# ── Bottom row — difference maps ──────────────────────────────────────────
    if has_diff:
        # Panels e, f, g: each result minus the true model
        diff_panels = []
        if m_init   is not None: diff_panels.append((m_init   - m_true, "(e) Starting − True"))
        if m_expert is not None: diff_panels.append((m_expert - m_true, "(f) Expert − True"))
        if m_bo     is not None: diff_panels.append((m_bo     - m_true, "(g) BO − True"))

        # Panel h: BO minus Expert — shows directly where BO improves on the expert.
        # Positive (red) => BO faster than expert there; negative (blue) => slower.
        if (m_bo is not None) and (m_expert is not None):
            diff_panels.append((m_bo - m_expert, "(h) BO − Expert"))

        # Symmetric diverging scale across ALL difference panels
        diff_max = max(
            (np.abs(d).max() if d is not None else 0.0)
            for d, _ in diff_panels
        ) or 500.0

        im_d = None
        for ax, (d, title) in zip(axes_bot, diff_panels[:n_top]):
            if d is None:
                ax.axis("off"); continue
            im_d = ax.imshow(d / 1_000, extent=ext, aspect="auto",
                             cmap=CMAP_DIFF,
                             vmin=-diff_max/1_000, vmax=diff_max/1_000)
            ax.set_title(title, fontsize=8, pad=3)
            ax.set_xlabel("Distance (km)", fontsize=7)
        axes_bot[0].set_ylabel("Depth (km)", fontsize=7)

        # Turn off any bottom axis that did not receive a panel (safety net).
        for ax in axes_bot[len(diff_panels):]:
            ax.axis("off")

        if im_d:
            cb2 = fig.colorbar(im_d, ax=axes_bot, fraction=0.012, pad=0.01)
            cb2.set_label("Δ Velocity (km/s)", fontsize=7)
            cb2.ax.tick_params(labelsize=6)

    save_fig(fig, "fig08_velocity_models")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 9 — VERTICAL VELOCITY PROFILES
# ──────────────────────────────────────────────────────────────────────────────

def fig09_profiles():
    m_true = CTX.m_true
    if m_true is None:
        log.warning("Fig 9 skipped: true model not found"); return

    m_expert = CTX.m_expert()
    m_bo     = CTX.m_bo_best()
    m_init   = CTX.m_init

    z_km = np.arange(NZ) * DZ / 1_000
    x_positions_km = [4.0, 8.5, 13.0]   # 3 representative locations
    x_indices = [min(int(x * 1_000 / DX), NX - 1) for x in x_positions_km]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.5), sharey=True)

    for ax, (x_km, ix) in zip(axes, zip(x_positions_km, x_indices)):
        if m_init   is not None:
            ax.plot(m_init[:, ix]   / 1_000, z_km, ":", color=C["init"],
                    lw=1.1, label="Starting model")
        if m_expert is not None:
            ax.plot(m_expert[:, ix] / 1_000, z_km, "-", color=C["expert"],
                    lw=1.4, label="Expert FWI")
        if m_bo is not None:
            ax.plot(m_bo[:, ix]     / 1_000, z_km, "-", color=C["bo"],
                    lw=1.4, label="BO-optimal FWI")
        ax.plot(m_true[:, ix]   / 1_000, z_km, "-", color=C["ref"],
                lw=2.0, alpha=0.85, label="True model")

        ax.set_title(f"x = {x_km:.1f} km", fontsize=8)
        ax.set_xlabel("P-wave velocity (km/s)", fontsize=8)
        ax.invert_yaxis()
        ax.grid(lw=0.4, alpha=0.5)

    axes[0].set_ylabel("Depth (km)", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=True, loc="lower right")
    fig.suptitle("Vertical velocity profiles at three locations", fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig09_profiles")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 10 — NOISE ROBUSTNESS
# ──────────────────────────────────────────────────────────────────────────────

def fig10_noise():
    rows = CTX.expC
    if not rows:
        log.warning("Fig 10 skipped: Exp C not done"); return

    snr  = [r["snr_db"]  for r in rows]
    bo_J = [r["bo_J"]    for r in rows]
    ex_J = [r["expert_J"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.7, 3.0), sharey=False)

    # J vs SNR
    ax = axes[0]
    ax.plot(snr, bo_J, "o-", color=C["bo"],     lw=1.6, ms=6, label="BO-optimal")
    ax.plot(snr, ex_J, "s--", color=C["expert"], lw=1.6, ms=6, label="Expert")
    ax.set_xlabel("SNR (dB)", fontsize=9)
    ax.set_ylabel("Best J (normalised RMSE)", fontsize=9)
    ax.set_title("(a) Model quality vs noise level", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(lw=0.4, alpha=0.5)

    # Relative improvement vs SNR
    ax2 = axes[1]
    rel = [(e - b) / e * 100 for b, e in zip(bo_J, ex_J)]
    ax2.bar(snr, rel, color=C["bo"], alpha=0.75, edgecolor=C["bo"], width=6)
    ax2.axhline(0, color="grey", lw=0.7)
    ax2.set_xlabel("SNR (dB)", fontsize=9)
    ax2.set_ylabel("BO improvement over expert (%)", fontsize=9)
    ax2.set_title("(b) Relative gain of BO schedule", fontsize=9)
    ax2.grid(axis="y", lw=0.4, alpha=0.5)

    fig.tight_layout()
    save_fig(fig, "fig10_noise")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 11 — STARTING-MODEL SENSITIVITY
# ──────────────────────────────────────────────────────────────────────────────

def fig11_startmodel():
    dg = CTX.expD_good
    dp = CTX.expD_poor
    if dg is None and dp is None:
        log.warning("Fig 11 skipped: Exp D not done"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.7, 3.0))

    # ── Convergence traces ────────────────────────────────────────────────────
    ax = axes[0]
    for d, label, col in [(dg, "Good init", C["good"]),
                           (dp, "Poor init", C["poor"])]:
        if d and "incumbent_trace" in d:
            trace = np.array(d["incumbent_trace"])
            ev    = np.arange(1, len(trace) + 1)
            ax.plot(ev, trace, color=col, lw=1.5, label=label)
            ax.scatter(ev[-1], trace[-1], s=40, color=col, zorder=5)
    ax.set_xlabel("Evaluations", fontsize=9)
    ax.set_ylabel("Incumbent J", fontsize=9)
    ax.set_title("(a) Convergence trace", fontsize=9)
    ax.legend(fontsize=8); ax.grid(lw=0.4, alpha=0.5)

    # ── Best J bar comparison ─────────────────────────────────────────────────
    ax2 = axes[1]
    labels, vals, cols = [], [], []
    for d, lbl, col in [(dg, "Good init", C["good"]),
                         (dp, "Poor init", C["poor"])]:
        if d:
            labels.append(lbl); vals.append(d["best_J"]); cols.append(col)
    if vals:
        bars = ax2.bar(labels, vals, color=cols, alpha=0.78, edgecolor=cols, width=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Best J (normalised RMSE)", fontsize=9)
    ax2.set_title("(b) Final model quality", fontsize=9)
    ax2.grid(axis="y", lw=0.4, alpha=0.5)

    fig.suptitle("Starting-model sensitivity (Experiment D)", fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig11_startmodel")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 12 — PARAMETER IMPORTANCE (GP inverse length scales)
# ──────────────────────────────────────────────────────────────────────────────

def fig12_importance():
    d = CTX.expA_bo
    if d is None:
        log.warning("Fig 12 skipped: Exp A not done"); return

    imp = d.get("param_importances", {})
    if not imp:
        log.warning("Fig 12 skipped: param_importances missing from bo_result.json")
        return

    from fwi.schedule import PARAM_NAMES

    pretty = {"f_min": "$f_{\\min}$", "f_max": "$f_{\\max}$",
              "K_raw": "$K$",          "gamma": "$\\gamma$",
              "beta":  "$\\beta$"}
    names  = [pretty.get(k, k) for k in PARAM_NAMES]
    values = [imp.get(k, 0.0)  for k in PARAM_NAMES]
    order  = np.argsort(values)[::-1]
    names  = [names[i]  for i in order]
    values = [values[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.72, 3.0))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax = axes[0]
    bar_cols = [C["bo"]] * len(names)
    bars = ax.barh(names, values, color=bar_cols, alpha=0.78,
                   edgecolor=C["bo"], height=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)
    ax.set_xlabel("Relative importance  ($1/\\ell$, normalised)", fontsize=8)
    ax.set_title("(a) GP-derived sensitivity\n(inverse length scale)", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    # ── Compute budget pie ────────────────────────────────────────────────────
    ax2 = axes[1]
    labels_pie = ["FWI evaluations\n(toy2dac)", "BO overhead\n(GP + EI)", "LHS warm-start"]
    sizes  = [80, 5, 15]
    colors = [C["bo"], "#EF9F27", "#1D9E75"]
    wedges, _, autotexts = ax2.pie(
        sizes, labels=labels_pie, autopct="%1.0f%%",
        colors=colors, startangle=130,
        textprops={"fontsize": 7.5},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax2.set_title("(b) Compute budget\nbreakdown", fontsize=8)

    fig.suptitle("Parameter sensitivity and computational cost", fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig12_importance")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE REGISTRY
# ──────────────────────────────────────────────────────────────────────────────


def fig13_walltime():
    """Wall-time per FWI evaluation coloured by K (number of groups)."""
    records = load_jsonl(find_jsonl())
    if not records:
        log.warning("Fig 13 skipped: no iteration log"); return

    iters  = [r["iteration"]       for r in records]
    times  = [r["wall_time_s"]/60  for r in records]
    Ks     = [(r.get("metrics") or {}).get("schedule", {}).get("K", 0)
              for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.72, 3.0))
    k_vals = sorted(set(k for k in Ks if k)) or [1]
    norm_k = plt.Normalize(vmin=min(k_vals), vmax=max(k_vals))
    cmap_k = plt.cm.plasma
    bar_cols = [cmap_k(norm_k(k)) if k else "#888" for k in Ks]

    axes[0].bar(iters, times, color=bar_cols, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Evaluation index", fontsize=9)
    axes[0].set_ylabel("Wall time (min)", fontsize=9)
    axes[0].set_title("(a) Wall time per FWI call", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap_k, norm=norm_k)
    cb = fig.colorbar(sm, ax=axes[0], fraction=0.04, pad=0.02)
    cb.set_label("K (groups)", fontsize=8); cb.ax.tick_params(labelsize=7)

    if any(Ks):
        axes[1].scatter(Ks, times, s=45, c=bar_cols,
                        edgecolors="#333", linewidths=0.4, zorder=4)
        axes[1].set_xlabel("Number of frequency groups K", fontsize=9)
        axes[1].set_ylabel("Wall time (min)", fontsize=9)
        axes[1].set_title("(b) Cost scales with K", fontsize=9)
        axes[1].grid(lw=0.4, alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, "K not logged", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9)

    fig.suptitle("Computational cost of FWI evaluations", fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig13_walltime")


FIGURE_REGISTRY = {
    1:  ("Workflow schematic",            fig01_workflow,          "none"),
    2:  ("Marmousi model panels",         fig02_marmousi,          "template"),
    3:  ("Schedule parameterisation",     fig03_schedule_params,   "none"),
    4:  ("GP surrogate evolution",        fig04_gp_surrogate,      "exp_A + BoTorch"),
    5:  ("EI acquisition surface",        fig05_ei_surface,        "exp_A + BoTorch"),
    6:  ("Convergence comparison",        fig06_convergence,       "exp_B (exp_A fallback)"),
    7:  ("Schedule comparison",           fig07_schedule_compare,  "exp_A"),
    8:  ("Recovered velocity models",     fig08_velocity_models,   "exp_A + template"),
    9:  ("Vertical velocity profiles",    fig09_profiles,          "exp_A + template"),
    10: ("Noise robustness",              fig10_noise,             "exp_C"),
    11: ("Starting-model sensitivity",    fig11_startmodel,        "exp_D"),
    12: ("Parameter importance",          fig12_importance,        "exp_A"),
    13: ("Wall-time breakdown",            fig13_walltime,          "exp_A JSONL"),
}


def list_figures():
    """Print status of all figures (data available or not)."""
    print(f"\n{'#':>2}  {'Title':<34} {'Data needed':<24} {'Status'}")
    print("-" * 76)
    for n, (title, fn, data) in FIGURE_REGISTRY.items():
        # Quick data check
        status = "✓ ready"
        if "exp_A" in data:
            if not (RESULTS_DIR / "exp_A" / "bo_result.json").exists():
                status = "⏳ waiting for Exp A"
        if "exp_B" in data:
            if not (RESULTS_DIR / "exp_B" / "summary.json").exists():
                status = "⏳ waiting for Exp B"
        if "exp_C" in data:
            if not list((RESULTS_DIR / "exp_C").glob("snr_*")):
                status = "⏳ waiting for Exp C"
        if "exp_D" in data:
            if not (RESULTS_DIR / "exp_D" / "good" / "result.json").exists():
                status = "⏳ waiting for Exp D"
        if "template" in data or data == "none":
            if not TEMPLATE_DIR.exists():
                status = "✗ template dir not found"
        print(f"{n:>2}  {title:<34} {data:<24} {status}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all manuscript figures for the BO-FWI paper."
    )
    parser.add_argument(
        "--figs", nargs="*", type=int, default=None,
        metavar="N",
        help="Figure numbers to generate (default: all). E.g. --figs 6 7 8",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all figures and their data status, then exit.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=FIGURES_DIR,
        help=f"Output directory (default: {FIGURES_DIR})",
    )
    args = parser.parse_args()

    if args.list:
        list_figures()
        return

    # Override output dir globally
    if args.outdir != FIGURES_DIR:
        globals()["FIGURES_DIR"] = args.outdir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    to_run = args.figs if args.figs else list(FIGURE_REGISTRY)

    log.info(f"Generating {len(to_run)} figure(s) → {FIGURES_DIR}")
    log.info("-" * 54)

    n_ok = n_skip = 0
    for n in to_run:
        if n not in FIGURE_REGISTRY:
            log.warning(f"Unknown figure number: {n} (valid: 1–12)")
            continue
        title, fn, _ = FIGURE_REGISTRY[n]
        log.info(f"Fig {n:02d}  {title}")
        try:
            fn()
            n_ok += 1
        except Exception as exc:
            log.error(f"  ✗  Fig {n} failed: {exc}")
            n_skip += 1

    log.info("-" * 54)
    log.info(f"Done.  {n_ok} generated,  {n_skip} failed/skipped.")
    log.info(f"Output: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
