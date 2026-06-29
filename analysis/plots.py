"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

analysis/plots.py
──────────────────
All figures for the manuscript, each as an independent function.

Figure inventory 
────────────────────────────────────────
  fig1_workflow()           — BO loop schematic
  fig2_marmousi_panels()    — true / init / poor starting models
  fig3_schedule_param()     — effect of gamma on spacing
  fig4_gp_evolution()       — GP posterior at successive iterations
  fig5_acquisition()        — EI surface + next candidate
  fig6_convergence()        — incumbent J vs evaluation (headline figure)
  fig7_schedule_compare()   — discovered vs expert schedule
  fig8_velocity_models()    — true / expert / BO recovered models + diff maps
  fig9_profiles()           — vertical velocity profiles
  fig10_noise_robustness()  — RMSE vs SNR
  fig11_start_model()       — good vs poor starting model results
  fig12_importances()       — GP length-scale importance bar chart

Call save_fig(fig, "fig6_convergence") to write to ./results/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; switch to "TkAgg" locally
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family":        "serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

FIGURES_DIR = Path("./results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette (colour-blind safe)
C_BO     = "#2166ac"   # blue   — BO
C_RS     = "#d6604d"   # red    — random search
C_GS     = "#4dac26"   # green  — grid search
C_EXP    = "#8073ac"   # purple — expert
CMAP_VEL = "jet"       # standard for velocity models in exploration seismology


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str, exts: Tuple[str, ...] = ("pdf", "png")) -> None:
    for ext in exts:
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)


def _axes_label(ax, letter: str) -> None:
    """Add a bold panel label (a), (b), … to an axes."""
    ax.text(-0.12, 1.02, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3 — Schedule parameterisation (gamma effect)
# ──────────────────────────────────────────────────────────────────────────────

def fig3_schedule_param(
    f_min: float = 3.0,
    f_max: float = 22.0,
    K: int = 5,
) -> plt.Figure:
    from fwi.schedule import make_schedule
    gammas = [0.5, 1.0, 1.5, 2.5]
    colors = ["#d73027", "#fc8d59", "#4575b4", "#313695"]

    fig, axes = plt.subplots(len(gammas), 1, figsize=(8, 5), sharex=True)

    for ax, g, col in zip(axes, gammas, colors):
        theta = [f_min, f_max, K, g, 0.1]
        sched = make_schedule(theta)
        for grp in sched.groups:
            ax.barh(
                0, grp.f_hi - grp.f_lo, left=grp.f_lo,
                height=0.6, color=col, alpha=0.75, edgecolor="white", lw=0.5,
            )
            ax.plot(grp.f_center, 0, "k|", ms=8, lw=1.5)
        ax.set_ylabel(f"γ={g}", fontsize=9, rotation=0, labelpad=40, va="center")
        ax.set_yticks([])
        ax.set_xlim(f_min - 1, f_max + 1)
        ax.grid(axis="x", lw=0.5, alpha=0.5)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        "Effect of spacing exponent γ on frequency-group layout\n"
        f"(f_min={f_min} Hz, f_max={f_max} Hz, K={K})",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 6 — Convergence (incumbent J vs evaluations) — HEADLINE FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def fig6_convergence(
    bo_trace:      np.ndarray,
    rs_mean:       np.ndarray,
    rs_std:        np.ndarray,
    gs_trace:      np.ndarray,
    expert_J:      float,
    n_init:        int = 10,
    target_J:      Optional[float] = None,
) -> plt.Figure:
    """
    Plot incumbent objective J vs number of FWI evaluations.

    Parameters
    ----------
    bo_trace  : (n_total,) incumbent J for BO
    rs_mean   : (n_total,) mean incumbent J over RS seeds
    rs_std    : (n_total,) std  incumbent J over RS seeds
    gs_trace  : (n_gs,)    incumbent J for grid search
    expert_J  : scalar J for the expert schedule
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    evals_bo = np.arange(1, len(bo_trace) + 1)
    evals_rs = np.arange(1, len(rs_mean)  + 1)
    evals_gs = np.arange(1, len(gs_trace) + 1)

    # BO trace
    ax.plot(evals_bo, bo_trace, color=C_BO, lw=2.0, label="BO (EI + GP)", zorder=4)

    # Warm-start boundary
    ax.axvline(n_init + 0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(n_init + 0.8, ax.get_ylim()[1] * 0.98, "BO\nstarts",
            fontsize=7, color="grey", va="top")

    # Random search (mean ± 1 std)
    ax.plot(evals_rs, rs_mean, color=C_RS, lw=1.5, label="Random search (mean)")
    ax.fill_between(evals_rs, rs_mean - rs_std, rs_mean + rs_std,
                    color=C_RS, alpha=0.15, label="Random (±1σ)")

    # Grid search
    ax.step(evals_gs, gs_trace, where="post",
            color=C_GS, lw=1.5, label="Grid search", ls="-.")

    # Expert horizontal line
    ax.axhline(expert_J, color=C_EXP, lw=1.2, ls=":", label=f"Expert J={expert_J:.4f}")

    # Target threshold
    if target_J is not None:
        ax.axhline(target_J, color="black", lw=0.8, ls="--", alpha=0.4,
                   label=f"Target J={target_J:.4f}")

    ax.set_xlabel("Number of FWI evaluations")
    ax.set_ylabel("Incumbent normalised RMSE (J)")
    ax.set_title("Sample efficiency: BO vs baselines")
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 7 — Discovered vs expert schedule
# ──────────────────────────────────────────────────────────────────────────────

def fig7_schedule_compare(
    bo_theta:   np.ndarray,
    expert_f_min: float = 3.0,
    expert_f_max: float = 20.0,
    expert_groups: Optional[List[tuple]] = None,
    f_sample:   float = 0.5,
) -> plt.Figure:
    from fwi.schedule import make_schedule, expert_schedule

    bo_sched  = make_schedule(bo_theta, f_sample=f_sample)
    exp_sched = expert_schedule(f_min=expert_f_min, f_max=expert_f_max,
                                groups=expert_groups, f_sample=f_sample)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    labels    = ["Expert schedule", f"BO-optimal schedule (K={len(bo_sched)})"]
    scheds    = [exp_sched, bo_sched]
    colors    = [C_EXP, C_BO]

    for ax, sched, label, col in zip(axes, scheds, labels, colors):
        for grp in sched.groups:
            ax.barh(0, grp.f_hi - grp.f_lo, left=grp.f_lo,
                    height=0.55, color=col, alpha=0.7,
                    edgecolor="white", lw=0.8)
            ax.plot(grp.f_center, 0, "k|", ms=9, lw=1.5)
            ax.text(grp.f_center, 0.38, f"{grp.f_center:.1f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_ylabel(label, rotation=0, labelpad=120, va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(min(expert_f_min, float(bo_theta[0])) - 1,
                    max(expert_f_max, float(bo_theta[1])) + 1)
        ax.grid(axis="x", lw=0.5, alpha=0.4)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Frequency schedule comparison", y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 8 — Velocity models (true / expert / BO) + difference maps
# ──────────────────────────────────────────────────────────────────────────────

def fig8_velocity_models(
    m_true:   np.ndarray,
    m_expert: np.ndarray,
    m_bo:     np.ndarray,
    dx: float = 1.25,
    dz: float = 1.25,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    vmin  = vmin or float(m_true.min())
    vmax  = vmax or float(m_true.max())
    dv    = max(abs(m_expert - m_true).max(), abs(m_bo - m_true).max())

    nz, nx = m_true.shape
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]   # km

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.08)

    panels_top  = [m_true, m_expert, m_bo]
    titles_top  = ["(a) True Marmousi", "(b) Expert FWI", "(c) BO-optimised FWI"]
    axes_top    = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panels_bot  = [None, m_expert - m_true, m_bo - m_true]
    titles_bot  = ["", "(d) Expert − True", "(e) BO − True"]
    axes_bot    = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Top row — velocity models
    im_v = None
    for ax, panel, title in zip(axes_top, panels_top, titles_top):
        im_v = ax.imshow(panel / 1000, extent=extent, aspect="auto",
                         cmap=CMAP_VEL, vmin=vmin/1000, vmax=vmax/1000)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xlabel("Distance (km)", fontsize=8)
    axes_top[0].set_ylabel("Depth (km)", fontsize=8)

    cb1 = fig.colorbar(im_v, ax=axes_top, fraction=0.015, pad=0.01)
    cb1.set_label("Velocity (km/s)", fontsize=8)

    # Bottom row — difference maps
    axes_bot[0].axis("off")
    im_d = None
    for ax, panel, title in zip(axes_bot[1:], panels_bot[1:], titles_bot[1:]):
        im_d = ax.imshow(panel / 1000, extent=extent, aspect="auto",
                         cmap="RdBu_r", vmin=-dv/1000, vmax=dv/1000)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xlabel("Distance (km)", fontsize=8)
    axes_bot[1].set_ylabel("Depth (km)", fontsize=8)

    cb2 = fig.colorbar(im_d, ax=axes_bot[1:], fraction=0.015, pad=0.01)
    cb2.set_label("Δ Velocity (km/s)", fontsize=8)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 9 — Vertical profiles
# ──────────────────────────────────────────────────────────────────────────────

def fig9_profiles(
    m_true:   np.ndarray,
    m_expert: np.ndarray,
    m_bo:     np.ndarray,
    m_init:   np.ndarray,
    x_positions_km: List[float],   # e.g. [2.0, 4.5, 7.0]
    dx: float = 1.25,
    dz: float = 1.25,
) -> plt.Figure:
    nz, nx = m_true.shape
    z_km   = np.arange(nz) * dz / 1000

    fig, axes = plt.subplots(1, len(x_positions_km),
                              figsize=(4 * len(x_positions_km), 5), sharey=True)
    if len(x_positions_km) == 1:
        axes = [axes]

    for ax, x_km in zip(axes, x_positions_km):
        ix = min(int(x_km * 1000 / dx), nx - 1)

        ax.plot(m_init[:, ix]   / 1000, z_km, ":", color="grey",  lw=1.2,
                label="Initial")
        ax.plot(m_expert[:, ix] / 1000, z_km, "-", color=C_EXP,   lw=1.5,
                label="Expert")
        ax.plot(m_bo[:, ix]     / 1000, z_km, "-", color=C_BO,    lw=1.5,
                label="BO")
        ax.plot(m_true[:, ix]   / 1000, z_km, "k-", lw=2.0, alpha=0.85,
                label="True")

        ax.set_title(f"x = {x_km} km", fontsize=9)
        ax.set_xlabel("Velocity (km/s)", fontsize=8)
        ax.invert_yaxis()
        ax.grid(lw=0.4, alpha=0.5)

    axes[0].set_ylabel("Depth (km)", fontsize=8)
    axes[-1].legend(fontsize=8, frameon=True, loc="lower right")
    fig.suptitle("Vertical velocity profiles", y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 10 — Noise robustness
# ──────────────────────────────────────────────────────────────────────────────

def fig10_noise_robustness(
    snr_db_list:  List[float],
    bo_J_list:    List[float],
    expert_J_list: List[float],
    metric_name:  str = "Normalised RMSE (J)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr_db_list, bo_J_list,     "o-",  color=C_BO,  lw=1.8,
            ms=7, label="BO-optimal")
    ax.plot(snr_db_list, expert_J_list, "s--", color=C_EXP, lw=1.8,
            ms=7, label="Expert")
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title("Model quality vs noise level", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(lw=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig 12 — Parameter importance (GP length scales)
# ──────────────────────────────────────────────────────────────────────────────

def fig12_importances(
    importances: Dict[str, float],
) -> plt.Figure:
    from fwi.schedule import PARAM_NAMES
    names  = list(importances.keys())
    values = list(importances.values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bar chart
    ax = axes[0]
    bars = ax.barh(names, values, color=[C_BO] * len(names), alpha=0.8)
    ax.set_xlabel("Relative importance (1/ℓ, normalised)", fontsize=9)
    ax.set_title("GP-derived parameter importance\n(inverse length scale)", fontsize=10)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    # Right: simple cost-breakdown pie placeholder
    ax2 = axes[1]
    labels  = ["FWI evaluations", "BO overhead", "LHS warm-start"]
    sizes   = [85, 5, 10]   # <<< Replace with measured values
    wedges, _, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=["#4575b4", "#d73027", "#fee090"],
        startangle=140, textprops={"fontsize": 8},
    )
    ax2.set_title("Compute budget breakdown", fontsize=10)

    fig.suptitle("Parameter sensitivity and computational cost", fontsize=11)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fig F — J vs Ju scatter (Experiment F)
# ──────────────────────────────────────────────────────────────────────────────

def fig_F_correlation(
    J_vals:   np.ndarray,
    Ju_vals:  np.ndarray,
    pearson_r: float,
) -> plt.Figure:
    valid = np.isfinite(J_vals) & np.isfinite(Ju_vals)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(J_vals[valid], Ju_vals[valid], s=35, alpha=0.7, color=C_BO)

    # Linear fit
    m, b = np.polyfit(J_vals[valid], Ju_vals[valid], 1)
    x_line = np.linspace(J_vals[valid].min(), J_vals[valid].max(), 50)
    ax.plot(x_line, m * x_line + b, "--", color=C_EXP, lw=1.5,
            label=f"Linear fit  r={pearson_r:.3f}")

    ax.set_xlabel("J (supervised — normalised RMSE)", fontsize=9)
    ax.set_ylabel("Ju (unsupervised — data residual)", fontsize=9)
    ax.set_title("Fidelity of the field-deployable objective", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(lw=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: generate all figures from saved JSON results
# ──────────────────────────────────────────────────────────────────────────────

def make_all_figures(results_dir: str | Path = "./results") -> None:
    """
    Load JSON result files and generate all manuscript figures.
    Run this AFTER completing all experiments.
    """
    rd = Path(results_dir)

    # Fig 3 — schedule parameterisation (no data needed)
    save_fig(fig3_schedule_param(), "fig3_schedule_param")

    # Fig 6 — convergence
    exp_B = rd / "exp_B"
    if exp_B.exists():
        bo_trace = np.array(json.loads((exp_B / "bo_trace.json").read_text()))
        rs_mean  = np.array(json.loads((exp_B / "rs_mean.json").read_text()))
        rs_std   = np.array(json.loads((exp_B / "rs_std.json").read_text()))
        gs_trace = np.array(json.loads((exp_B / "gs_trace.json").read_text()))
        expert_J = json.loads((rd / "exp_A" / "expert_result.json").read_text())["J"]
        save_fig(fig6_convergence(bo_trace, rs_mean, rs_std, gs_trace, expert_J),
                 "fig6_convergence")

    # Fig 7 — schedule comparison
    exp_A = rd / "exp_A"
    if exp_A.exists():
        bo_theta = np.array(json.loads(
            (exp_A / "bo_result.json").read_text())["best_theta"])
        save_fig(fig7_schedule_compare(bo_theta), "fig7_schedule_compare")

    # Fig 10 — noise robustness
    exp_C = rd / "exp_C"
    if exp_C.exists():
        snr_list, bo_list, exp_list = [], [], []
        for snr_dir in sorted(exp_C.iterdir()):
            if snr_dir.is_dir():
                data = json.loads((snr_dir / "results.json").read_text())
                snr_list.append(data["snr_db"])
                bo_list.append(data["bo_J"])
                exp_list.append(data["expert_J"])
        save_fig(fig10_noise_robustness(snr_list, bo_list, exp_list),
                 "fig10_noise_robustness")

    # Fig F — J vs Ju correlation
    exp_F = rd / "exp_F"
    if exp_F.exists():
        data = json.loads((exp_F / "correlation.json").read_text())
        save_fig(fig_F_correlation(
            np.array(data["J_values"]),
            np.array(data["Ju_values"]),
            data["pearson_r"],
        ), "figF_J_Ju_correlation")

    print("All available figures generated.")


if __name__ == "__main__":
    make_all_figures()
