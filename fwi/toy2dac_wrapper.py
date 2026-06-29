"""
Project : BO-Toy2DAC-Frequency-Scheduling
Written by MAU
GitHub  : https://github.com/Ugeoscience/BO-Toy2DAC-Frequency-Scheduling
License : Copyright (c) 2026, Ugeoscience. BSD 3-Clause License: redistribution and use in source/binary forms are permitted, with or without modification, provided the copyright notice, license conditions, and disclaimer are retained; the Ugeoscience name or contributor names may not be used for endorsement without prior written permission; the software is provided "AS IS" without warranties or liability.

fwi/toy2dac_wrapper.py
───────────────────────
Python interface to toy2dac V2.6 (SEISCOPE, Métivier & Brossier 2016).
"""

from __future__ import annotations

import os
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schedule import FrequencySchedule, FrequencyGroup

logger = logging.getLogger(__name__)

TEMPLATE = ""
BIN      = ""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Toy2dacConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    toy2dac_bin:     str = BIN
    base_data_dir:   str = TEMPLATE
    true_model_file: str = f"{TEMPLATE}/vp_Marmousi_exact"
    init_model_file: str = f"{TEMPLATE}/vp_Marmousi_init"   # default start
    density_file:    str = f"{TEMPLATE}/rho"
    acq_file:        str = f"{TEMPLATE}/acqui"

    # ── Grid (681×141 @ 25 m → 384084 bytes float32 ✓) ───────────────────────
    nx: int   = 681
    nz: int   = 141
    dx: float = 25.0   # m  (isotropic grid: dx == dz)
    dz: float = 25.0   # m

    # ── MPI ──────────────────────────────────────────────────────────────────
    mpi_np:       int = 16       # number of MPI ranks  (mpirun -n N)
    mpi_launcher: str = "mpirun"  # "mpirun" or "srun" for SLURM clusters

    # ── Per-process threads ───────────────────────────────────────────────────
    n_threads: int = 1          # OMP_NUM_THREADS per MPI rank (usually 1)

    # ── Binary I/O ───────────────────────────────────────────────────────────
    model_dtype: str = "float32"
    model_order: str = "F"      # Fortran column-major


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FWIResult:
    final_model:     np.ndarray
    group_histories: List[Dict]
    schedule:        FrequencySchedule
    wall_time_s:     float
    work_dir:        Path
    success:         bool = True
    error_msg:       str  = ""

    @property
    def n_groups(self) -> int:
        return len(self.group_histories)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Toy2dacWrapper:
    """
    Drives toy2dac V2.6 for a complete multi-scale FWI run.

    Usage
    ─────
    cfg     = Toy2dacConfig(mpi_np=8, init_model_file="/path/to/custom_init")
    wrapper = Toy2dacWrapper(cfg, work_root=Path(""))
    result  = wrapper.run(schedule, run_id="bo_iter_001")
    model   = result.final_model   # (nz=141, nx=681) float32 array
    """

    def __init__(self, config: Toy2dacConfig, work_root: Path = Path("./runs")):
        self.cfg       = config
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        schedule:           FrequencySchedule,
        run_id:             str            = "run",
        snr_db:             Optional[float] = None,   # None = noise-free
        keep_intermediates: bool           = False,
    ) -> FWIResult:
        """
        Execute a complete multi-scale FWI for the given frequency schedule.

        Each FrequencyGroup triggers:
          1. MODELING (mode=0) with vp_Marmousi_exact → data_modeling
          2. Optional Gaussian noise injection into data_modeling
          3. INVERSION (mode=1) with current starting model → param_vp_final

        Parameters
        ----------
        schedule  : FrequencySchedule  — the BO candidate to evaluate
        run_id    : str                — unique label; used for the work dir
        snr_db    : float | None       — if set, adds Gaussian noise at that
                    SNR (dB) to data_modeling before inversion (Experiment C)
        keep_intermediates : bool      — if False, wavefield/residual files
                    are removed after each group to save disk
        """
        work_dir = self.work_root / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{run_id}] FWI start  K={len(schedule)}  "
                    f"snr_db={snr_db}  dir={work_dir}")

        t0            = time.perf_counter()
        current_model = self._load_model(self.cfg.init_model_file)
        histories: List[Dict] = []

        try:
            for group in schedule.groups:
                grp_dir = work_dir / f"group_{group.index:02d}"
                grp_dir.mkdir(exist_ok=True)

                history = self._run_group(
                    group=group,
                    start_model=current_model,
                    grp_dir=grp_dir,
                    snr_db=snr_db,
                )
                histories.append(history)
                current_model = history["final_model"]

                logger.info(
                    f"  [{run_id}] group {group.index:02d} done  "
                    f"fcost_reduction={history.get('misfit_reduction', float('nan')):.4f}"
                )

                if not keep_intermediates:
                    for f in grp_dir.iterdir():
                        if f.name.startswith("wavefield") or f.name.startswith("data_cal"):
                            if not f.is_symlink():
                                f.unlink(missing_ok=True)

        except Exception as exc:
            logger.error(f"[{run_id}] FWI failed at group {len(histories)}: {exc}")
            return FWIResult(
                final_model=current_model,
                group_histories=histories,
                schedule=schedule,
                wall_time_s=time.perf_counter() - t0,
                work_dir=work_dir,
                success=False,
                error_msg=str(exc),
            )

        # Persist final model
        self._save_model(current_model, work_dir / "vp_final")
        return FWIResult(
            final_model=current_model,
            group_histories=histories,
            schedule=schedule,
            wall_time_s=time.perf_counter() - t0,
            work_dir=work_dir,
            success=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-group: MODELING → (noise) → INVERSION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_group(
        self,
        group:       FrequencyGroup,
        start_model: np.ndarray,
        grp_dir:     Path,
        snr_db:      Optional[float] = None,
    ) -> Dict:
        """
        Full synthetic workflow for one frequency group:
          A. Link static template files (once per group dir)
          B. Write freq_management for this group
          C. Run MODELING with vp_Marmousi_exact → data_modeling
          D. [Optional] inject Gaussian noise into data_modeling
          E. Write current start model as vp_Marmousi_init
          F. Run INVERSION → param_vp_final
          G. Read param_vp_final and parse convergence
        """

        # A ── Link template files ──────────────────────────────────────────
        self._link_static_files(grp_dir)

        # Clean mutable outputs left by any previous failed run
        for name in _MUTABLE_OUTPUTS:
            p = grp_dir / name
            if p.exists() and not p.is_symlink():
                p.unlink()

        # B ── Frequency file ───────────────────────────────────────────────
        self._write_freq_file(grp_dir / "freq_management", group.frequencies)

        # ═══════════════════════════════════════════════════════════════════
        # C — MODELING MODE (mode=0): generate synthetic observed data
        # ═══════════════════════════════════════════════════════════════════
        (grp_dir / "toy2dac_input").write_text(
            "0\n"           # mode 0 = MODELING
            "1\n"           # forward tool: 1 = isotropic acoustic
            "acqui\n"       # acquisition file (symlinked in grp_dir)
        )
        (grp_dir / "fdfd_input").write_text(
            f"{self.cfg.nz} {self.cfg.nx}\n"
            f"{self.cfg.dx:.1f}\n"
            "vp_Marmousi_exact qp rho epsilon_m delta_m theta_m\n"
            "90. 10\n"
            "1\n"           # Hicks interpolation
            "0\n"           # free surface
            "1 0\n"         # source type (1=vert force), receiver type (0=pressure)
            "0.\n"          # Laplace constant
        )

        stdout_mod, _ = self._execute("MODELING", grp_dir)

        data_mod = grp_dir / "data_modeling"
        if not data_mod.exists():
            raise RuntimeError(
                f"MODELING step did not produce data_modeling in {grp_dir}"
            )

        # D ── Noise injection (Experiment C) ──────────────────────────────
        if snr_db is not None:
            self._inject_noise(data_mod, snr_db)
            logger.info(f"    Noise injected at SNR={snr_db} dB")

        # ═══════════════════════════════════════════════════════════════════
        # E — Write current starting model
        # ═══════════════════════════════════════════════════════════════════
        self._save_model(start_model, grp_dir / "vp_Marmousi_init")

        # ═══════════════════════════════════════════════════════════════════
        # F — INVERSION MODE (mode=1)
        # ═══════════════════════════════════════════════════════════════════
        (grp_dir / "toy2dac_input").write_text(
            "1\n"           # mode 1 = INVERSION
            "1\n"
            "acqui\n"
        )
        (grp_dir / "fdfd_input").write_text(
            f"{self.cfg.nz} {self.cfg.nx}\n"
            f"{self.cfg.dx:.1f}\n"
            "vp_Marmousi_init qp rho epsilon_m delta_m theta_m\n"
            "90. 10\n"
            "1\n"
            "0\n"
            "1 0\n"
            "0.\n"
        )
        # fwi_input is already symlinked from template (line 1 = "data_modeling")
        stdout_inv, stderr_inv = self._execute("INVERSION", grp_dir)

        # G ── Read output model ────────────────────────────────────────────
        model_out = grp_dir / "param_vp_final"
        if not model_out.exists():
            raise RuntimeError(
                f"INVERSION did not produce param_vp_final in {grp_dir}\n"
                f"stderr:\n{stderr_inv[-1000:]}"
            )
        final_model = self._load_model(model_out)

        # H ── Parse convergence (prefer iterate file, fall back to stdout)
        convergence = self._parse_convergence(grp_dir, stdout_inv)
        convergence.update({
            "final_model":      final_model,
            "group_index":      group.index,
            "frequencies_hz":   group.frequencies.tolist(),
            "f_lo":             group.f_lo,
            "f_hi":             group.f_hi,
        })
        return convergence

    # ─────────────────────────────────────────────────────────────────────────
    # MPI execution
    # ─────────────────────────────────────────────────────────────────────────

    def _execute(self, mode: str, cwd: Path) -> Tuple[str, str]:
        """
        Run toy2dac via MPI:  mpirun -n N /path/to/toy2dac
        toy2dac V2.6 reads all input files (toy2dac_input, fdfd_input,
        fwi_input, freq_management, …) from the current working directory.
        No filename argument is passed on the command line.
        """
        cmd = [
            self.cfg.mpi_launcher,
            "-n", str(self.cfg.mpi_np),
            str(Path(self.cfg.toy2dac_bin).resolve()),
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.cfg.n_threads)

        logger.info(f"    [{mode}] {' '.join(cmd)}  (cwd={cwd.name})")

        proc = subprocess.run(
            cmd, cwd=cwd, env=env,
            capture_output=True, text=True,
        )

        # Log stdout to a file so nothing is lost on failure
        (cwd / f"stdout_{mode.lower()}.txt").write_text(proc.stdout)
        (cwd / f"stderr_{mode.lower()}.txt").write_text(proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"toy2dac [{mode}] failed (exit {proc.returncode})\n"
                f"Last stderr:\n{proc.stderr[-2000:]}"
            )
        return proc.stdout, proc.stderr

    # ─────────────────────────────────────────────────────────────────────────
    # freq_management writer
    # ─────────────────────────────────────────────────────────────────────────

    def _write_freq_file(self, path: Path, frequencies: np.ndarray) -> None:
        """
        toy2dac V2.6 freq_management format:
          line 1: NFREQ  (integer)
          line 2: f1 f2 … fN  (Hz, space-separated, 6 decimal places)
        """
        freqs_str = " ".join(f"{float(f):.6f}" for f in frequencies)
        path.write_text(f"{len(frequencies)}\n{freqs_str}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Binary I/O  (raw float32, Fortran column-major)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self, path: str | Path) -> np.ndarray:
        """
        Load a toy2dac binary model file.
        Format: raw float32, NO Fortran record markers, column-major (F-order).
        Size must be exactly nz × nx × 4 bytes.
        """
        path  = Path(path)
        dtype = np.dtype(self.cfg.model_dtype)
        raw   = np.frombuffer(path.read_bytes(), dtype=dtype)

        expected = self.cfg.nz * self.cfg.nx
        if raw.size != expected:
            raise ValueError(
                f"{path.name}: got {raw.size} floats, expected {expected} "
                f"(nz={self.cfg.nz} × nx={self.cfg.nx})"
            )
        return raw.reshape((self.cfg.nz, self.cfg.nx), order="F")

    def _save_model(self, model: np.ndarray, path: Path) -> None:
        """Save (nz, nx) array as raw float32 in Fortran column-major order."""
        arr = np.asarray(model, dtype=self.cfg.model_dtype)
        path.write_bytes(arr.ravel(order="F").tobytes())

    def load_true_model(self) -> np.ndarray:
        """Load vp_Marmousi_exact for metric computation."""
        return self._load_model(self.cfg.true_model_file)

    # ─────────────────────────────────────────────────────────────────────────
    # Noise injection  (Experiment C)
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_noise(
        self,
        data_path: Path,
        snr_db:    float,
        seed:      int = 42,
    ) -> None:
        """
        Add zero-mean Gaussian noise to data_modeling at the specified SNR.

        data_modeling is stored as raw float32 (complex frequency-domain data
        laid out as alternating Re/Im float32 values, or as packed complex64).
        Treating the whole file as float32 is correct for noise injection since
        we want to perturb the amplitude uniformly.

        SNR (dB) = 20 · log10( ‖signal‖ / ‖noise‖ )
        """
        raw      = np.frombuffer(data_path.read_bytes(), dtype=np.float32).copy()
        rng      = np.random.default_rng(seed)
        sig_rms  = np.sqrt(np.mean(raw ** 2))
        snr_lin  = 10.0 ** (snr_db / 20.0)
        noise_std = sig_rms / snr_lin
        raw      += rng.normal(0.0, noise_std, raw.shape).astype(np.float32)
        data_path.write_bytes(raw.tobytes())

    # ─────────────────────────────────────────────────────────────────────────
    # Static file linking
    # ─────────────────────────────────────────────────────────────────────────

    def _link_static_files(self, grp_dir: Path) -> None:
        """
        Populate the group working directory with all static template files.

        Strategy:
          Binary model files (_BINARY_MODEL_FILES): always COPY.
            Fortran DIRECT ACCESS reads fail on cross-filesystem symlinks —
            the kernel creates a 0-byte placeholder, giving "Non-existing
            record number" on record 1.  Physical copies avoid this entirely.
          Everything else (acqui 6.5 MB, fwi_input, mumps_input …): symlink
            to save disk space.

        Safety: any 0-byte file that Fortran accidentally created is removed
        before we attempt to copy/link, so stale artifacts never block re-runs.
        """
        base            = Path(self.cfg.base_data_dir)
        expected_size   = self.cfg.nx * self.cfg.nz * 4   # float32 model bytes

        for name in _STATIC_NAMES:
            src = base / name
            dst = grp_dir / name

            # ── Remove Fortran-created 0-byte ghost files ─────────────────
            if dst.exists() and not dst.is_symlink():
                if dst.stat().st_size == 0 and name in _BINARY_MODEL_FILES:
                    logger.warning(
                        f"Removing 0-byte Fortran artifact: {dst.name}  "
                        f"(likely from previous failed run)"
                    )
                    dst.unlink()

            # ── Skip if already present and non-empty ─────────────────────
            if dst.is_symlink() or (dst.exists() and dst.stat().st_size > 0):
                continue

            # ── Source missing ─────────────────────────────────────────────
            if not src.exists():
                if name == "data_weight":
                    dst.write_bytes(b"")    # legitimately empty placeholder
                else:
                    logger.warning(f"Template file not found (skipped): {src}")
                continue

            # ── Binary model files: physical copy ──────────────────────────
            if name in _BINARY_MODEL_FILES:
                shutil.copy2(src, dst)
                # Verify the copy has the right byte count
                got  = dst.stat().st_size
                want = expected_size
                if name not in ("fbathy", "mumps_input") and got != want:
                    raise RuntimeError(
                        f"Copy size mismatch for {name}: "
                        f"got {got} bytes, expected {want} "
                        f"(nx={self.cfg.nx} × nz={self.cfg.nz} × 4)"
                    )
                logger.debug(f"Copied  {name}  ({got:,} bytes)")

            # ── Everything else: symlink (fallback to copy) ────────────────
            else:
                try:
                    dst.symlink_to(src.resolve())
                    logger.debug(f"Linked  {name}  → {src.resolve()}")
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied  {name}  (symlink failed)")

    # ─────────────────────────────────────────────────────────────────────────
    # Convergence parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_convergence(self, grp_dir: Path, stdout: str) -> Dict:
        """
        Primary: parse iterate_LB.dat or iterate_PLB.dat (SEISCOPE format):
          <iter>  fcost =  X.XXXXXE+00  gnorm =  X.XXXXXE+00  step = ...

        Fallback: scan stdout for lines containing 'fcost' or 'cost'.
        """
        misfit_vals:  List[float] = []
        gnorm_vals:   List[float] = []

        # ── Try iterate files first ────────────────────────────────────────
        for fname in ("iterate_LB.dat", "iterate_PLB.dat"):
            iterate_path = grp_dir / fname
            if iterate_path.exists() and iterate_path.stat().st_size > 0:
                for line in iterate_path.read_text().splitlines():
                    nums = _extract_floats(line)
                    # Format: iter_num fcost gnorm step  (≥3 numbers)
                    if len(nums) >= 3:
                        misfit_vals.append(nums[1])   # fcost
                        gnorm_vals.append(nums[2])    # gnorm
                if misfit_vals:
                    break   # got data from this file; skip stdout

        # ── Fallback: scan stdout ──────────────────────────────────────────
        if not misfit_vals:
            for line in stdout.splitlines():
                ll = line.lower()
                if "fcost" in ll or ("cost" in ll and "=" in ll):
                    nums = _extract_floats(line)
                    if nums:
                        misfit_vals.append(nums[0])
                if "gnorm" in ll or ("gradient" in ll and "norm" in ll):
                    nums = _extract_floats(line)
                    if nums:
                        gnorm_vals.append(nums[0])

        arr = np.array(misfit_vals) if misfit_vals else np.array([np.nan])
        reduction = (
            (arr[0] - arr[-1]) / (arr[0] + 1e-30)
            if arr.size > 1 and np.isfinite(arr[0]) else 0.0
        )

        return {
            "misfit_history":   arr.tolist(),
            "gnorm_history":    gnorm_vals,
            "misfit_initial":   float(arr[0]),
            "misfit_final":     float(arr[-1]),
            "misfit_reduction": float(reduction),
            "n_iterations":     int(arr.size),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Files that are read-only for toy2dac (never overwritten by a run).
_STATIC_NAMES = [
    # ── Acquisition (symlinked — large) ───────────────────────────────────
    "acqui",              # source/receiver geometry   (6.5 MB)
    # ── Binary model files (COPIED, not symlinked) ────────────────────────
    "vp_Marmousi_exact",  # true P-wave velocity       (~384 KB)  ← ROOT CAUSE if missing
    "rho",                # density model              (~384 KB)
    "qp",                 # quality factor             (~384 KB)
    "epsilon_m",          # Thomsen epsilon            (~384 KB)
    "delta_m",            # Thomsen delta              (~384 KB)
    "theta_m",            # tilt angle                 (~384 KB)
    "fbathy",             # bathymetry                 (~2.7 KB)
    "hess_sm",            # Hessian smoother           (~384 KB, optional)
    "hess_th",            # Hessian threshold          (~384 KB, optional)
    # ── Small text / config files (symlinked) ─────────────────────────────
    "mumps_input",        # MUMPS settings             (63 bytes)
    "data_weight_file",   # weight definitions         (6 bytes)
    "data_weight",        # weight values              (0 bytes placeholder)
    "fwi_input",          # FWI params + obs data name (1261 bytes)
]

# Binary model files read by Fortran DIRECT ACCESS — MUST be physical copies,
# not symlinks.  Reason: cross-filesystem symlinks
# /home NFS) cause Fortran to open a 0-byte placeholder file and then fail
# with "Non-existing record number" on record 1.
_BINARY_MODEL_FILES = {
    "vp_Marmousi_exact",   # true velocity model   (~384 KB)
    "rho",                 # density               (~384 KB)
    "qp",                  # quality factor        (~384 KB)
    "epsilon_m",           # Thomsen epsilon       (~384 KB)
    "delta_m",             # Thomsen delta         (~384 KB)
    "theta_m",             # tilt angle            (~384 KB)
    "hess_sm",             # Hessian smoother      (~384 KB, optional)
    "hess_th",             # Hessian threshold     (~384 KB, optional)
    "fbathy",              # bathymetry            (~2.7 KB)
}

# Files generated during a run that must NOT carry over between runs.
_MUTABLE_OUTPUTS = [
    "data_modeling", "data_cal", "data_cal_weight", "data_cal_weight_init",
    "data_obs_weight", "gradient", "gradient_prec", "fcost_data_reg",
    "norm_model", "hess_rw", "iterate_LB.dat", "iterate_PLB.dat",
    "param_vp_inter", "param_vp_final", "param_qp_inter", "param_qp_final",
    "param_rho_inter", "param_rho_final", "invparinter",
    "fort.10", "fort.14", "fort.15", "fort.58", "fort.59",
    "wavefield", "init.grd", "param_vp_final.grd",
    "vp_Marmousi_init",   # the model we write ourselves each group
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_floats(text: str) -> List[float]:
    """Return all floats found in a string (handles Fortran E-notation)."""
    import re
    pattern = r"[-+]?\d+\.?\d*(?:[eEdD][+-]?\d+)?"
    results = []
    for tok in re.findall(pattern, text):
        try:
            results.append(float(tok.replace("d", "e").replace("D", "E")))
        except ValueError:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(cfg: Toy2dacConfig) -> bool:
    """Quick pre-flight check before starting the BO loop."""
    checks = {
        "toy2dac binary":       Path(cfg.toy2dac_bin).exists(),
        "vp_Marmousi_exact":    Path(cfg.true_model_file).exists(),
        "vp_Marmousi_init":     Path(cfg.init_model_file).exists(),
        "template dir":         Path(cfg.base_data_dir).is_dir(),
        "acqui file":           Path(cfg.acq_file).exists(),
        "fwi_input":            (Path(cfg.base_data_dir) / "fwi_input").exists(),
        "freq_management":      (Path(cfg.base_data_dir) / "freq_management").exists(),
        "model size correct":   _check_model_size(cfg),
    }
    ok = True
    for name, passed in checks.items():
        icon = "✓" if passed else "✗"
        logger.info(f"  {icon}  {name}")
        if not passed:
            ok = False
    if ok:
        logger.info("  All checks passed — ready to run.")
    return ok


def _check_model_size(cfg: Toy2dacConfig) -> bool:
    expected = cfg.nx * cfg.nz * np.dtype(cfg.model_dtype).itemsize
    for f in (cfg.true_model_file, cfg.init_model_file):
        p = Path(f)
        if p.exists() and p.stat().st_size != expected:
            logger.error(
                f"  Size mismatch: {p.name} is {p.stat().st_size} bytes, "
                f"expected {expected} (nx={cfg.nx}×nz={cfg.nz}×4)"
            )
            return False
    return True
