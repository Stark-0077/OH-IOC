"""Generate Fig. 1: measurements, parameter error and horizon estimate of one trial.

Runs one trial of the Table I study of the double integrator and writes one
PDF per panel to the manuscript's results/ folder.  Matplotlib typesets the
figure text with LaTeX, so a LaTeX installation is needed.
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ekf import run_ekf
from forward_double_integrator import SYSTEM
from generate_table import MEASUREMENT_TYPES, NOISE_LEVELS, add_noise, observation_matrix, setup_trial


# %% Parameter selection
TRIAL = 28  # index of the Table I trial
NOISE_INDEX = 0  # lowest noise level
# Stem of the three panel files read by main.tex.
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "fig1_double_integrator"

COLORS = {"States": "#0072B2", "Controls": "#D55E00", "Both": "#009E73"}
MARKERS = {"States": "o", "Controls": "s", "Both": "^"}
FIGURE_STYLE = {
    # At least 8 pt on the page.
    "font.family": "serif",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 2.6,
    "lines.markeredgewidth": 0,
    "text.usetex": True,
}
LEGEND_STYLE = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 1.0,
    "columnspacing": 1.0,
    "handlelength": 1.5,
    "handletextpad": 0.5,
    "borderpad": 0.3,
}


# %% The trial

def run_trial():
    """Run the EKF for each measurement type and return what the panels show."""
    model, truth, times, samples, theta_initial = setup_trial(SYSTEM, TRIAL, perturbation=0.0)
    noise_sd = NOISE_LEVELS[NOISE_INDEX]
    noisy_samples = add_noise(samples, SYSTEM, TRIAL, NOISE_INDEX)

    errors, horizons = {}, {}
    for measurement_type in MEASUREMENT_TYPES:
        C = observation_matrix(model, measurement_type)
        estimates, _, _ = run_ekf(
            model, C, noisy_samples @ C.T, times, theta_initial, SYSTEM.P0, noise_sd**2
        )
        errors[measurement_type] = np.linalg.norm(estimates - SYSTEM.theta_true, axis=1)
        horizons[measurement_type] = [model.solve(theta).horizon for theta in estimates]
    return times, noisy_samples, truth.horizon, errors, horizons


# %% Plotting

def new_panel(height, bottom, top=0.97):
    """Empty panel; the side margins are shared so that the panels align."""
    figure, axis = plt.subplots(figsize=(3.5, height))
    figure.subplots_adjust(left=0.19, right=0.985, bottom=bottom, top=top)
    axis.grid(True, alpha=0.22, linewidth=0.45)
    return figure, axis


def main():
    if shutil.which("latex") is None:  # check before the EKF runs and before any file is written
        sys.exit("generate_fig_1.py needs LaTeX, but no latex command was found.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    times, samples, true_horizon, errors, horizons = run_trial()
    update_times = np.concatenate(([0.0], times))  # the initial estimate is at time zero
    plt.rcParams.update(FIGURE_STYLE)

    # (a) Measurements, with the states and controls in the colours used below.
    figure, axis = new_panel(height=0.8, bottom=0.08)
    signals = [(r"$q$", "States", "-"), (r"$\dot{q}$", "States", ":"), (r"$u$", "Controls", "-")]
    for index, (label, group, linestyle) in enumerate(signals):
        axis.plot(
            times,
            samples[:, index],
            label=label,
            color=COLORS[group],
            marker=MARKERS[group],
            linestyle=linestyle,
        )
    axis.set_xlim(0.0, times[-1])
    axis.set_ylim(-1.3, 2.6)
    axis.set_yticks([-1.0, 0.0, 1.0, 2.0])
    axis.set_ylabel(r"$y_k$", labelpad=2)
    axis.tick_params(axis="x", labelbottom=False)
    axis.legend(loc="upper right", ncol=3, **{**LEGEND_STYLE, "handlelength": 2.2})
    figure.savefig(f"{OUTPUT}_a.pdf")

    # (b) Parameter error.
    figure, axis = new_panel(height=0.8, bottom=0.08, top=0.92)
    for measurement_type in MEASUREMENT_TYPES:
        axis.semilogy(
            update_times,
            errors[measurement_type],
            label=measurement_type,
            color=COLORS[measurement_type],
            marker=MARKERS[measurement_type],
            markevery=slice(1, None),
        )
    axis.set_xlim(0.0, times[-1])
    axis.set_ylim(1e-3, 1e1)
    axis.set_yticks([1e-3, 1e-1, 1e1])
    axis.set_ylabel(r"$\|\theta^\circ-\hat{\theta}_k\|$", labelpad=2)
    axis.tick_params(axis="x", labelbottom=False)
    axis.legend(loc="upper right", ncol=3, **LEGEND_STYLE)
    figure.savefig(f"{OUTPUT}_b.pdf")

    # (c) Horizon estimate; the legend of the measurement types is in (b).
    figure, axis = new_panel(height=1.0, bottom=0.33)
    for measurement_type in MEASUREMENT_TYPES:
        axis.plot(
            update_times,
            horizons[measurement_type],
            color=COLORS[measurement_type],
            marker=MARKERS[measurement_type],
            markevery=slice(1, None),
        )
    reference = axis.axhline(
        true_horizon, color="black", linestyle="--", linewidth=1.0, label=r"$T^\star(\theta^\circ)$"
    )
    axis.set_xlim(0.0, times[-1])
    axis.set_ylim(2.65, 3.05)
    axis.set_yticks([2.7, 2.8, 2.9, 3.0])
    axis.set_ylabel(r"$T^\star(\hat{\theta}_k)$ (s)", labelpad=2)
    axis.set_xlabel(r"Time, $t$ (s)", labelpad=1)
    axis.legend(handles=[reference], loc="upper right", **LEGEND_STYLE)
    figure.savefig(f"{OUTPUT}_c.pdf")

    for measurement_type in MEASUREMENT_TYPES:
        print(
            f"{measurement_type}: final parameter error {errors[measurement_type][-1]:.3f}, "
            f"final horizon error {abs(horizons[measurement_type][-1] - true_horizon):.3f} s"
        )
    print(f"Saved {OUTPUT.name}_a.pdf, _b.pdf and _c.pdf in {OUTPUT.parent}")


if __name__ == "__main__":
    main()
