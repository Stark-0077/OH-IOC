"""Generate Table I and the perturbed-initial-state studies of Section V.

    python generate_table.py                                     # Table I
    python generate_table.py table_perturbed_double_integrator   # Section V-A, perturbed x0
    python generate_table.py table_perturbed_unicycle            # Section V-B, perturbed x0

Each study runs the EKF in every trial with state, control and combined
measurements at each noise level, and writes table.tex and summary.txt to
results/<study>/.  The other settings are at the top of this file.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ekf import run_ekf, sensitivities
from forward_double_integrator import SYSTEM as DOUBLE_INTEGRATOR
from forward_unicycle import SYSTEM as UNICYCLE
from oh_ocp import HORIZON_FLOOR


# %% Parameter selection
STUDIES = {  # name (and output folder): table rows, bound of the random initial-state offset
    "table": ([DOUBLE_INTEGRATOR, UNICYCLE], 0.0),  # Table I
    "table_perturbed_double_integrator": ([DOUBLE_INTEGRATOR], 0.25),  # Section V-A
    "table_perturbed_unicycle": ([UNICYCLE], 0.5),  # Section V-B
}
TRIALS = 50  # N
MEASUREMENTS = 25  # K
NOISE_LEVELS = [0.001, 0.005, 0.010]  # standard deviations; the bound b is sqrt(3) times larger
NOISE_BOUND_TEXT = [r"$\sqrt{3}$", r"$5\sqrt{3}$", r"$10\sqrt{3}$"]  # b in units of 1e-3, per level
MEASUREMENT_TYPES = ["States", "Controls", "Both"]
RADIUS = 0.6  # initial parameter error
WORKERS = 6
MACHINE = "an M3 Max MacBook Pro (Python 3.11, CasADi 3.7, IPOPT 3.14)"  # for the table footnote
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

# Each random quantity of a trial has its own generator, so that no draw depends
# on another: generate_fig_1.py can reproduce one trial without the others.
# (The two system seeds are 10_000 apart, so the double integrator's noise at
# level i + 1 reuses the unicycle's noise stream at level i.)
INITIAL_STATE_SEED_OFFSET = 900_000_000
INITIAL_ESTIMATE_SEED_OFFSET = 700_000_000
NOISE_SEED_STRIDE = 10_000  # noise at level i: seed + NOISE_SEED_STRIDE * i + trial


# %% One trial (also used by generate_fig_1.py)

def observation_matrix(model, measurement_type):
    """Matrix C_k selecting the states, the controls or both from col(x, u)."""
    n, m = model.state_dimension, model.control_dimension
    identity = np.eye(n + m)
    return {"States": identity[:n], "Controls": identity[n:], "Both": identity}[measurement_type]


def setup_trial(system, trial, perturbation):
    """Model, true solution, sample times, noise-free col(x, u) at those times, and theta_hat_0."""
    # Initial state, with a random offset if the perturbation is not zero.
    rng = np.random.default_rng(system.seed + trial + INITIAL_STATE_SEED_OFFSET)
    model = system.model(system.x0 + rng.uniform(-perturbation, perturbation, system.x0.size))

    # True trajectory at the evenly spaced sample times t_k = k T*/(K + 1).
    truth = model.solve(system.theta_true)
    times = np.arange(1, MEASUREMENTS + 1) / (MEASUREMENTS + 1) * truth.horizon
    samples = np.array([truth.xi(t) for t in times])

    # Initial estimate at distance RADIUS from the true parameters, in a random direction.
    rng = np.random.default_rng(system.seed + trial + INITIAL_ESTIMATE_SEED_OFFSET)
    direction = rng.normal(size=system.theta_true.size)
    theta_initial = system.theta_true + RADIUS * (direction / np.linalg.norm(direction))
    return model, truth, times, samples, theta_initial


def add_noise(samples, system, trial, noise_index):
    """Samples plus noise drawn uniformly from [-b, b] for every state and control.

    The measurement types of a trial therefore share the noise on the
    quantities they have in common.
    """
    bound = np.sqrt(3.0) * NOISE_LEVELS[noise_index]
    rng = np.random.default_rng(system.seed + NOISE_SEED_STRIDE * noise_index + trial)
    return samples + rng.uniform(-bound, bound, samples.shape)


def run_trial(system, trial, perturbation):
    """Run the EKF for every noise level and measurement type of one trial."""
    model, truth, times, samples, theta_initial = setup_trial(system, trial, perturbation)
    parameter_error = np.zeros((len(NOISE_LEVELS), len(MEASUREMENT_TYPES)))
    horizon_error = np.zeros_like(parameter_error)
    final_horizon_at_floor = np.zeros(parameter_error.shape, dtype=bool)
    runtime = np.zeros_like(parameter_error)
    update_derivatives = []
    for i, noise_sd in enumerate(NOISE_LEVELS):
        noisy_samples = add_noise(samples, system, trial, i)
        for j, measurement_type in enumerate(MEASUREMENT_TYPES):
            C = observation_matrix(model, measurement_type)
            estimates, runtimes, derivatives = run_ekf(
                model, C, noisy_samples @ C.T, times, theta_initial, system.P0, noise_sd**2
            )
            final = model.solve(estimates[-1])  # T*(theta_hat_K), without the bound of Algorithm 1
            parameter_error[i, j] = np.linalg.norm(estimates[-1] - system.theta_true)
            horizon_error[i, j] = abs(final.horizon - truth.horizon)
            final_horizon_at_floor[i, j] = final.bound_active
            runtime[i, j] = runtimes.mean()
            update_derivatives.append(derivatives)
    update_derivatives = np.concatenate(update_derivatives)

    return {
        "parameter_error": parameter_error,
        "horizon_error": horizon_error,
        "final_horizon_at_floor": final_horizon_at_floor,
        "runtime": runtime,
        "true_horizon": truth.horizon,
        # Equation (23) at the true parameters.
        "horizon_derivative": sensitivities(model, system.theta_true, truth, truth.horizon)[-1],
        # Smallest |(23)| over the updates (NaN entries: horizon bound active).
        "smallest_update_derivative": np.nanmin(np.abs(update_derivatives)),
        "bound_active_updates": np.isnan(update_derivatives).sum(),
    }


# %% Summary and LaTeX table (the layout of Table I in main.tex)

TABLE_HEADER = r"""\begin{table*}[!t]
\centering
\caption{Simulation study results for three measurement types, using $K=<K>$ measurements, and $<N>$ trials for each type, noise level, and system.<PERTURBATION>
Errors are mean $\pm$ standard deviation, with best mean errors over measurement type bolded.
}
\label{<LABEL>}
\footnotesize
\setlength{\tabcolsep}{2.7pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cccccccccc@{}}
\toprule
\multirow{3}{*}[-0.5ex]{System}
& \multirow{3}{*}[-0.5ex]{\shortstack{Noise bound\\$b$ $(\times10^{-3})$}}
& \multicolumn{3}{c}{Final Parameter Error $\|\hat\theta_K-\theta^\circ\|$}
& \multicolumn{3}{c}{Final Horizon Error $|T^\star(\hat\theta_K)-T^\star(\theta^\circ)|$}
& \multirow{3}{*}[-0.5ex]{\shortstack{Minimum\\$\smash[b]{|\partial h/\partial T|}$ in \eqref{eq:terminal_hamiltonian_nondegeneracy}\\over Updates}}
& \multirow{3}{*}[-0.5ex]{\shortstack{Average\\Runtime/\\Update (s)}}
\\
&
& \multicolumn{3}{c}{with Measurements of:}
& \multicolumn{3}{c}{with Measurements of:}
&
&
\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
& & States & Controls & Both & States & Controls & Both & & \\
\midrule
"""

TABLE_FOOTER = r"""\bottomrule
\end{tabular*}
\vspace{1mm}
\begin{minipage}{\textwidth}
  \vspace{1mm}
\footnotesize
Runtimes on <MACHINE>; time between measurements: <SPACINGS>.
\end{minipage}
\vspace{-8mm}
\end{table*}
"""


def summary_text(system, results, perturbation):
    """Plain-text statistics of one system."""
    horizons, derivatives = results["true_horizon"], results["horizon_derivative"]
    parameter_errors = results["parameter_error"]
    lines = [
        f"{system.name}: {TRIALS} trials, {MEASUREMENTS} measurements, "
        f"initial-state offset bound {perturbation}",
        f"  true horizon in [{horizons.min():.3f}, {horizons.max():.3f}] s, "
        f"dh/dT at the true parameters in [{derivatives.min():.4f}, {derivatives.max():.4f}]",
        f"  smallest |dh/dT| over EKF updates {results['smallest_update_derivative'].min():.4f}, "
        f"updates with the horizon bound active {results['bound_active_updates'].sum()}",
        f"  runs whose final horizon is the solver floor ({HORIZON_FLOOR} s) "
        f"{results['final_horizon_at_floor'].sum()}, runs whose final parameter error exceeds "
        f"the initial {RADIUS} {(parameter_errors > RADIUS).sum()}",
    ]
    for i, noise_sd in enumerate(NOISE_LEVELS):
        for j, measurement_type in enumerate(MEASUREMENT_TYPES):
            parameter = parameter_errors[:, i, j]
            horizon = results["horizon_error"][:, i, j]
            lines.append(
                f"  sigma={noise_sd:.3f} {measurement_type:>8}: "
                f"parameter error {parameter.mean():.3f} +/- {parameter.std(ddof=1):.3f}, "
                f"horizon error {horizon.mean():.3f} +/- {horizon.std(ddof=1):.3f} s, "
                f"runtime/update {results['runtime'][:, i, j].mean():.4f} s"
            )
    return "\n".join(lines) + "\n"


def table_rows(system, results):
    """LaTeX rows of one system, one per noise level."""
    name = system.name
    if " " in name:  # one word per line
        name = r"\shortstack{" + name.replace(" ", r"\\") + "}"
    derivative_text = f"${results['smallest_update_derivative'].min():.4f}$"

    lines = [rf"\multirow{{3}}{{*}}{{{name}}}"]
    for i, bound_text in enumerate(NOISE_BOUND_TEXT):
        lines.append(f"& {bound_text}")
        for errors in (results["parameter_error"][:, i], results["horizon_error"][:, i]):
            means = [f"{mean:.3f}" for mean in errors.mean(axis=0)]
            deviations = [f"{deviation:.3f}" for deviation in errors.std(axis=0, ddof=1)]
            smallest = min(means, key=float)
            for mean, deviation in zip(means, deviations):
                if mean == smallest:  # smallest mean in bold
                    mean = rf"\mathbf{{{mean}}}"
                lines.append(f"& ${mean} \\pm {deviation}$")
        lines.append(rf"& \multirow{{3}}{{*}}{{{derivative_text}}}" if i == 0 else "&")
        lines.append(f"& ${results['runtime'][:, i].mean():.4f}$ " + r"\\")
    return "\n".join(lines) + "\n"


def spacing_text(system, results):
    """Time between measurements, T*/(K + 1), for the table footnote."""
    spacing = results["true_horizon"] / (MEASUREMENTS + 1)
    low, high = f"{spacing.min():.3f}", f"{spacing.max():.3f}"
    value = low if low == high else f"{low}$--${high}"
    return rf"${value}\,\mathrm{{s}}$ ({system.name.lower()})"


# %% Main

def main(study):
    systems, perturbation = STUDIES[study]
    assert len(NOISE_BOUND_TEXT) == len(NOISE_LEVELS), "give one NOISE_BOUND_TEXT per noise level"
    output_directory = RESULTS_DIRECTORY / study
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries, rows, spacings = [], [], []
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # for the workers: idle BLAS threads slow them down
    with ProcessPoolExecutor(WORKERS) as pool:
        for system in systems:
            trials = list(pool.map(run_trial, [system] * TRIALS, range(TRIALS), [perturbation] * TRIALS))
            results = {key: np.array([trial[key] for trial in trials]) for key in trials[0]}
            summaries.append(summary_text(system, results, perturbation))
            rows.append(table_rows(system, results))
            spacings.append(spacing_text(system, results))
            print(summaries[-1], flush=True)

    caption = "" if perturbation == 0 else (
        rf" Initial states perturbed by uniform offsets in $[-{perturbation},{perturbation}]^n$."
    )
    label = "tab:combined_initialization_noise_measurement" if study == "table" else f"tab:{study}"
    header = (
        TABLE_HEADER.replace("<K>", str(MEASUREMENTS))
        .replace("<N>", str(TRIALS))
        .replace("<PERTURBATION>", caption)
        .replace("<LABEL>", label)
    )
    footer = TABLE_FOOTER.replace("<MACHINE>", MACHINE).replace("<SPACINGS>", ", ".join(spacings))
    (output_directory / "table.tex").write_text(header + "\\midrule\n".join(rows) + footer)
    (output_directory / "summary.txt").write_text("\n".join(summaries))
    print(f"Saved table.tex and summary.txt in {output_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", default="table", choices=STUDIES)
    main(parser.parse_args().study)
