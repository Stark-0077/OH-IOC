<h1 align="center">Online Optimal-Horizon Inverse Optimal Control</h1><p align="center">
  Tian Zhao · Ding Zhang · Tin Tran · Philipp Braun · Timothy L. Molloy
</p><p align="center">
  Australian National University · Monash University
</p><p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#numerical-examples">Numerical Examples</a> ·
  <a href="#install-and-run">Install and Run</a> ·
  <a href="#contact">Contact</a>
</p>

## Overview

How can we learn an optimal control objective from a demonstration when the task's duration is itself an unknown optimization variable?

This paper considers online optimal-horizon inverse optimal control (OH-IOC) for continuous-time systems. Given sequential, discrete-time measurements of optimal states and controls, the proposed extended Kalman filter (EKF) recursively estimates the cost-function parameters and the optimal horizon.

## Numerical Examples

The paper evaluates the method on two representive systems: double integator and unicycle. 

| System | Dynamics | State | Control | Unknown parameters |
| --- | --- | --- | --- | --- |
| Double integrator | Linear | $x=[q,\dot q]^\top\in\mathbb{R}^2$ | $u\in\mathbb{R}$ | $\theta=[\theta_u,\theta_q]^\top\in\mathbb{R}^2$ |
| Unicycle | Nonlinear | $x=[p_x,p_y,\phi]^\top\in\mathbb{R}^3$ | $u=[v,\omega]^\top\in\mathbb{R}^2$ | $\theta=[\theta_v,\theta_\omega,\theta_p,\theta_\phi]^\top\in\mathbb{R}^4$ |

The experiments examine parameter and horizon estimation errors across measurement types and noise levels, together with runtime per update.

## Install and Run

This folder reproduces the numerical results in Section V of the paper: Table I,
Fig. 1, and the perturbed-initial-state studies described in the text.

### Install

Python 3.8 or newer, and a LaTeX installation for Fig. 1. 

```bash
python3 -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt
```

### Run

With the environment active, from this folder:

```bash
python generate_table.py                                     # Table I
python generate_table.py table_perturbed_double_integrator   # Section V-A, perturbed initial states
python generate_table.py table_perturbed_unicycle            # Section V-B, perturbed initial states
python generate_fig_1.py                                     # Fig. 1
```

Each `generate_table.py` study writes `table.tex` (in the layout of Table I) and
`summary.txt` to `results/<study>/`, replacing the files there.
`generate_fig_1.py` writes the three plots of Fig. 1 to `../results/`. 
Other settings, such as the number of trials or worker processes, are at the top of each script.

### Files

| File | Contents |
| --- | --- |
| `ekf.py` | The EKF (4) and its Jacobian, computed by Algorithm 1 (Propositions 3.1 to 3.3) |
| `oh_ocp.py` | The optimal-horizon OCP (1), solved with Rockit and IPOPT |
| `forward_double_integrator.py`, `forward_unicycle.py` | The systems of Sections V-A and V-B, with their parameters |
| `generate_table.py` | The Monte Carlo studies |
| `generate_fig_1.py` | Fig. 1, which shows trial 28 of the Table I study at the lowest noise level |

Equation, proposition and algorithm numbers in the code refer to the paper.

## Contact

For questions about this work, please contact
[Tian Zhao](mailto:tian.zhao1@anu.edu.au).