# NOVA: 6-DoF Aerospace Flight Simulation Engine

NOVA is a high-fidelity, deterministic 6-Degree-of-Freedom (6-DoF) aerospace and astrodynamics simulation engine. Designed with a modular, multi-stage pipeline, it facilitates the simulation of complex launch trajectories, orbital transfers, and structural load analysis.

## System Architecture
NOVA utilizes a strict 13-stage deterministic pipeline. Each stage is immutable, ensuring that the simulation results are reproducible and independent of system time.

*   **Physics Core:** RK4 numerical integrator evolving a 13-dimensional state vector.
*   **Kinematics:** Quaternion-based attitude representation ($\mathbb{H}$) to avoid gimbal lock.
*   **Guidance & Control:** PID-based SAS rate-damping and control surface deflection modeling.
*   **Structural Analysis:** Multibody component graph evaluating Euler buckling and shear yield margins.

### Mathematical Baseline
The simulation evolves the vehicle state $\mathbf{y}(t)$ using the following dynamics:

Translational:

$$\dot{\mathbf{v}}_{\text{ECI}} = \frac{\mathbf{F}_{\text{net}}}{m}$$

Rotational:

$$\dot{\boldsymbol{\omega}}_{\text{body}} = \mathbf{I}^{-1} (\boldsymbol{\tau}_{\text{net}} - \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}))$$

Kinematic:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \boldsymbol{\omega}_{\text{body}}$$

## Release & Version Roadmap
- [x] **v1.0.0** — Core deterministic 6-DoF engine, RK4 13-stage pipeline, N-body astrodynamics, glass-cockpit HUD, and validation suite. *(Current Baseline)*
- [ ] **v2.0.0** — Lie Group $SO(3)$ manifold rotational integration.
- [ ] **v3.0.0** — Euler-Bernoulli modal bending and aeroelastic coupling.
- [ ] **v4.0.0** — Full Aero-Structural-Control autonomous GNC engine (EKF, Trajectory Optimizer, and MPC).

## Installation
To clone the repository and begin execution:

```bash
git clone [https://github.com/your-username/Project-NOVA.git](https://github.com/your-username/Project-NOVA.git)
cd Project-NOVA
pip install numpy pygame pytest
python3 main.py --mode sim --t_max 600