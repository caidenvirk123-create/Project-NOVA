# Kinematics & Dynamics Derivation

## 1. The State Vector
The vehicle state $\mathbf{x} \in \mathbb{R}^{13}$ is defined in the ECI (Earth-Centered Inertial) frame for translation and the Body frame for rotation:

```math
\mathbf{x} = \begin{bmatrix} x & y & z & v_x & v_y & v_z & q_w & q_x & q_y & q_z & \omega_x & \omega_y & \omega_z \end{bmatrix}^T
```

## 2. Attitude Propagation
We use quaternions to represent orientation. The unit-norm constraint $\|\mathbf{q}\| = 1$ is maintained via explicit renormalization after every RK4 integration step to prevent secular drift.

The kinematic derivative is:

```math
\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Xi}(\mathbf{q}) \boldsymbol{\omega}
```

where $\boldsymbol{\Xi}(\mathbf{q})$ is the skew-symmetric matrix transforming angular velocity to quaternion rate.

## 3. Numerical Integration
We employ a 4th-order Runge-Kutta (RK4) integration scheme. Given $\dot{\mathbf{x}} = f(t, \mathbf{x})$, we compute:

```math
\mathbf{k}_1 = f(t_n, \mathbf{x}_n)
```

```math
\mathbf{k}_2 = f\left(t_n + \frac{\Delta t}{2}, \mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_1\right)
```

```math
\mathbf{k}_3 = f\left(t_n + \frac{\Delta t}{2}, \mathbf{x}_n + \frac{\Delta t}{2}\mathbf{k}_2\right)
```

```math
\mathbf{k}_4 = f(t_n + \Delta t, \mathbf{x}_n + \Delta t \mathbf{k}_3)
```

```math
\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
```

Global truncation error is $\mathcal{O}(\Delta t^4)$, ensuring energy conservation properties are maintained over long orbital duration simulations.