from aerosandbox.optimization import Opti
from aerosandbox.geometry.airfoil import *
from aerosandbox import cas
import numpy as np
from numpy import pi
from singularities.linear_strength_vortex import calculate_induced_velocity
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))

### Set up givens

a = Airfoil("naca4412").repanel(n_points_per_side=100)
x = a.x()
y = a.y()
alpha_deg = 5

N = len(a.coordinates)  # number of airfoil nodes
# N_w = 50  # number of wake nodes

### Set up optimization framework and unknowns
opti = Opti()
gamma = opti.variable(
    n_vars=N,
    init_guess=0
)

### Calculate the field points
x_midpoints = (x[1:] + x[:-1]) / 2
y_midpoints = (y[1:] + y[:-1]) / 2

### Calculate the induced velocity at field points
u_field, v_field = calculate_induced_velocity(
    x_field=x_midpoints,
    y_field=y_midpoints,
    x_panel=x,
    y_panel=y,
    gamma=gamma,
    backend="casadi"
)

### Compute normal induced velocities
panel_dx = np.diff(x)
panel_dy = np.diff(y)
panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

yp_hat_x = -xp_hat_y
yp_hat_y = xp_hat_x

normal_induced_velocities = u_field * yp_hat_x + v_field * yp_hat_y

### Compute normal freestream velocities
alpha_rad = alpha_deg * pi / 180
u_freestream = np.cos(alpha_rad)
v_freestream = np.sin(alpha_rad)
normal_freestream_velocities = u_freestream * yp_hat_x + v_freestream * yp_hat_y

### Add in flow tangency constraint
opti.subject_to(normal_induced_velocities + normal_freestream_velocities == 0)

### Add in Kutta condition
opti.subject_to(gamma[0] + gamma[-1] == 0)

### Solve
sol = opti.solve()
gamma = sol.value(gamma)
u_field = sol.value(u_field)
v_field = sol.value(v_field)

### Plot flowfield
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=200)

plt.plot(x, y, "k")
plt.axis("equal")

margin = 0.4
res = 30
X, Y = np.meshgrid(
    np.linspace(-margin, 1 + margin, round(res * (1 + 2 * margin) / (2 * margin))),
    np.linspace(-margin, margin, res),
    indexing='ij',
)
X = X.flatten()
Y = Y.flatten()

U, V = calculate_induced_velocity(
    x_field=X,
    y_field=Y,
    x_panel=x,
    y_panel=y,
    gamma=gamma,
    backend="numpy"
)
U += u_freestream
V += v_freestream

plt.quiver(
    X, Y, U, V,
    (U ** 2 + V ** 2) ** 0.5,
    scale=30
)
plt.show()

### Plot C_p
u_field, v_field = calculate_induced_velocity(
    x_field=x_midpoints,
    y_field=y_midpoints,
    x_panel=x,
    y_panel=y,
    gamma=gamma,
)

Q = (
            (u_field + u_freestream) ** 2 +
            (v_field + v_freestream) ** 2
    ) ** 0.5
C_p = 1 - Q ** 2

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(x_midpoints, C_p)
plt.show()
