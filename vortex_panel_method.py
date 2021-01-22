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

a = Airfoil("dae11").repanel(n_points_per_side=50)
x_panel = a.x()
y_panel = a.y()
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
x_midpoints = (x_panel[1:] + x_panel[:-1]) / 2
y_midpoints = (y_panel[1:] + y_panel[:-1]) / 2

### Compute freestream velocities
alpha_rad = alpha_deg * pi / 180
u_freestream = cas.cos(alpha_rad)
v_freestream = cas.sin(alpha_rad)


### Calculate the local velocity at field points
def calculate_velocity(
        x_field,
        y_field,
        gamma,
        backend="numpy"
):
    u_field_induced, v_field_induced = calculate_induced_velocity(
        x_field=x_field,
        y_field=y_field,
        x_panel=x_panel,
        y_panel=y_panel,
        gamma=gamma,
        backend=backend
    )
    u_field = u_field_induced + u_freestream
    v_field = v_field_induced + v_freestream

    return u_field, v_field


u_field, v_field = calculate_velocity(
    x_field=x_midpoints,
    y_field=y_midpoints,
    gamma=gamma,
    backend="casadi"
)

### Compute normal velocities
panel_dx = np.diff(x_panel)
panel_dy = np.diff(y_panel)
panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

yp_hat_x = -xp_hat_y
yp_hat_y = xp_hat_x

normal_velocities = u_field * yp_hat_x + v_field * yp_hat_y

### Add in flow tangency constraint
opti.subject_to(normal_velocities == 0)

### Add in Kutta condition
opti.subject_to(gamma[0] + gamma[-1] == 0)

### Solve
sol = opti.solve()
gamma = sol.value(gamma)
u_field = sol.value(u_field)
v_field = sol.value(v_field)

### Calculate lift coefficient
total_vorticity = np.sum(
    (gamma[1:] + gamma[:-1]) / 2 *
    panel_length
)
Cl = 2 * total_vorticity
print(f"Cl: {Cl}")

### Plot flowfield
fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=200)

margin = 0.4
res = 30
x = np.linspace(-margin, 1 + margin, res)  # round(res * (1 + 2 * margin) / (2 * margin)))
y = np.linspace(-margin, margin, res)
X, Y = np.meshgrid(
    x,
    y,
    # indexing='ij',
)
X = X.flatten()
Y = Y.flatten()

U, V = calculate_velocity(
    x_field=X,
    y_field=Y,
    gamma=gamma,
)
speed = (U ** 2 + V ** 2) ** 0.5
# plt.quiver(
#     X, Y, U, V, speed,
#     scale=30
# )
plt.streamplot(
    x,
    y,
    U.reshape(len(y), len(x)),
    V.reshape(len(y), len(x)),
    color=speed.reshape(len(y), len(x)),
    density=1,
)

plt.fill(x_panel, y_panel, "k", linewidth=0, zorder=4)

plt.axis("equal")

plt.show()

### Plotly
# import plotly.figure_factory as ff
# fig = ff.create_streamline(
#     x,
#     y,
#     U.reshape(len(y), len(x)),
#     V.reshape(len(y), len(x)),
#     # color=speed.reshape(len(y), len(x))
# )
# fig.show()

### Plot C_p
u_field, v_field = calculate_velocity(
    x_field=x_midpoints,
    y_field=y_midpoints,
    gamma=gamma,
)

Q = (
            u_field ** 2 +
            v_field ** 2
    ) ** 0.5
C_p = 1 - Q ** 2

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(x_midpoints, C_p)
plt.show()
