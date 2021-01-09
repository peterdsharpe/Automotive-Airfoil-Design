from aerosandbox.optimization import Opti
from aerosandbox.geometry.airfoil import *
import casadi as cas
import numpy as np
from numpy import pi

### Set up givens

a = Airfoil("naca4412")
x = a.x()
y = a.y()
alpha_deg = 0
u = cas.cos(pi / 180 * alpha_deg)  # x-velocity
v = cas.sin(pi / 180 * alpha_deg)  # y-velocity
N = len(a.coordinates)  # number of airfoil nodes
N_w = 50  # number of wake nodes

### Compute local panel coordinates
x_field = np.expand_dims(x, 1)
y_field = np.expand_dims(y, 1)
x_panel = x
y_panel = y

r_1 = (
              (x_field - x_panel) ** 2 +
              (y_field - y_panel) ** 2
      ) ** 0.5
theta_1 = np.arctan2(
    x_field - x_panel,
    y_field - y_panel,
)
dx_panel = np.roll(x_panel, -1) - x_panel
dy_panel = np.roll(y_panel, -1) - y_panel
panel_rotation_matrix =


### Initalize an optimization environment
opti = Opti()

### Define unknowns
gamma = opti.variable(n_vars=N, init_guess=0)
sigma = 0  # opti.variable(n_vars=n_points, init_guess=0)


### Compute streamfunction
def psi(x, y):
    def psi_gamma_plus(j):
        return (
                x_bar_1 * cas.log(r_1) -
                x_bar_2 * cas.log(r_2) +
                x_bar_2 - x_bar_1 +
                y_bar * (theta_1 - theta_2)
        )

    def psi_gamma_minus(j):
        return (
                (x_bar_1 + x_bar_2) * psi_gamma_plus(j)
        )

    # vortex_integral =

    psi_val = (
            u * y + v * x
            + 1 / (4 * pi) * (vortex_integral + source_integral)
    )
    return psi_val
