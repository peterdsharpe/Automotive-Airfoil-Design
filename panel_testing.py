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

N = len(a.coordinates)  # number of airfoil nodes
N_w = 50  # number of wake nodes

### Set up optimization framework and unknowns
opti = Opti()
gamma = opti.variable(
    n_vars=N,
    init_guess=0
)

### Compute local panel coordinates, using nomenclature from
# Katz and Plotkin, "Low Speed Aerodynamics", Fig. 11.29: Nomenclature for a linear-strength surface singularity element.
x_panel = x
y_panel = y
x_field = np.expand_dims(
    (x_panel[1:] + x_panel[:-1]) / 2,
    1
)
y_field = np.expand_dims(
    (y_panel[1:] + y_panel[:-1]) / 2,
    1
)

# Compute r vector
r_vector = np.dstack((  # i-th field point, j-th panel point, k index is [x, y].
    x_field - x_panel,
    y_field - y_panel
))

# Compute local coordinate system
panel_vector = np.vstack((
    np.roll(x_panel, -1) - x_panel,
    np.roll(y_panel, -1) - y_panel
)).T  # The [x, y] vector from the i-th panel point to the next.
xp_panel = np.linalg.norm(panel_vector, axis=1)  # The distance from the i-th panel point to the next
xp_hat = panel_vector / np.expand_dims(xp_panel,
                                       1)  # The local xp-axis direction for panel i.
zp_hat = np.vstack((  # The local zp-axis direction for panel i.
    -xp_hat[:, 1],
    xp_hat[:, 0]
)).T

# Compute local coordinates of interest
xp_field = np.einsum(  # xp-coordinate of the i-th field point in the ref. frame of the j-th panel
    'ijk,jk->ij',
    r_vector, xp_hat
)
zp_field = np.einsum(  # zp-coordinate of the i-th field point in the ref. frame of the j-th panel
    'ijk,jk->ij',
    r_vector, zp_hat
)
r = (xp_field ** 2 + zp_field ** 2) ** 0.5  # distance from the i-th field point to the j-th panel node
theta = np.arctan2(zp_field, xp_field)  # Angle from the tangent line of the j-th panel to the i-th field point

# Calculate differential ("q_{j+1} - q_j" for some variable q) quantities
# dxp_field = np.diff(xp_field)
# dzp_field = np.diff(zp_field)
dxp_panel = np.diff(xp_panel)
dgamma = cas.diff(gamma)

### Calculate the velocity influence matrix
A = np.zeros((N,
              N))  # The influence of the j-th panel node vortex strength on the normal velocity at the i-th panel. The last row is the Kutta condition.

for i in range(N - 1):
    for j in range(N - 1):

        ### Add the influences of the first term of Eq. 11.100 in Katz & Plotkin
        A[i, j] -= (
                           xp_panel[j] - xp_field[i, j]
                   ) / (2 * pi * xp_panel[j]) * np.log(r[i, j] / r[i, j + 1])
        A[i, j + 1] -= (
                           xp_field[i, j]
                       ) / (2 * pi * xp_panel[j]) * np.log(r[i, j] / r[i, j + 1])

        ### Add the influences of the second term of Eq. 11.100 in Katz & Plotkin
        if i != j:
            second_term = (
                                  zp_field[i, j] / (2 * pi) / xp_panel[j]
                          ) * (xp_panel[j] / zp_field[i, j] + (theta[i, j + 1] - theta[i, j]))
        else:
            second_term = 1 / (2 * pi)

        A[i, j + 1] += second_term
        A[i, j] -= second_term

# Add in Kutta condition
A[-1, 0] = 1
A[-1, -1] = 1

### Form the RHS
alpha_rad = pi / 180 * alpha_deg
U = np.array([
    cas.cos(alpha_rad),
    cas.sin(alpha_rad)
])  # Freestream velocity direction
RHS = np.einsum(
    "j,ij->i",
    U,
    zp_hat,
)
# Complete the Kutta condition
RHS[-1] = 0

### Solve the linear system
gamma = np.linalg.solve(A, RHS)

# Calculate local induced velocities
# for i in range(N):  # For the i-th field point
#     for j in range(N - 1):  # For the j-th panel
#         u_p = 0
#         u_p += zp_field[i, j] / (2 * pi) * (
#                 dgamma[j] / dxp_panel[j]
#         ) * np.log(r[i, j + 1] / r[i, j])
#         u_p += (
#                 gamma[j] * dxp_field[i, j] + dgamma * (zp_field)
#         )

### Drela notation
# x_field = np.expand_dims(x, 1)
# y_field = np.expand_dims(y, 1)
# x_panel = x
# y_panel = y
#
# r_vector = np.dstack(( # i-th field point, j-th panel point, k index is [x, y].
#     x_field - x_panel,
#     y_field - y_panel
# ))
#
# dx_panel = np.roll(x_panel, -1) - x_panel
# dy_panel = np.roll(y_panel, -1) - y_panel
# panel_vector = np.vstack((dx_panel, dy_panel)).T # The [x, y] vector from the i-th panel point to the next.
# panel_unit_vector = panel_vector / np.expand_dims(np.linalg.norm(panel_vector, axis=1), 1)
# panel_unit_vector_normal = np.vstack(( # The panel_vector, rotated clockwise 90 degrees.
#     panel_unit_vector[:,1],
#     -panel_unit_vector[:,0]
# )).T
#
# x_bar_1 = np.einsum(
#     'ijk,jk->ij',
#     r_vector, panel_unit_vector
# )
# y_bar_1 = np.einsum(
#     'ijk,jk->ij',
#     r_vector, panel_unit_vector_normal
# )
# r_1 = (
#               x_bar_1 ** 2 +
#               y_bar_1 ** 2
#       ) ** 0.5
#
# theta_1 = np.arctan2(
#     x_field - x_panel,
#     y_field - y_panel,
# )


# ### Initialize an optimization environment
# opti = Opti()
#
# ### Define unknowns
# gamma = opti.variable(n_vars=N, init_guess=0)
# sigma = 0  # opti.variable(n_vars=n_points, init_guess=0)
#
#
# ### Compute streamfunction
# def psi(x, y):
#     def psi_gamma_plus(j):
#         return (
#                 x_bar_1 * cas.log(r_1) -
#                 x_bar_2 * cas.log(r_2) +
#                 x_bar_2 - x_bar_1 +
#                 y_bar * (theta_1 - theta_2)
#         )
#
#     def psi_gamma_minus(j):
#         return (
#                 (x_bar_1 + x_bar_2) * psi_gamma_plus(j)
#         )
#
#     # vortex_integral =
#
#     psi_val = (
#             u * y + v * x
#             + 1 / (4 * pi) * (vortex_integral + source_integral)
#     )
#     return psi_val
