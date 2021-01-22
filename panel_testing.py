from aerosandbox.optimization import Opti
from aerosandbox.geometry.airfoil import *
from aerosandbox import cas
import numpy as np
from numpy import pi
from singularities.linear_strength_vortex import calculate_induced_velocity

### Set up givens

a = Airfoil("naca4412").repanel(n_points_per_side=50)
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
induced_velocity_at_field_point_x = cas.GenDM_zeros(N)
induced_velocity_at_field_point_y = cas.GenDM_zeros(N)

for i in range(N-1): # for each panel i
    u, v = calculate_induced_velocity(
        x_field=x_midpoints,
        y_field=y_midpoints,
        x_panel_start=x[i],
        y_panel_start=y[i],
        x_panel_end=x[i+1],
        y_panel_end=y[i+1],
        gamma_start=gamma[i],
        gamma_end=gamma[i+1],
        backend="casadi"
    )



# # # Add in Kutta condition
# # A[-1, 0] = 1
# # A[-1, -1] = 1
# 
# ### Form the RHS
# alpha_rad = pi / 180 * alpha_deg
# Q = np.array([
#     cas.cos(alpha_rad),
#     cas.sin(alpha_rad)
# ])  # Freestream velocity direction
# RHS = -np.einsum(
#     "j,ij->i",
#     Q,
#     zp_hat,
# )
# # Complete the Kutta condition
# RHS[-1] = 0
# 
# ### Solve the linear system
# gamma = np.linalg.solve(A, RHS)
# 
# ### Perturbation velocity calculation
# Q_t = 1 + (gamma[1:] + gamma[:-1]) / 4
# C_p = 1 - Q_t ** 2
# 
# ### Plot C_p
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# sns.set(palette=sns.color_palette("husl"))
# fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
# plt.plot(x_field, C_p)
# plt.show()
