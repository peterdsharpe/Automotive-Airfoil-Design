from linear_strength_vortex import *

X, Y = np.meshgrid(
    np.linspace(-1, 2, 50),
    np.linspace(-1, 1, 50),
    indexing='ij',
)
X = X.flatten()
Y = Y.flatten()

U, V = calculate_induced_velocity(
    x_field=X,
    y_field=Y,
    x_panel_start=0.5,
    y_panel_start=0,
    x_panel_end=1,
    y_panel_end=1,
    backend="casadi"
)
