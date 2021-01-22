from numpy import pi
import numpy as np
from aerosandbox import cas
from typing import Union


def calculate_induced_velocity_single_panel_panel_coordinates(
        xp_field: Union[float, np.ndarray],
        yp_field: Union[float, np.ndarray],
        gamma_start: float = 1.,
        gamma_end: float = 1.,
        xp_panel_end: float = 1.,
        backend: str = "numpy",
) -> [Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point (xp_field, yp_field) in a 2D potential-flow flowfield.

    The `p` suffix in `xp...` and `yp...` denotes the use of the panel coordinate system, where:
        * xp_hat is along the length of the panel
        * yp_hat is orthogonal (90 deg. counterclockwise) to it.

    In this flowfield, there is only one singularity element: A line vortex going from (0, 0) to (xp_panel_end, 0).
    The strength of this vortex varies linearly from:
        * gamma_start at (0, 0), to:
        * gamma_end at (xp_panel_end, 0).

    By convention here, positive gamma induces clockwise swirl in the flow field.
        
    Function returns the 2D velocity u, v in the local coordinate system of the panel.

    Inputs x and y can be 1D ndarrays representing various field points,
    in which case the resulting velocities u and v have corresponding dimensionality.

    The `backend` parameter selects a numerical backend to be used, with the options "numpy" and "casadi".
    NumPy is significantly faster, but CasADi is differentiable.

    Equations from the seminal textbook "Low Speed Aerodynamics" by Katz and Plotkin.
    Equations 11.99 and 11.100.
        * Note: there is an error in equation 11.100 in Katz and Plotkin, at least in the 2nd ed:
        The last term of equation 11.100, which is given as:
            (x_{j+1} - x_j) / z + (theta_{j+1} - theta_j)
        has a sign error and should instead be written as:
            (x_{j+1} - x_j) / z - (theta_{j+1} - theta_j)

    """
    ### Modify any incoming floats
    if isinstance(xp_field, (float, int)):
        xp_field = np.array([xp_field])
    if isinstance(yp_field, (float, int)):
        yp_field = np.array([yp_field])

    ### Define functions according to the backend to be used, and validate backend argument
    if backend == "numpy":
        arctan2 = lambda y, x: np.arctan2(y, x)
        ln = lambda x: np.log(x)
        abs = lambda x: np.abs(x)
    elif backend == "casadi":
        arctan2 = lambda y, x: cas.arctan2(y, x)
        ln = lambda x: cas.log(x)
        abs = lambda x: np.fabs(x)
    else:
        raise ValueError("Bad value of 'backend'!")

    ### Determine which points are effectively on the panel, necessitating different math:
    is_on_panel = abs(yp_field) <= 1e-8

    ### Do some geometry calculation
    r_1 = (
                  xp_field ** 2 +
                  yp_field ** 2
          ) ** 0.5
    r_2 = (
                  (xp_field - xp_panel_end) ** 2 +
                  yp_field ** 2
          ) ** 0.5
    theta_1 = arctan2(yp_field, xp_field)
    theta_2 = arctan2(yp_field, xp_field - xp_panel_end)

    ##### Naive implementation in the following comment block, shown here for interpretability:
    """
    ### Calculate u
    u_term_1 = (
            yp_field
            / (2 * pi)
            * (gamma_end - gamma_start)
            / xp_panel_end
            * np.log(r_2 / r_1)
    )
    u_term_2 = (
                       gamma_start * xp_panel_end + (gamma_end - gamma_start) * xp_field
               ) / (
                       2 * pi * xp_panel_end
               ) * (theta_2 - theta_1)

    u = u_term_1 + u_term_2

    ### Calculate v
    v_term_1 = (
                       gamma_start * xp_panel_end + (gamma_end - gamma_start) * xp_field
               ) / (
                       2 * pi * xp_panel_end
               ) * np.log(r_2 / r_1)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_term_2 = np.where(
            yp_field != 0,
            (
                    yp_field
                    / (2 * pi)
                    * (gamma_end - gamma_start)
                    / xp_panel_end
            ) * (
                    xp_panel_end / yp_field + (theta_2 - theta_1)
            ),
            (gamma_end - gamma_start) / (2 * pi)
        )

    v = v_term_1 + v_term_2
    """

    ##### Optimized equivalent implementation, for speed:
    ln_r_2_r_1 = ln(r_2 / r_1)
    d_theta = theta_2 - theta_1
    d_gamma = gamma_end - gamma_start
    tau = 2 * pi
    u_term_1_quantity = (yp_field
                         / tau
                         * d_gamma
                         / xp_panel_end
                         )
    u_term_2_quantity = (
                                gamma_start * xp_panel_end + d_gamma * xp_field
                        ) / (
                                tau * xp_panel_end
                        )

    ### Calculate u
    u_term_1 = u_term_1_quantity * ln_r_2_r_1
    u_term_2 = u_term_2_quantity * d_theta
    u = u_term_1 + u_term_2

    ### TEST
    if backend == "numpy":
        u[is_on_panel] = 0
    elif backend == "casadi":
        u = cas.if_else(
            is_on_panel,
            0,
            u
        )

    ### Calculate v
    v_term_1 = u_term_2_quantity * ln_r_2_r_1

    if backend == "numpy":  # This is basically an optimized version of np.where
        v_term_2 = np.empty_like(v_term_1)
        v_term_2[~is_on_panel] = u_term_1_quantity[~is_on_panel] * (
                xp_panel_end / yp_field[~is_on_panel] -
                d_theta[~is_on_panel]
        )
        v_term_2[is_on_panel] = d_gamma / tau
    elif backend == "casadi":
        yp_field = cas.if_else(
            is_on_panel,
            1,
            yp_field
        )
        v_term_2 = cas.if_else(
            is_on_panel,
            d_gamma / tau,
            u_term_1_quantity * (
                    xp_panel_end / yp_field -
                    d_theta
            ),
        )

    v = v_term_1 + v_term_2

    ### Return
    return u, v


def calculate_induced_velocity_single_panel(
        x_field: Union[float, np.ndarray],
        y_field: Union[float, np.ndarray],
        x_panel_start: float,
        y_panel_start: float,
        x_panel_end: float,
        y_panel_end: float,
        gamma_start: float = 1.,
        gamma_end: float = 1.,
        backend: str = "numpy"
) -> [Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point (x_field, y_field) in a 2D potential-flow flowfield.

    In this flowfield, there is only one singularity element:
    A line vortex going from (x_panel_start, y_panel_start) to (x_panel_end, y_panel_end).
    The strength of this vortex varies linearly from:
        * gamma_start at (x_panel_start, y_panel_start), to:
        * gamma_end at (x_panel_end, y_panel_end).

    By convention here, positive gamma induces clockwise swirl in the flow field.

    Function returns the 2D velocity u, v in the global coordinate system (x, y).

    Inputs x and y can be 1D ndarrays representing various field points,
    in which case the resulting velocities u and v have the corresponding dimensionality.

    The `backend` parameter selects a numerical backend to be used, with the options "numpy" and "casadi".
    NumPy is significantly faster, but CasADi is differentiable.

    """
    ### Calculate the panel coordinate transform (x -> xp, y -> yp), where
    panel_dx = x_panel_end - x_panel_start
    panel_dy = y_panel_end - y_panel_start
    panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

    xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
    xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

    yp_hat_x = -xp_hat_y
    yp_hat_y = xp_hat_x

    ### Transform the field points in to panel coordinates
    x_field_relative = x_field - x_panel_start
    y_field_relative = y_field - y_panel_start

    xp_field = x_field_relative * xp_hat_x + y_field_relative * xp_hat_y  # dot product with the xp unit vector
    yp_field = x_field_relative * yp_hat_x + y_field_relative * yp_hat_y  # dot product with the xp unit vector

    ### Do the vortex math
    up, vp = calculate_induced_velocity_single_panel_panel_coordinates(
        xp_field=xp_field,
        yp_field=yp_field,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        xp_panel_end=panel_length,
        backend=backend
    )

    ### Transform the velocities in panel coordinates back to global coordinates
    u = up * xp_hat_x + vp * yp_hat_x
    v = up * xp_hat_y + vp * yp_hat_y

    ### Return
    return u, v


def calculate_induced_velocity(
        x_field: Union[float, np.ndarray],
        y_field: Union[float, np.ndarray],
        x_panel: np.ndarray,
        y_panel: np.ndarray,
        gamma: np.ndarray,
        backend: str = "numpy",
) -> [Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the induced velocity at a point (x_field, y_field) in a 2D potential-flow flowfield.

    In this flowfield, the following singularity elements are assumed:
        A line vortex that passes through the coordinates specified in (x_panel, y_panel). Each of these vertices is
        called a "node".
        The vorticity of this line vortex per unit length varies linearly between subsequent nodes.
        The vorticity at each node is specified by the parameter gamma.

    By convention here, positive gamma induces clockwise swirl in the flow field.

    Function returns the 2D velocity u, v in the global coordinate system (x, y).

    Inputs x and y can be 1D ndarrays representing various field points,
    in which case the resulting velocities u and v have the corresponding dimensionality.

    The `backend` parameter selects a numerical backend to be used, with the options "numpy" and "casadi".
    NumPy is significantly faster, but CasADi is differentiable.

    """
    try:
        N = len(x_panel)
    except TypeError:
        N = x_panel.shape[0]

    for i in range(N - 1):
        u, v = calculate_induced_velocity_single_panel(
            x_field=x_field,
            y_field=y_field,
            x_panel_start=x_panel[i],
            y_panel_start=y_panel[i],
            x_panel_end=x_panel[i + 1],
            y_panel_end=y_panel[i + 1],
            gamma_start=gamma[i],
            gamma_end=gamma[i + 1],
            backend=backend
        )
        if i == 0:
            u_field = u
            v_field = v
        else:
            u_field += u
            v_field += v

    return u_field, v_field


if __name__ == '__main__':
    X, Y = np.meshgrid(
        np.linspace(-1, 1, 50),
        np.linspace(-1, 1, 50),
        indexing='ij',
    )
    X = X.flatten()
    Y = Y.flatten()

    U, V = calculate_induced_velocity(
        x_field=X,
        y_field=Y,
        x_panel=[-0.5, 0.5, 0.5, -0.5],
        y_panel=[-0.5, -0.5, 0.5, 0.5],
        gamma=[1, 1, -1, 1]
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

    plt.quiver(
        X, Y, U, V,
        (U ** 2 + V ** 2) ** 0.5,
        scale=10
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.title(r"Linear-Strength Vortex: Induced Velocity")
    plt.tight_layout()
    # plt.savefig("C:/Users/User/Downloads/temp.svg")
    plt.show()

    calculate_induced_velocity(
        x_field=-10,
        y_field=0.1,
        x_panel=[-0.5, 0.5, 0.5, -0.5],
        y_panel=[-0.5, -0.5, 0.5, 0.5],
        gamma=[1, 1, -1, 1]
    )
