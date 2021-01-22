from numpy import pi
import numpy as np


def calculate_induced_velocity_panel_coordinates(
        xp_field,
        yp_field,
        gamma_start=1.,
        gamma_end=1.,
        xp_panel_end=1.,
):
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
    in which case u and v have corresponding dimensionality.

    Equations from the seminal textbook "Low Speed Aerodynamics" by Katz and Plotkin.
    Equations 11.99 and 11.100.

    """
    ### Do some geometry calculation
    r_1 = (
                  xp_field ** 2 +
                  yp_field ** 2
          ) ** 0.5
    r_2 = (
                  (xp_field - xp_panel_end) ** 2 +
                  yp_field ** 2
          ) ** 0.5
    theta_1 = np.arctan2(yp_field, xp_field)
    theta_2 = np.arctan2(yp_field, xp_field - xp_panel_end)

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
    log_r_2_r_1 = np.log(r_2 / r_1)
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
    u_term_1 = u_term_1_quantity * log_r_2_r_1
    u_term_2 = u_term_2_quantity * d_theta
    u = u_term_1 + u_term_2

    ### Calculate v
    v_term_1 = u_term_2_quantity * log_r_2_r_1

    yp_field_is_nonzero = yp_field != 0
    v_term_2 = np.empty_like(v_term_1)
    v_term_2[yp_field_is_nonzero] = u_term_1_quantity[yp_field_is_nonzero] * (
            xp_panel_end / yp_field[yp_field_is_nonzero] +
            d_theta[yp_field_is_nonzero]
    )
    v_term_2[~yp_field_is_nonzero] = d_gamma / tau

    v = v_term_1 + v_term_2

    ### Return
    return u, v


def calculate_induced_velocity(
        x_field,
        y_field,
        x_panel_start,
        y_panel_start,
        x_panel_end,
        y_panel_end,
        gamma_start=1.,
        gamma_end=1.,
):
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
    in which case u and v have the corresponding dimensionality.

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
    up, vp = calculate_induced_velocity_panel_coordinates(
        xp_field=xp_field,
        yp_field=yp_field,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        xp_panel_end=panel_length
    )

    ### Transform the velocities in panel coordinates back to global coordinates
    u = up * xp_hat_x + vp * yp_hat_x
    v = up * xp_hat_y + vp * yp_hat_y

    ### Return
    return u, v


if __name__ == '__main__':
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
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

    plt.quiver(
        X, Y, U, V,
        (U ** 2 + V ** 2) ** 0.5,
        scale=5
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.title(r"Linear-Strength Vortex: Induced Velocity")
    plt.tight_layout()
    # plt.savefig("C:/Users/User/Downloads/temp.svg")
    plt.show()
