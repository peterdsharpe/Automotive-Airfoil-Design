import aerosandbox as asb
from aerosandbox import cas, Airfoil
import numpy as np
from numpy import pi
from aerodynamics.singularities import calculate_induced_velocity_line_singularities
from aerosandbox.tools.casadi_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
import copy


class AutomotiveAnalysis():
    def __init__(self,
                 airfoils: List[Airfoil] = None
                 ):
        if airfoils is None:
            self.airfoils = [
                Airfoil("naca2412").translate(translate_y=0.2)
            ]
        else:
            self.airfoils = airfoils

    def analyze(self):
        opti = asb.Opti()

        for airfoil in self.airfoils:
            airfoil.gamma = opti.variable(
                n_vars=airfoil.n_points()
            )
            airfoil.sigma = np.zeros(airfoil.n_points())

        ### Make a function to calculate the local velocity at arbitrary points
        def calculate_velocity(
                x_field,
                y_field,
                backend="numpy"
        ) -> [np.ndarray, np.ndarray]:
            u_freestream = 1
            v_freestream = 0

            u_field = u_freestream
            v_field = v_freestream

            for airfoil in self.airfoils:
                u_field_induced, v_field_induced = calculate_induced_velocity_line_singularities(
                    x_field=x_field,
                    y_field=y_field,
                    x_panels=airfoil.x(),
                    y_panels=airfoil.y(),
                    gamma=airfoil.gamma,
                    sigma=airfoil.sigma,
                    backend=backend
                )

                u_field_induced_mirror, v_field_induced_mirror = calculate_induced_velocity_line_singularities(
                    x_field=x_field,
                    y_field=y_field,
                    x_panels=airfoil.x(),
                    y_panels=-airfoil.y(),
                    gamma=-airfoil.gamma,
                    sigma=airfoil.sigma,
                    backend=backend
                )
                u_field_induced += u_field_induced_mirror
                v_field_induced += v_field_induced_mirror

                u_field += u_field_induced
                v_field += v_field_induced

            return u_field, v_field

        self.calculate_velocity = calculate_velocity

        for airfoil in self.airfoils:
            ### Compute normal velocities at the middle of each panel
            x_midpoints = trapz(airfoil.x())
            y_midpoints = trapz(airfoil.y())

            u_midpoints, v_midpoints = calculate_velocity(
                x_field=x_midpoints,
                y_field=y_midpoints,
                backend="casadi"
            )

            panel_dx = diff(airfoil.x())
            panel_dy = diff(airfoil.y())
            panel_length = (panel_dx ** 2 + panel_dy ** 2) ** 0.5

            xp_hat_x = panel_dx / panel_length  # x-coordinate of the xp_hat vector
            xp_hat_y = panel_dy / panel_length  # y-coordinate of the yp_hat vector

            yp_hat_x = -xp_hat_y
            yp_hat_y = xp_hat_x

            normal_velocities = u_midpoints * yp_hat_x + v_midpoints * yp_hat_y

            ### Add in flow tangency constraint
            opti.subject_to(normal_velocities == 0)

            ### Add in Kutta condition
            opti.subject_to(airfoil.gamma[0] + airfoil.gamma[-1] == 0)

        ### Calculate lift
        total_vorticity = cas.sum1(
            (airfoil.gamma[1:] + airfoil.gamma[:-1]) / 2 *
            panel_length
        )

        ### Solve
        sol = opti.solve()
        for airfoil in self.airfoils:
            airfoil.substitute_solution(sol)
        self.total_vorticity = sol.value(total_vorticity)

    def visualize_matplotlib(self, res=100):
        ### Plot the flowfield
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=250)

        margin = 0.5
        xlim = (
            -0.5,
            1.5
        )
        ylim = (
            0,
            1
        )
        x = np.linspace(*xlim, round(res * (1 + 2 * margin) / (2 * margin)))
        y = np.linspace(*ylim, res)
        X, Y = np.meshgrid(
            x,
            y,
        )
        X = X.flatten()
        Y = Y.flatten()

        U, V = self.calculate_velocity(
            x_field=X,
            y_field=Y,
        )
        speed = (U ** 2 + V ** 2) ** 0.5
        # plt.quiver(
        #     X, Y, U, V, speed,
        #     scale=30
        # )

        from palettable.colorbrewer.diverging import RdBu_4 as streamplot_colormap

        plt.streamplot(
            x,
            y,
            U.reshape(len(y), len(x)),
            V.reshape(len(y), len(x)),
            color=speed.reshape(len(y), len(x)),
            density=2,
            arrowsize=0,
            cmap=streamplot_colormap.mpl_colormap,
        )

        for airfoil in self.airfoils:
            plt.fill(airfoil.x(), airfoil.y(), "k", linewidth=0, zorder=4)
        CB = plt.colorbar(
            orientation="horizontal",
            shrink=0.8,
            aspect=40,
        )
        CB.set_label(r"Relative Airspeed ($U/U_\infty$)")
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        plt.clim(0.4, 1.6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r"$x/c$")
        plt.ylabel(r"$y/c$")
        plt.title(rf"Automotive Aerodynamics: Flow Field")
        plt.annotate(
            rf"Downforce $ = {-self.total_vorticity:.3f}\rho_\infty U_\infty^2 Lc$",
            (0.02, 0.97),
            xycoords='axes fraction',
            ha="left",
            va="top",
            backgroundcolor=(1, 1, 1, 0.5)
        )
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    a = AutomotiveAnalysis(
        airfoils=[
            Airfoil("e423")
                .repanel(n_points_per_side=50)
                .scale(1, -1)
                .rotate(7 * pi / 180, 0.4, 0)
                .translate(0, 0.25),
            Airfoil("s1223")
                .repanel(n_points_per_side=25)
                .scale(1, -1)
                .scale(0.4, 0.4)
                .rotate(35 * pi / 180)
                .translate(0.9, 0.37),
        ]
    )
    a.analyze()
    a.visualize_matplotlib()
