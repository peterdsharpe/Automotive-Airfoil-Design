import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_2D.singularities import calculate_induced_velocity_line_singularities
import matplotlib.pyplot as plt
from typing import List
import matplotlib.patches as mpatches

if __name__ == '__main__':
    a = AutomotiveAnalysis(
        airfoils=[
            asb.Airfoil("e423")
                .repanel(n_points_per_side=50)
                .scale(1, -1)
                .rotate(np.radians(7), 0.4, 0)
                .translate(0, 0.25),
            asb.Airfoil("s1223")
                .repanel(n_points_per_side=25)
                .scale(1, -1)
                .scale(0.4, 0.4)
                .rotate(np.radians(35))
                .translate(0.9, 0.37),
        ]
    )
    a.analyze()
    a.visualize_matplotlib_streamlines()
    a.visualize_matplotlib_Cp()
