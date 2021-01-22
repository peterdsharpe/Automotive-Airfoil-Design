from linear_strength_vortex import *

print(
    calculate_induced_velocity_single_panel_panel_coordinates(
        xp_field = 0.5,
        yp_field = 1e-6,
        gamma_start = 1,
        gamma_end= 0,
        xp_panel_end=1
    )
)