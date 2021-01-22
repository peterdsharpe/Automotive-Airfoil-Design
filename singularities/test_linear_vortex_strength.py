from .linear_strength_vortex import *

def test_calculate_induced_velocity_panel_coordinates():
    X, Y = np.meshgrid(
        np.linspace(-1, 2, 50),
        np.linspace(-1, 1, 50),
        indexing='ij',
    )
    X = X.flatten()
    Y = Y.flatten()

    U, V = calculate_induced_velocity_panel_coordinates(
        xp_field=X,
        yp_field=Y,
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

def test_calculate_induced_velocity():
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
        x_panel_start=1,
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