import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import aerosandbox as asb
import aerosandbox.numpy as np

n_kulfan_inputs_per_side = 3

kulfan_slider_components = []
for side in ["Upper", "Lower"]:
    kulfan_slider_components.append(dcc.Markdown(f"##### {side} Surface"))
    for i in range(n_kulfan_inputs_per_side):
        kulfan_slider_components.extend([
            dcc.Markdown(f"{side} {i+1}:"),
            dcc.Slider(
                id=f'kulfan_{side}_{i}',
                min=0,
                max=1,
                step=0.001,
                value=0.5
            )
        ])


### Build the app
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY], title="Aircraft Design with Dash")
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dcc.Markdown("""
                # Automotive Airfoil Analysis with AeroSandbox and Dash
                
                *Peter Sharpe*
                """)
            ], width=True),
            dbc.Col([
                html.Img(src="assets/MIT-logo-red-gray-72x38.svg", alt="MIT Logo", height="30px"),
            ], width=1)
        ], align="end"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Markdown("#### Shape Parameters (Kulfan)"),
                    *kulfan_slider_components
                ]),
                html.Hr(),
                html.Div([
                    html.H5("Commands"),
                    dbc.Button("Display", id="display_geometry", color="primary", style={"margin": "5px"})
                ]),
                html.Hr(),
                html.Div([
                    html.H5("Aerodynamic Performance"),
                    dbc.Spinner(
                        html.P(id='output'),
                        color="primary",
                    )
                ])
            ], width=3),
            dbc.Col([
                # html.Div(id='display')
                dbc.Spinner(
                    dcc.Graph(id='display', style={'height': '80vh'}),
                    color="primary"
                )
            ], width=True)
        ]),
        html.Hr(),
        html.P([
            html.A("Source code", href="https://github.com/peterdsharpe/AeroSandbox-Interactive-Demo"),
            ". Aircraft design tools powered by ",
            html.A("AeroSandbox", href="https://peterdsharpe.github.com/AeroSandbox"),
            ". Build beautiful UIs for your scientific computing apps with ",
            html.A("Plot.ly ", href="https://plotly.com/"),
            "and ",
            html.A("Dash", href="https://plotly.com/dash/"),
            "!",
        ]),
    ],
    fluid=True
)


def make_table(dataframe):
    return dbc.Table.from_dataframe(
        dataframe,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={

        }
    )



if __name__ == '__main__':
    app.run_server(debug=False)
