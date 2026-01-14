"""
Layout for Timeseries Exploration tab
Allows interactive exploration of timeseries data across scenarios
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_timeseries_layout(data_loader):
    """Create the Timeseries Exploration tab layout"""

    return dbc.Container([
        html.H3("Timeseries Exploration", className="mb-4"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-year-selector',
                            options=[],  # Will be populated by callback
                            value=2035,
                            clearable=False
                        )
                    ], width=2),

                    # Scenario selector (multi-select)
                    dbc.Col([
                        html.Label("Scenarios (select 1 or more):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-scenario-selector',
                            options=[],  # Will be populated by callback
                            value=[],
                            multi=True,
                            placeholder="Select scenarios..."
                        )
                    ], width=4),

                    # Timeseries type selector
                    dbc.Col([
                        html.Label("Timeseries Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-type-selector',
                            options=[],  # Will be populated by callback
                            value=None,
                            clearable=False,
                            placeholder="Select type..."
                        )
                    ], width=4),
                ], className="mb-3"),

                dbc.Row([
                    # Country selector
                    dbc.Col([
                        html.Label("Country:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-country-selector',
                            options=[],  # Will be populated by callback
                            value=None,
                            clearable=False,
                            placeholder="Select country..."
                        )
                    ], width=3),

                    # Carrier selector (optional filter)
                    dbc.Col([
                        html.Label("Carrier (optional filter):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-carrier-selector',
                            options=[],  # Will be populated by callback
                            value=None,
                            multi=True,
                            placeholder="All carriers"
                        )
                    ], width=4),

                    # Time range selector
                    dbc.Col([
                        html.Label("Time Range:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-timerange-selector',
                            options=[
                                {'label': 'Full Year', 'value': 'full'},
                                {'label': 'January', 'value': '2013-01'},
                                {'label': 'July', 'value': '2013-07'},
                                {'label': 'First Week (Jan)', 'value': 'week1'},
                                {'label': 'Summer Week (Jul)', 'value': 'week_summer'},
                                {'label': 'Winter Week (Jan)', 'value': 'week_winter'},
                            ],
                            value='week1',
                            clearable=False
                        )
                    ], width=3),

                    # Plot type
                    dbc.Col([
                        html.Label("Plot Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ts-plot-type-selector',
                            options=[
                                {'label': 'Line Plot', 'value': 'line'},
                                {'label': 'Stacked Area', 'value': 'area'},
                            ],
                            value='area',  # Default to area (will switch to line for Electricity Balance via callback)
                            clearable=False
                        )
                    ], width=2),
                ], className="mb-2")
            ])
        ], className="mb-4"),

        # Info banner
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.Strong("⚡ Fast Loading with Parquet: "),
                    "Optimized data loading (~2-5 seconds). Subsequent loads for the same parameters are instant (cached). Demand shown as black dashed line."
                ], color="success", className="mb-3")
            ])
        ]),

        # Timeseries plot
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="ts-loading",
                    type="circle",
                    fullscreen=False,
                    children=[
                        dcc.Graph(id='ts-plot', style={'height': '600px'})
                    ]
                )
            ])
        ]),

        # Summary/info
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Timeseries Info"),
                    dbc.CardBody(id='ts-info')
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
