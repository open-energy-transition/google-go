"""
Layout for CI_50 scenario tab
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from utils.colors import format_scenario_name


def create_layout(data_loader):
    """Create the CI_50 tab layout"""

    # Get available data
    stats = data_loader.get_summary_stats('CI_50')
    years = stats.get('years', [])
    scenarios = stats.get('scenarios', [])
    metrics = stats.get('metrics', [])

    return dbc.Container([
        html.H3("CI_50 Scenario Analysis", className="mb-4"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ci50-year-selector',
                            options=[{'label': str(y), 'value': y} for y in years],
                            value=years[0] if years else None,
                            clearable=False
                        )
                    ], width=3),

                    # Scenario selector
                    dbc.Col([
                        html.Label("Scenario:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ci50-scenario-selector',
                            options=[{'label': format_scenario_name(s), 'value': s} for s in scenarios],
                            value=scenarios[0] if scenarios else None,
                            clearable=False
                        )
                    ], width=3),

                    # Metric selector
                    dbc.Col([
                        html.Label("Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ci50-metric-selector',
                            options=[{'label': m, 'value': m} for m in metrics],
                            value=metrics[0] if metrics else None,
                            clearable=False
                        )
                    ], width=4),

                    # Plot type selector
                    dbc.Col([
                        html.Label("Plot Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='ci50-plot-type-selector',
                            options=[
                                {'label': 'Bar Chart', 'value': 'bar'},
                                {'label': 'Stacked Area', 'value': 'area'},
                                {'label': 'Pie Chart', 'value': 'pie'},
                                {'label': 'Stacked Bar (All Years)', 'value': 'stacked_bar'},
                                {'label': 'Year Comparison', 'value': 'year_comparison'},
                                {'label': 'Technology Trajectory', 'value': 'trajectory'}
                            ],
                            value='bar',
                            clearable=False
                        )
                    ], width=2),
                ])
            ])
        ], className="mb-4"),

        # Visualization area
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="ci50-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='ci50-main-plot', style={'height': '600px'})
                    ]
                )
            ], width=12)
        ]),

        # Additional controls and secondary plots
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Carrier Selection"),
                    dbc.CardBody([
                        dcc.Checklist(
                            id='ci50-carrier-selector',
                            options=[],  # Will be populated by callback
                            value=[],
                            labelStyle={'display': 'block', 'margin': '5px'},
                            inputStyle={'marginRight': '10px'}
                        )
                    ])
                ])
            ], width=3),

            dbc.Col([
                dcc.Loading(
                    id="ci50-secondary-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='ci50-secondary-plot', style={'height': '400px'})
                    ]
                )
            ], width=9)
        ], className="mt-4"),

        # Summary statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Summary Statistics"),
                    dbc.CardBody(id='ci50-summary-stats')
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
