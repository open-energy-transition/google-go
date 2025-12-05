"""
Layout for single scenario analysis (unified layout for CI_25, CI_50, CI_noadd)
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from utils.colors import format_scenario_name, format_main_scenario_name


def create_layout(data_loader):
    """Create the single scenario analysis tab layout"""

    # Get available data - use CI_25 as template for years and metrics
    all_main_scenarios = ['CI_25', 'CI_50', 'CI_noadd']
    stats = data_loader.get_summary_stats('CI_25')
    years = stats.get('years', [])
    metrics = stats.get('metrics', [])

    # Get scenarios for the first main scenario as default
    scenarios = stats.get('scenarios', [])

    return dbc.Container([
        html.H3("Single Scenario Analysis", className="mb-4"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                # First row - Main scenario selector
                dbc.Row([
                    dbc.Col([
                        html.Label("Main Scenario:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='single-main-scenario-selector',
                            options=[
                                {'label': format_main_scenario_name(s), 'value': s}
                                for s in all_main_scenarios
                            ],
                            value='CI_25',
                            clearable=False,
                            style={'fontSize': '14px'}
                        )
                    ], width=12),
                ], className="mb-3"),

                # Second row - Other selectors
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='single-year-selector',
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': str(y), 'value': y} for y in years],
                            value=years[0] if years else None,
                            clearable=False
                        )
                    ], width=2),

                    # Sub-scenario selector
                    dbc.Col([
                        html.Label("Sub-Scenario:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='single-scenario-selector',
                            options=[{'label': format_scenario_name(s), 'value': s} for s in scenarios],
                            value=scenarios[0] if scenarios else None,
                            clearable=False
                        )
                    ], width=3),

                    # Metric selector
                    dbc.Col([
                        html.Label("Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='single-metric-selector',
                            options=[{'label': m, 'value': m} for m in metrics],
                            value=metrics[0] if metrics else None,
                            clearable=False
                        )
                    ], width=5),

                    # Plot type selector
                    dbc.Col([
                        html.Label("Plot Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='single-plot-type-selector',
                            options=[
                                {'label': 'Bar Chart', 'value': 'bar'},
                                {'label': 'Stacked Bar (All Years)', 'value': 'stacked_bar'},
                                {'label': 'Stacked Area', 'value': 'area'},
                                {'label': 'Pie Chart', 'value': 'pie'},
                                {'label': 'Year Comparison', 'value': 'year_comparison'},
                                {'label': 'Year on Year Evolution', 'value': 'year_on_year_evolution'}
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
                    id="single-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='single-main-plot', style={'height': '600px'})
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
                            id='single-carrier-selector',
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
                    id="single-secondary-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='single-secondary-plot', style={'height': '400px'})
                    ]
                )
            ], width=9)
        ], className="mt-4"),

        # Summary statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Summary Statistics"),
                    dbc.CardBody(id='single-summary-stats')
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
