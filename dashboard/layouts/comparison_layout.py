"""
Layout for Comparison tab (comparing CI_25, CI_50, CI_noadd)
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_layout(data_loader):
    """Create the Comparison tab layout"""

    # Get available data from all scenarios
    scenarios_data = {
        'CI_25': data_loader.get_summary_stats('CI_25'),
        'CI_50': data_loader.get_summary_stats('CI_50'),
        'CI_noadd': data_loader.get_summary_stats('CI_noadd')
    }

    # Get common years and metrics across all scenarios
    all_years = set()
    all_metrics = set()
    all_sub_scenarios = set()

    for stats in scenarios_data.values():
        all_years.update(stats.get('years', []))
        all_metrics.update(stats.get('metrics', []))
        all_sub_scenarios.update(stats.get('scenarios', []))

    years = sorted(list(all_years))
    metrics = sorted(list(all_metrics))

    # Show ALL sub-scenarios from all main scenarios
    # (since naming is different, user can pick the ones they want to compare)
    sub_scenarios = sorted(list(all_sub_scenarios))

    return dbc.Container([
        html.H3("Cross-Scenario Comparison", className="mb-4"),
        html.P("Compare results across CI_25, CI_50, and CI_noadd scenarios",
               className="text-muted"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='comp-year-selector',
                            options=[{'label': str(y), 'value': y} for y in years],
                            value=years[0] if years else None,
                            clearable=False
                        )
                    ], width=2),

                    # Sub-scenario selector (NEW)
                    dbc.Col([
                        html.Label("Sub-Scenario:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='comp-subscenario-selector',
                            options=[{'label': s, 'value': s} for s in sub_scenarios],
                            value='baseline' if 'baseline' in sub_scenarios else (sub_scenarios[0] if sub_scenarios else None),
                            clearable=False
                        )
                    ], width=3),

                    # Metric selector
                    dbc.Col([
                        html.Label("Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='comp-metric-selector',
                            options=[{'label': m, 'value': m} for m in metrics],
                            value=metrics[0] if metrics else None,
                            clearable=False
                        )
                    ], width=3),

                    # Scenarios to compare
                    dbc.Col([
                        html.Label("Main Scenarios:", style={'fontWeight': 'bold'}),
                        dcc.Checklist(
                            id='comp-scenarios-selector',
                            options=[
                                {'label': 'CI_25', 'value': 'CI_25'},
                                {'label': 'CI_50', 'value': 'CI_50'},
                                {'label': 'CI_noadd', 'value': 'CI_noadd'}
                            ],
                            value=['CI_25', 'CI_50', 'CI_noadd'],
                            inline=True,
                            labelStyle={'marginRight': '15px'},
                            inputStyle={'marginRight': '5px'}
                        )
                    ], width=4),
                ])
            ])
        ], className="mb-4"),

        # Main comparison plot
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="comp-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='comp-main-plot', style={'height': '600px'})
                    ]
                )
            ], width=12)
        ]),

        # Secondary comparison - side by side
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scenario Details: CI_25"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="comp-ci25-loading",
                            type="default",
                            children=[
                                dcc.Graph(id='comp-ci25-plot', style={'height': '350px'})
                            ]
                        )
                    ])
                ])
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scenario Details: CI_50"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="comp-ci50-loading",
                            type="default",
                            children=[
                                dcc.Graph(id='comp-ci50-plot', style={'height': '350px'})
                            ]
                        )
                    ])
                ])
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scenario Details: CI_noadd"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="comp-cinoadd-loading",
                            type="default",
                            children=[
                                dcc.Graph(id='comp-cinoadd-plot', style={'height': '350px'})
                            ]
                        )
                    ])
                ])
            ], width=4),
        ], className="mt-4"),

        # Difference analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Difference Analysis"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="comp-diff-loading",
                            type="default",
                            children=[
                                dcc.Graph(id='comp-diff-plot', style={'height': '400px'})
                            ]
                        )
                    ])
                ])
            ])
        ], className="mt-4"),

        # Summary comparison table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Summary Comparison Table"),
                    dbc.CardBody(id='comp-summary-table')
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
