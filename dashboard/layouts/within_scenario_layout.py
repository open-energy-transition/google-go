"""
Layout for within-scenario comparison (comparing sub-scenarios within same main scenario)
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
from utils.colors import format_scenario_name


def create_within_scenario_layout(data_loader):
    """Create layout for within-scenario comparison tab"""

    # Get data for all scenarios
    scenarios_data = {
        'CI_25': data_loader.get_summary_stats('CI_25'),
        'CI_50': data_loader.get_summary_stats('CI_50'),
        'CI_noadd': data_loader.get_summary_stats('CI_noadd')
    }

    # Get years and metrics (should be same across all scenarios)
    years = scenarios_data['CI_25'].get('years', [])
    metrics = scenarios_data['CI_25'].get('metrics', [])

    # Get initial sub-scenarios for CI_50 (default main scenario)
    ci50_subscenarios = scenarios_data['CI_50'].get('scenarios', [])
    subscenario_options = [{'label': format_scenario_name(s), 'value': s} for s in ci50_subscenarios]
    default_sub1 = ci50_subscenarios[0] if len(ci50_subscenarios) > 0 else None
    default_sub2 = ci50_subscenarios[1] if len(ci50_subscenarios) > 1 else None

    return dbc.Container([
        html.H2("Within-Scenario Comparison", className="mb-4"),
        html.P("Compare two sub-scenarios within the same main scenario (e.g., compare Hourly 90% vs Hourly 95% within CI 50%)"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Main scenario selector
                    dbc.Col([
                        html.Label("Main Scenario:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='within-main-scenario-selector',
                            options=[
                                {'label': 'CI 25%', 'value': 'CI_25'},
                                {'label': 'CI 50%', 'value': 'CI_50'},
                                {'label': 'No Additional Constraints', 'value': 'CI_noadd'}
                            ],
                            value='CI_50',
                            clearable=False
                        )
                    ], width=3),

                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='within-year-selector',
                            options=[{'label': str(y), 'value': y} for y in years],
                            value=years[-1] if years else None,
                            clearable=False
                        )
                    ], width=2),

                    # Metric selector
                    dbc.Col([
                        html.Label("Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='within-metric-selector',
                            options=[{'label': m, 'value': m} for m in metrics],
                            value=metrics[0] if metrics else None,
                            clearable=False
                        )
                    ], width=3),

                    # Sub-scenario 1 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 1:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='within-subscenario1-selector',
                            options=subscenario_options,
                            value=default_sub1,
                            clearable=False
                        )
                    ], width=2),

                    # Sub-scenario 2 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 2:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='within-subscenario2-selector',
                            options=subscenario_options,
                            value=default_sub2,
                            clearable=False
                        )
                    ], width=2),
                ])
            ])
        ], className="mb-4"),

        # Comparison plot
        dbc.Card([
            dbc.CardBody([
                html.H4("Side-by-Side Comparison", className="mb-3"),
                dcc.Graph(id='within-comparison-plot', style={'height': '500px'})
            ])
        ], className="mb-4"),

        # Difference plot
        dbc.Card([
            dbc.CardBody([
                html.H4("Difference Analysis", className="mb-3"),
                html.P("Shows the difference between Sub-Scenario 2 and Sub-Scenario 1 (positive = higher in Scenario 2)"),
                dcc.Graph(id='within-difference-plot', style={'height': '400px'})
            ])
        ], className="mb-4"),

        # Summary statistics
        dbc.Card([
            dbc.CardBody([
                html.H4("Comparison Statistics", className="mb-3"),
                html.Div(id='within-summary-stats')
            ])
        ])
    ], fluid=True, className="p-4")
