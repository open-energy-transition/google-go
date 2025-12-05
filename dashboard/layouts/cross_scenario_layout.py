"""
Layout for cross-scenario comparison (comparing sub-scenarios across main scenarios)
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
from utils.colors import format_scenario_name


def create_cross_scenario_layout(data_loader):
    """Create layout for cross-scenario comparison tab"""

    # Get data for all scenarios
    scenarios_data = {
        'CI_25': data_loader.get_summary_stats('CI_25'),
        'CI_50': data_loader.get_summary_stats('CI_50'),
        'CI_noadd': data_loader.get_summary_stats('CI_noadd')
    }

    # Get years and metrics (should be same across all scenarios)
    years = scenarios_data['CI_25'].get('years', [])
    metrics = scenarios_data['CI_25'].get('metrics', [])

    # Build a map of subscenario -> main_scenario for later lookup
    # And collect ALL unique sub-scenarios across all main scenarios
    subscenario_to_main = {}
    all_subscenarios = []

    for main_scenario, stats in scenarios_data.items():
        for subscenario in stats.get('scenarios', []):
            if subscenario not in subscenario_to_main:
                subscenario_to_main[subscenario] = main_scenario
                all_subscenarios.append(subscenario)

    # Create dropdown options with formatted names
    subscenario_options = [{'label': format_scenario_name(s), 'value': s} for s in all_subscenarios]

    # Set defaults - explicitly set to trigger callback
    default_sub1 = all_subscenarios[0] if len(all_subscenarios) > 0 else 'baseline'
    default_sub2 = all_subscenarios[1] if len(all_subscenarios) > 1 else 'energy-match-50'

    return dbc.Container([
        html.H2("Cross-Scenario Comparison", className="mb-4"),
        html.P("Compare two sub-scenarios across different main scenarios (e.g., compare Baseline in CI_25 vs Baseline in CI_50)"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-year-selector',
                            options=[{'label': str(y), 'value': y} for y in years],
                            value=years[-1] if years else None,
                            clearable=False
                        )
                    ], width=2),

                    # Metric selector
                    dbc.Col([
                        html.Label("Metric:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-metric-selector',
                            options=[{'label': m, 'value': m} for m in metrics],
                            value=metrics[0] if metrics else None,
                            clearable=False
                        )
                    ], width=3),

                    # Plot type selector
                    dbc.Col([
                        html.Label("Plot Type:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-plot-type-selector',
                            options=[
                                {'label': 'Side-by-Side', 'value': 'comparison'},
                                {'label': 'Stacked Bar (All Years)', 'value': 'stacked_bar'},
                                {'label': 'Year Comparison', 'value': 'year_comparison'},
                                {'label': 'Year on Year Evolution', 'value': 'year_on_year_evolution'}
                            ],
                            value='comparison',
                            clearable=False
                        )
                    ], width=2),

                    # Sub-scenario 1 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 1:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-subscenario1-selector',
                            options=subscenario_options,
                            value=default_sub1,
                            clearable=False
                        )
                    ], width=2),

                    # Sub-scenario 2 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 2:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-subscenario2-selector',
                            options=subscenario_options,
                            value=default_sub2,
                            clearable=False
                        )
                    ], width=3),
                ])
            ])
        ], className="mb-4"),

        # Comparison plot
        dbc.Card([
            dbc.CardBody([
                html.H4("Side-by-Side Comparison", className="mb-3"),
                dcc.Graph(id='cross-comparison-plot', style={'height': '500px'})
            ])
        ], className="mb-4"),

        # Difference plot
        dbc.Card([
            dbc.CardBody([
                html.H4("Difference Analysis", className="mb-3"),
                html.P("Shows the difference between Sub-Scenario 2 and Sub-Scenario 1 (positive = higher in Scenario 2)"),
                dcc.Graph(id='cross-difference-plot', style={'height': '400px'})
            ])
        ], className="mb-4"),

        # Summary statistics
        dbc.Card([
            dbc.CardBody([
                html.H4("Comparison Statistics", className="mb-3"),
                html.Div(id='cross-summary-stats')
            ])
        ])
    ], fluid=True, className="p-4")
