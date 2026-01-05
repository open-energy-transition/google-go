"""
Layout for cross-scenario comparison
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
from utils.colors import format_scenario_name


def create_cross_scenario_layout(data_loader):
    """Create layout for cross-scenario comparison tab"""

    # Get data from consolidated results
    stats = data_loader.get_summary_stats()
    years = stats.get('years', [])
    metrics = stats.get('metrics', [])
    all_subscenarios = stats.get('scenarios', [])

    # Create dropdown options with formatted names
    subscenario_options = [{'label': format_scenario_name(s), 'value': s} for s in all_subscenarios]

    # Set defaults - explicitly set to trigger callback
    default_sub1 = all_subscenarios[0] if len(all_subscenarios) > 0 else 'baseline'
    default_sub2 = all_subscenarios[1] if len(all_subscenarios) > 1 else 'energy-match-50'
    default_sub3 = all_subscenarios[2] if len(all_subscenarios) > 2 else None
    default_sub4 = all_subscenarios[3] if len(all_subscenarios) > 3 else None

    return dbc.Container([
        html.H2("Cross-Scenario Comparison", className="mb-4"),
        html.P("Compare up to 4 scenarios"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-year-selector',
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': str(y), 'value': y} for y in years],
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
                                {'label': 'Stacked Bar + Total Line', 'value': 'stacked_with_total'},
                                {'label': 'Year Comparison', 'value': 'year_comparison'},
                                {'label': 'Year on Year Evolution', 'value': 'year_on_year_evolution'}
                            ],
                            value='comparison',
                            clearable=False
                        )
                    ], width=2),

                    # Grouping toggle
                    dbc.Col([
                        html.Label("Group By:", style={'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='cross-grouping-selector',
                            options=[
                                {'label': 'Year', 'value': 'year'},
                                {'label': 'Scenario', 'value': 'scenario'}
                            ],
                            value='year',
                            inline=True,
                            labelStyle={'marginRight': '15px'}
                        )
                    ], width=3),
                ]),

                # Second row for sub-scenarios
                dbc.Row([
                    # Sub-scenario 1 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 1:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-subscenario1-selector',
                            options=subscenario_options,
                            value=default_sub1,
                            clearable=False
                        )
                    ], width=3),

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

                    # Sub-scenario 3 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 3 (optional):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-subscenario3-selector',
                            options=subscenario_options,
                            value=default_sub3,
                            clearable=True,
                            placeholder="Select (optional)"
                        )
                    ], width=3),

                    # Sub-scenario 4 selector
                    dbc.Col([
                        html.Label("Sub-Scenario 4 (optional):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='cross-subscenario4-selector',
                            options=subscenario_options,
                            value=default_sub4,
                            clearable=True,
                            placeholder="Select (optional)"
                        )
                    ], width=3),
                ], className="mt-2")
            ])
        ], className="mb-4"),

        # Carrier selection and plot area
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Carrier Selection"),
                    dbc.CardBody([
                        dcc.Checklist(
                            id='cross-carrier-selector',
                            options=[],  # Will be populated by callback
                            value=[],
                            labelStyle={'display': 'block', 'margin': '5px'},
                            inputStyle={'marginRight': '10px'}
                        )
                    ])
                ])
            ], width=3),

            dbc.Col([
                # Comparison plot
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Side-by-Side Comparison", className="mb-3"),
                        dcc.Graph(id='cross-comparison-plot', style={'height': '500px'})
                    ])
                ])
            ], width=9)
        ], className="mb-4"),

        # Difference plot (full width)
        dbc.Card([
            dbc.CardBody([
                html.H4("Difference Analysis", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Scenario A (baseline):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='diff-scenario-a-selector',
                            options=subscenario_options,
                            value=default_sub1,
                            clearable=False
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Scenario B (compare to):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='diff-scenario-b-selector',
                            options=subscenario_options,
                            value=default_sub2,
                            clearable=False
                        )
                    ], width=3),
                    dbc.Col([
                        html.P("Shows difference: Scenario B - Scenario A (positive = higher in B)",
                               style={'marginTop': '30px', 'fontSize': '14px', 'color': '#666'})
                    ], width=6)
                ], className="mb-3"),
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
