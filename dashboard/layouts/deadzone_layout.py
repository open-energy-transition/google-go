"""
Layout for Dead Zone Analysis tab
Provides three types of frontier comparisons:
1. Spatial scope + scenario fixed, compare years
2. Spatial scope fixed, compare scenarios and years
3. Scenario + year fixed, compare spatial scopes
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_deadzone_layout(data_loader):
    """Create the Dead Zone Analysis tab layout"""

    # Get available data from consolidated results
    stats = data_loader.get_summary_stats()
    years = stats.get('years', [])
    scenarios = stats.get('scenarios', [])

    return dbc.Container([
        html.H3("Energy Procurement Frontier - Frontier Comparisons", className="mb-4"),

        # Control panel
        dbc.Card([
            dbc.CardBody([
                # Selectors
                dbc.Row([
                    # Year selector
                    dbc.Col([
                        html.Label("Year:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='deadzone-year-selector',
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': str(y), 'value': y} for y in years],
                            value='all',
                            clearable=False
                        )
                    ], width=2),

                    # Scenario selector
                    dbc.Col([
                        html.Label("Scenarios (select up to 5 or All):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='deadzone-scenario-selector',
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': s, 'value': s} for s in scenarios],
                            value=['baseline'],
                            clearable=False,
                            multi=True  # Allow multiple selection
                        )
                    ], width=5),

                    # Country/spatial scope selector
                    dbc.Col([
                        html.Label("Countries (select up to 5 or All):", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='deadzone-country-selector',
                            options=[{'label': 'All', 'value': 'all'}],  # Will be populated by callback
                            value=['EU'],
                            clearable=False,
                            multi=True  # Allow multiple selection
                        )
                    ], width=5),
                ], className="mb-2")
            ])
        ], className="mb-4"),

        # Dead zone plot
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="deadzone-loading",
                    type="default",
                    children=[
                        dcc.Graph(id='deadzone-plot', style={'height': '700px'})
                    ]
                )
            ])
        ]),

        # Summary/description
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Summary"),
                    dbc.CardBody(id='deadzone-summary')
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
