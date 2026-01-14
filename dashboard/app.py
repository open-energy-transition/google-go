"""
Interactive Dashboard for Energy System Results
Visualizes results from CI_25, CI_50, and CI_noadd scenarios
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

# Import utilities
from utils.data_loader import DataLoader
from utils.colors import ColorMapper
from layouts import single_scenario_layout, cross_scenario_layout, deadzone_layout, timeseries_layout, insights_layout
from callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Google-Go Analysis Dashboard"
)

# Initialize data loader
data_loader = DataLoader(results_dir="../results")

# Load all data at startup
print("Loading data...")
data_loader.load_all_data()
print("Data loaded successfully!")

# Define the app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Google-Go Analysis Dashboard",
                   className="text-center mb-2 mt-4",
                   style={"color": "#1f77b4", "fontWeight": "bold"}),
            html.H5("Interactive Visualization Energy System Modelling Results",
                   className="text-center mb-4",
                   style={"color": "#666"})
        ])
    ]),

    # Tabs
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id='main-tabs', value='single-tab', children=[
                dcc.Tab(label='Single Scenario Analysis', value='single-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Cross-Scenario Comparison', value='cross-scenario-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Energy Procurement Frontier', value='deadzone-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Timeseries Exploration', value='timeseries-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Key Insights', value='insights-tab',
                       style={'fontWeight': 'bold'}),
            ], style={'fontSize': '16px'})
        ])
    ]),

    # Tab content - ALL LAYOUTS PRELOADED
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(single_scenario_layout.create_layout(data_loader),
                        id='single-content', style={'display': 'block'}),
                html.Div(cross_scenario_layout.create_cross_scenario_layout(data_loader),
                        id='cross-scenario-content', style={'display': 'none'}),
                html.Div(deadzone_layout.create_deadzone_layout(data_loader),
                        id='deadzone-content', style={'display': 'none'}),
                html.Div(timeseries_layout.create_timeseries_layout(data_loader),
                        id='timeseries-content', style={'display': 'none'}),
                html.Div(insights_layout.create_insights_layout(data_loader),
                        id='insights-content', style={'display': 'none'}),
            ], id='tab-content', className='mt-4')
        ])
    ]),

], fluid=True, style={"maxWidth": "1800px"})

# Callback to show/hide tab content
@app.callback(
    [Output('single-content', 'style'),
     Output('cross-scenario-content', 'style'),
     Output('deadzone-content', 'style'),
     Output('timeseries-content', 'style'),
     Output('insights-content', 'style')],
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    styles = [{'display': 'none'}] * 5
    if tab == 'single-tab':
        styles[0] = {'display': 'block'}
    elif tab == 'cross-scenario-tab':
        styles[1] = {'display': 'block'}
    elif tab == 'deadzone-tab':
        styles[2] = {'display': 'block'}
    elif tab == 'timeseries-tab':
        styles[3] = {'display': 'block'}
    elif tab == 'insights-tab':
        styles[4] = {'display': 'block'}
    return styles

# Register all callbacks
register_callbacks(app, data_loader)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
