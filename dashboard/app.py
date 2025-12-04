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
from layouts import ci25_layout, ci50_layout, cinoadd_layout, comparison_layout, within_scenario_layout
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
            html.H5("Interactive Visualization of CI_25, CI_50, and CI_noadd Scenarios",
                   className="text-center mb-4",
                   style={"color": "#666"})
        ])
    ]),

    # Tabs
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id='main-tabs', value='ci25-tab', children=[
                dcc.Tab(label='CI_25', value='ci25-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='CI_50', value='ci50-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='CI_noadd', value='cinoadd-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Comparison', value='comparison-tab',
                       style={'fontWeight': 'bold'}),
                dcc.Tab(label='Within-Scenario', value='within-scenario-tab',
                       style={'fontWeight': 'bold'}),
            ], style={'fontSize': '16px'})
        ])
    ]),

    # Tab content
    dbc.Row([
        dbc.Col([
            html.Div(id='tab-content', className='mt-4')
        ])
    ]),

    # Store data loader in a hidden div (for callbacks to access)
    html.Div(id='data-store', style={'display': 'none'})

], fluid=True, style={"maxWidth": "1800px"})

# Callback to render tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'ci25-tab':
        return ci25_layout.create_layout(data_loader)
    elif tab == 'ci50-tab':
        return ci50_layout.create_layout(data_loader)
    elif tab == 'cinoadd-tab':
        return cinoadd_layout.create_layout(data_loader)
    elif tab == 'comparison-tab':
        return comparison_layout.create_layout(data_loader)
    elif tab == 'within-scenario-tab':
        return within_scenario_layout.create_within_scenario_layout(data_loader)
    return html.Div("Select a tab")

# Register all callbacks
register_callbacks(app, data_loader)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
