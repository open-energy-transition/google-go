"""
Callbacks for the dashboard
Handles all interactive components and updates
"""
from dash import Input, Output, State, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.colors import ColorMapper


def register_callbacks(app, data_loader):
    """Register all callbacks for the dashboard"""

    # Initialize color mapper
    color_mapper = ColorMapper(colors_csv_path="../results/colors.csv")

    # ==================== CI_25 Callbacks ====================
    @app.callback(
        [Output('ci25-carrier-selector', 'options'),
         Output('ci25-carrier-selector', 'value')],
        [Input('ci25-metric-selector', 'value')]
    )
    def update_ci25_carriers(metric):
        """Update available carriers based on selected metric"""
        if not metric:
            return [], []

        carriers = data_loader.get_carriers_for_metric('CI_25', metric)
        options = [{'label': c, 'value': c} for c in carriers]
        # Select all by default
        return options, carriers

    @app.callback(
        Output('ci25-main-plot', 'figure'),
        [Input('ci25-year-selector', 'value'),
         Input('ci25-scenario-selector', 'value'),
         Input('ci25-metric-selector', 'value'),
         Input('ci25-plot-type-selector', 'value'),
         Input('ci25-carrier-selector', 'value')]
    )
    def update_ci25_main_plot(year, scenario, metric, plot_type, carriers):
        """Update main plot for CI_25"""
        return create_plot('CI_25', year, scenario, metric, plot_type,
                          carriers, data_loader, color_mapper)

    @app.callback(
        Output('ci25-secondary-plot', 'figure'),
        [Input('ci25-year-selector', 'value'),
         Input('ci25-scenario-selector', 'value'),
         Input('ci25-metric-selector', 'value'),
         Input('ci25-carrier-selector', 'value')]
    )
    def update_ci25_secondary_plot(year, scenario, metric, carriers):
        """Update secondary plot showing carrier breakdown"""
        return create_pie_breakdown('CI_25', year, scenario, metric,
                                   carriers, data_loader, color_mapper)

    @app.callback(
        Output('ci25-summary-stats', 'children'),
        [Input('ci25-year-selector', 'value'),
         Input('ci25-scenario-selector', 'value'),
         Input('ci25-metric-selector', 'value'),
         Input('ci25-plot-type-selector', 'value')]
    )
    def update_ci25_summary(year, scenario, metric, plot_type):
        """Update summary statistics for CI_25"""
        # Use multi-year stats for stacked_bar and year_comparison
        if plot_type in ['stacked_bar', 'year_comparison']:
            return create_multi_year_summary_stats('CI_25', scenario, metric, data_loader)
        return create_summary_stats('CI_25', year, scenario, metric, data_loader)

    # ==================== CI_50 Callbacks ====================
    @app.callback(
        [Output('ci50-carrier-selector', 'options'),
         Output('ci50-carrier-selector', 'value')],
        [Input('ci50-metric-selector', 'value')]
    )
    def update_ci50_carriers(metric):
        """Update available carriers based on selected metric"""
        if not metric:
            return [], []

        carriers = data_loader.get_carriers_for_metric('CI_50', metric)
        options = [{'label': c, 'value': c} for c in carriers]
        return options, carriers

    @app.callback(
        Output('ci50-main-plot', 'figure'),
        [Input('ci50-year-selector', 'value'),
         Input('ci50-scenario-selector', 'value'),
         Input('ci50-metric-selector', 'value'),
         Input('ci50-plot-type-selector', 'value'),
         Input('ci50-carrier-selector', 'value')]
    )
    def update_ci50_main_plot(year, scenario, metric, plot_type, carriers):
        """Update main plot for CI_50"""
        return create_plot('CI_50', year, scenario, metric, plot_type,
                          carriers, data_loader, color_mapper)

    @app.callback(
        Output('ci50-secondary-plot', 'figure'),
        [Input('ci50-year-selector', 'value'),
         Input('ci50-scenario-selector', 'value'),
         Input('ci50-metric-selector', 'value'),
         Input('ci50-carrier-selector', 'value')]
    )
    def update_ci50_secondary_plot(year, scenario, metric, carriers):
        """Update secondary plot showing carrier breakdown"""
        return create_pie_breakdown('CI_50', year, scenario, metric,
                                   carriers, data_loader, color_mapper)

    @app.callback(
        Output('ci50-summary-stats', 'children'),
        [Input('ci50-year-selector', 'value'),
         Input('ci50-scenario-selector', 'value'),
         Input('ci50-metric-selector', 'value'),
         Input('ci50-plot-type-selector', 'value')]
    )
    def update_ci50_summary(year, scenario, metric, plot_type):
        """Update summary statistics for CI_50"""
        if plot_type in ['stacked_bar', 'year_comparison']:
            return create_multi_year_summary_stats('CI_50', scenario, metric, data_loader)
        return create_summary_stats('CI_50', year, scenario, metric, data_loader)

    # ==================== CI_noadd Callbacks ====================
    @app.callback(
        [Output('cinoadd-carrier-selector', 'options'),
         Output('cinoadd-carrier-selector', 'value')],
        [Input('cinoadd-metric-selector', 'value')]
    )
    def update_cinoadd_carriers(metric):
        """Update available carriers based on selected metric"""
        if not metric:
            return [], []

        carriers = data_loader.get_carriers_for_metric('CI_noadd', metric)
        options = [{'label': c, 'value': c} for c in carriers]
        return options, carriers

    @app.callback(
        Output('cinoadd-main-plot', 'figure'),
        [Input('cinoadd-year-selector', 'value'),
         Input('cinoadd-scenario-selector', 'value'),
         Input('cinoadd-metric-selector', 'value'),
         Input('cinoadd-plot-type-selector', 'value'),
         Input('cinoadd-carrier-selector', 'value')]
    )
    def update_cinoadd_main_plot(year, scenario, metric, plot_type, carriers):
        """Update main plot for CI_noadd"""
        return create_plot('CI_noadd', year, scenario, metric, plot_type,
                          carriers, data_loader, color_mapper)

    @app.callback(
        Output('cinoadd-secondary-plot', 'figure'),
        [Input('cinoadd-year-selector', 'value'),
         Input('cinoadd-scenario-selector', 'value'),
         Input('cinoadd-metric-selector', 'value'),
         Input('cinoadd-carrier-selector', 'value')]
    )
    def update_cinoadd_secondary_plot(year, scenario, metric, carriers):
        """Update secondary plot showing carrier breakdown"""
        return create_pie_breakdown('CI_noadd', year, scenario, metric,
                                   carriers, data_loader, color_mapper)

    @app.callback(
        Output('cinoadd-summary-stats', 'children'),
        [Input('cinoadd-year-selector', 'value'),
         Input('cinoadd-scenario-selector', 'value'),
         Input('cinoadd-metric-selector', 'value'),
         Input('cinoadd-plot-type-selector', 'value')]
    )
    def update_cinoadd_summary(year, scenario, metric, plot_type):
        """Update summary statistics for CI_noadd"""
        if plot_type in ['stacked_bar', 'year_comparison']:
            return create_multi_year_summary_stats('CI_noadd', scenario, metric, data_loader)
        return create_summary_stats('CI_noadd', year, scenario, metric, data_loader)

    # ==================== Comparison Callbacks ====================
    @app.callback(
        Output('comp-main-plot', 'figure'),
        [Input('comp-year-selector', 'value'),
         Input('comp-subscenario-selector', 'value'),
         Input('comp-metric-selector', 'value'),
         Input('comp-scenarios-selector', 'value')]
    )
    def update_comparison_plot(year, subscenario, metric, scenarios):
        """Update main comparison plot"""
        return create_comparison_plot(year, subscenario, metric, scenarios,
                                     data_loader, color_mapper)

    @app.callback(
        [Output('comp-ci25-plot', 'figure'),
         Output('comp-ci50-plot', 'figure'),
         Output('comp-cinoadd-plot', 'figure')],
        [Input('comp-year-selector', 'value'),
         Input('comp-subscenario-selector', 'value'),
         Input('comp-metric-selector', 'value')]
    )
    def update_comparison_details(year, subscenario, metric):
        """Update individual scenario detail plots"""
        fig1 = create_scenario_detail_plot('CI_25', year, subscenario, metric, data_loader, color_mapper)
        fig2 = create_scenario_detail_plot('CI_50', year, subscenario, metric, data_loader, color_mapper)
        fig3 = create_scenario_detail_plot('CI_noadd', year, subscenario, metric, data_loader, color_mapper)
        return fig1, fig2, fig3

    @app.callback(
        Output('comp-diff-plot', 'figure'),
        [Input('comp-year-selector', 'value'),
         Input('comp-subscenario-selector', 'value'),
         Input('comp-metric-selector', 'value'),
         Input('comp-scenarios-selector', 'value')]
    )
    def update_difference_plot(year, subscenario, metric, scenarios):
        """Update difference analysis plot"""
        return create_difference_plot(year, subscenario, metric, scenarios, data_loader, color_mapper)

    @app.callback(
        Output('comp-summary-table', 'children'),
        [Input('comp-year-selector', 'value'),
         Input('comp-subscenario-selector', 'value'),
         Input('comp-metric-selector', 'value'),
         Input('comp-scenarios-selector', 'value')]
    )
    def update_comparison_table(year, subscenario, metric, scenarios):
        """Update comparison summary table"""
        return create_comparison_table(year, subscenario, metric, scenarios, data_loader)


# ==================== Helper Functions ====================

def create_plot(scenario, year, scenario_name, metric, plot_type, carriers, data_loader, color_mapper):
    """Create a plot based on parameters"""
    try:
        # Get data
        df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

        if df is None or df.empty:
            return create_empty_figure("No data available")

        # Filter by carriers if specified (level 2 is carrier)
        if carriers:
            df = df[df.index.get_level_values(2).isin(carriers)]

        # Aggregate data (sum across all columns for the selected year/scenario)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns for plotting
            data_series = df.iloc[:, 0]
        else:
            data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

        # Create plot based on type
        if plot_type == 'bar':
            fig = create_bar_plot(data_series, metric, color_mapper)
        elif plot_type == 'stacked_bar':
            fig = create_stacked_bar_all_years(scenario, scenario_name, metric, carriers, data_loader, color_mapper)
        elif plot_type == 'year_comparison':
            fig = create_year_comparison_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper)
        elif plot_type == 'area':
            fig = create_area_plot(data_series, metric, color_mapper)
        elif plot_type == 'pie':
            fig = create_pie_plot(data_series, metric, color_mapper)
        else:
            fig = create_bar_plot(data_series, metric, color_mapper)

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_bar_plot(data, metric, color_mapper):
    """Create a bar plot"""
    # Use level 2 for carriers (index structure is: metric, ylabel, carrier)
    carriers = data.index.get_level_values(2).tolist()
    values = data.values

    colors = [color_mapper.get_color(c, metric) for c in carriers]

    fig = go.Figure(data=[
        go.Bar(x=carriers, y=values, marker_color=colors)
    ])

    fig.update_layout(
        title=f"{metric}",
        xaxis_title="Carrier",
        yaxis_title="Value",
        template="plotly_white",
        hovermode='x unified'
    )

    return fig


def create_area_plot(data, metric, color_mapper):
    """Create a stacked area plot (for time series)"""
    # This is a simplified version - would need time series data
    return create_bar_plot(data, metric, color_mapper)


def create_pie_plot(data, metric, color_mapper):
    """Create a pie chart"""
    # Use level 2 for carriers (index structure is: metric, ylabel, carrier)
    carriers = data.index.get_level_values(2).tolist()
    values = data.values

    # Filter out negative or zero values for pie chart
    mask = values > 0
    carriers = [c for c, m in zip(carriers, mask) if m]
    values = values[mask]

    colors = [color_mapper.get_color(c, metric) for c in carriers]

    fig = go.Figure(data=[
        go.Pie(labels=carriers, values=values, marker_colors=colors)
    ])

    fig.update_layout(
        title=f"{metric} Distribution",
        template="plotly_white"
    )

    return fig


def create_pie_breakdown(scenario, year, scenario_name, metric, carriers, data_loader, color_mapper):
    """Create pie chart breakdown"""
    try:
        df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

        if df is None or df.empty:
            return create_empty_figure("No data available")

        # Filter by carriers if specified (level 2 is carrier)
        if carriers:
            df = df[df.index.get_level_values(2).isin(carriers)]

        if isinstance(df.columns, pd.MultiIndex):
            data_series = df.iloc[:, 0]
        else:
            data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

        return create_pie_plot(data_series, metric, color_mapper)

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_summary_stats(scenario, year, scenario_name, metric, data_loader):
    """Create comprehensive summary statistics HTML"""
    try:
        df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

        if df is None or df.empty:
            return html.P("No data available", className="text-muted")

        if isinstance(df.columns, pd.MultiIndex):
            data_series = df.iloc[:, 0]
        else:
            data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

        # Get carriers
        carriers = data_series.index.get_level_values(2).tolist()
        values = data_series.values

        # Calculate statistics
        total = values.sum()
        mean = values.mean()
        std = values.std()
        median = np.median(values)
        min_val = values.min()
        max_val = values.max()

        # Find top contributors
        positive_mask = values > 0
        if positive_mask.any():
            positive_values = values[positive_mask]
            positive_carriers = [c for c, m in zip(carriers, positive_mask) if m]
            top_indices = np.argsort(positive_values)[-5:][::-1]
            top_carriers = [(positive_carriers[i], positive_values[i]) for i in top_indices if i < len(positive_carriers)]
        else:
            top_carriers = []

        # Create styled summary with cards
        return html.Div([
            # Key metrics row
            html.Div([
                html.Div([
                    html.H5("Total", className="text-muted mb-1"),
                    html.H3(f"{total:.2f}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5("Mean", className="text-muted mb-1"),
                    html.H3(f"{mean:.2f}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5("Max", className="text-muted mb-1"),
                    html.H3(f"{max_val:.2f}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5("Carriers", className="text-muted mb-1"),
                    html.H3(f"{len(carriers)}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
            ], className="row mb-3"),

            # Detailed statistics
            html.Div([
                html.Div([
                    html.H6("Statistical Summary", className="mb-2", style={'fontWeight': 'bold'}),
                    html.Table([
                        html.Tbody([
                            html.Tr([html.Td("Median:", style={'fontWeight': 'bold'}), html.Td(f"{median:.2f}")]),
                            html.Tr([html.Td("Std Dev:", style={'fontWeight': 'bold'}), html.Td(f"{std:.2f}")]),
                            html.Tr([html.Td("Min:", style={'fontWeight': 'bold'}), html.Td(f"{min_val:.2f}")]),
                            html.Tr([html.Td("Range:", style={'fontWeight': 'bold'}), html.Td(f"{max_val - min_val:.2f}")]),
                        ])
                    ], className="table table-sm")
                ], className="col-md-6"),

                html.Div([
                    html.H6("Top Contributors", className="mb-2", style={'fontWeight': 'bold'}),
                    html.Ol([
                        html.Li(f"{carrier}: {value:.2f} ({value/total*100:.1f}%)")
                        for carrier, value in top_carriers
                    ]) if top_carriers else html.P("No positive values", className="text-muted")
                ], className="col-md-6"),
            ], className="row")
        ])

    except Exception as e:
        return html.P(f"Error: {str(e)}")


def create_multi_year_summary_stats(scenario, scenario_name, metric, data_loader):
    """Create summary statistics for multi-year comparisons"""
    try:
        stats = data_loader.get_summary_stats(scenario)
        years = stats.get('years', [])

        if not years or not scenario_name or not metric:
            return html.P("No data available", className="text-muted")

        # Collect data for all years
        year_totals = {}
        year_carrier_counts = {}
        all_carriers = set()

        for year in years:
            df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                data_series = df.iloc[:, 0]
            else:
                data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

            carriers = data_series.index.get_level_values(2).tolist()
            values = data_series.values

            year_totals[year] = values.sum()
            year_carrier_counts[year] = len([v for v in values if v > 0])
            all_carriers.update(carriers)

        if not year_totals:
            return html.P("No data available", className="text-muted")

        # Calculate growth rates
        sorted_years = sorted(year_totals.keys())
        if len(sorted_years) > 1:
            first_year_total = year_totals[sorted_years[0]]
            last_year_total = year_totals[sorted_years[-1]]
            total_growth = ((last_year_total - first_year_total) / first_year_total * 100) if first_year_total != 0 else 0
            years_span = sorted_years[-1] - sorted_years[0]
            annual_growth = total_growth / years_span if years_span > 0 else 0
        else:
            total_growth = 0
            annual_growth = 0

        # Create summary cards
        return html.Div([
            # Key metrics row
            html.Div([
                html.Div([
                    html.H5(f"{sorted_years[0]} Total", className="text-muted mb-1"),
                    html.H3(f"{year_totals[sorted_years[0]]:.2f}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#e3f2fd', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5(f"{sorted_years[-1]} Total", className="text-muted mb-1"),
                    html.H3(f"{year_totals[sorted_years[-1]]:.2f}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#e8f5e9', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5("Total Growth", className="text-muted mb-1"),
                    html.H3(f"{total_growth:+.1f}%", className="mb-0",
                           style={'color': 'green' if total_growth > 0 else 'red' if total_growth < 0 else 'gray'})
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#fff3e0', 'borderRadius': '5px', 'margin': '5px'}),

                html.Div([
                    html.H5("Unique Carriers", className="text-muted mb-1"),
                    html.H3(f"{len(all_carriers)}", className="mb-0")
                ], className="col-md-3 text-center p-2", style={'backgroundColor': '#f3e5f5', 'borderRadius': '5px', 'margin': '5px'}),
            ], className="row mb-3"),

            # Year-by-year breakdown
            html.Div([
                html.Div([
                    html.H6("Year-by-Year Totals", className="mb-2", style={'fontWeight': 'bold'}),
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Year"),
                                html.Th("Total"),
                                html.Th("Active Carriers"),
                                html.Th("Change")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(str(year)),
                                html.Td(f"{year_totals[year]:.2f}"),
                                html.Td(str(year_carrier_counts.get(year, 0))),
                                html.Td(
                                    f"{((year_totals[year] - year_totals[sorted_years[i-1]]) / year_totals[sorted_years[i-1]] * 100):+.1f}%"
                                    if i > 0 and year_totals[sorted_years[i-1]] != 0 else "-",
                                    style={'color': 'green' if i > 0 and year_totals[year] > year_totals[sorted_years[i-1]]
                                           else 'red' if i > 0 and year_totals[year] < year_totals[sorted_years[i-1]]
                                           else 'gray'}
                                )
                            ])
                            for i, year in enumerate(sorted_years)
                        ])
                    ], className="table table-sm table-striped")
                ], className="col-md-8"),

                html.Div([
                    html.H6("Growth Metrics", className="mb-2", style={'fontWeight': 'bold'}),
                    html.Table([
                        html.Tbody([
                            html.Tr([html.Td("Annual Avg Growth:", style={'fontWeight': 'bold'}),
                                   html.Td(f"{annual_growth:+.1f}%/year")]),
                            html.Tr([html.Td("Time Span:", style={'fontWeight': 'bold'}),
                                   html.Td(f"{years_span} years")]),
                            html.Tr([html.Td("Total Change:", style={'fontWeight': 'bold'}),
                                   html.Td(f"{last_year_total - first_year_total:+.2f}")]),
                        ])
                    ], className="table table-sm")
                ], className="col-md-4"),
            ], className="row")
        ])

    except Exception as e:
        return html.P(f"Error: {str(e)}")


def create_comparison_plot(year, subscenario, metric, scenarios, data_loader, color_mapper):
    """Create comparison plot across scenarios"""
    try:
        print(f"\n=== DEBUG: create_comparison_plot called ===")
        print(f"Year: {year}")
        print(f"Subscenario: {subscenario}")
        print(f"Metric: {metric}")
        print(f"Scenarios parameter: {scenarios}")
        print(f"Scenarios type: {type(scenarios)}")

        if not scenarios or not year or not metric or not subscenario:
            return create_empty_figure("Please select year, sub-scenario, metric, and at least one scenario")

        fig = go.Figure()
        scenarios_with_data = []

        for scenario in scenarios:
            print(f"\n--- Processing scenario: {scenario} ---")
            # Check if this sub-scenario exists in this main scenario
            stats = data_loader.get_summary_stats(scenario)
            available_subscenarios = stats.get('scenarios', [])
            print(f"Available subscenarios in {scenario}: {available_subscenarios}")

            if subscenario not in available_subscenarios:
                # Skip this scenario if the sub-scenario doesn't exist
                print(f"Skipping {scenario} - subscenario '{subscenario}' not found")
                continue

            print(f"Fetching data for {scenario}...")
            # Use the selected sub-scenario for comparison
            df = data_loader.get_data(scenario, year=year, scenario_name=subscenario, metric=metric)
            print(f"Data shape: {df.shape if df is not None else 'None'}")

            if df is None or df.empty:
                print(f"Data is None or empty for {scenario}")
                continue

            scenarios_with_data.append(scenario)
            print(f"Successfully added {scenario} to scenarios_with_data")

            if isinstance(df.columns, pd.MultiIndex):
                data_series = df.iloc[:, 0]
            else:
                data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

            # Use level 2 for carriers (index structure is: metric, ylabel, carrier)
            carriers = data_series.index.get_level_values(2).tolist()
            values = data_series.values
            print(f"Carriers: {carriers[:3]}... (showing first 3)")
            print(f"Values: {values[:3]}... (showing first 3)")

            fig.add_trace(go.Bar(
                name=scenario,
                x=carriers,
                y=values
            ))
            print(f"Added trace for {scenario}")

        if not scenarios_with_data:
            print(f"No scenarios with data found!")
            return create_empty_figure(f"Sub-scenario '{subscenario}' not found in any selected main scenarios")

        print(f"\nFinal scenarios_with_data: {scenarios_with_data}")
        print(f"Total traces in figure: {len(fig.data)}")

        fig.update_layout(
            title=f"Comparison: {metric} ({year}) - {subscenario}<br><sub>Showing: {', '.join(scenarios_with_data)}</sub>",
            xaxis_title="Carrier",
            yaxis_title="Value",
            template="plotly_white",
            barmode='group',
            hovermode='x unified'
        )

        print(f"=== DEBUG END ===\n")
        return fig

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_empty_figure(f"Error: {str(e)}")


def create_scenario_detail_plot(scenario, year, subscenario, metric, data_loader, color_mapper):
    """Create detail plot for a single scenario"""
    try:
        if not subscenario or not year or not metric:
            return create_empty_figure("No data available")

        # Use the selected sub-scenario
        df = data_loader.get_data(scenario, year=year, scenario_name=subscenario, metric=metric)

        if df is None or df.empty:
            return create_empty_figure("No data available")

        if isinstance(df.columns, pd.MultiIndex):
            data_series = df.iloc[:, 0]
        else:
            data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

        return create_pie_plot(data_series, metric, color_mapper)

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_difference_plot(year, subscenario, metric, scenarios, data_loader, color_mapper):
    """Create difference analysis plot"""
    try:
        if len(scenarios) < 2:
            return create_empty_figure("Select at least 2 scenarios to compare")

        if not subscenario:
            return create_empty_figure("Please select a sub-scenario")

        # Get data for all scenarios
        scenario_data = {}
        for scenario in scenarios:
            df = data_loader.get_data(scenario, year=year, scenario_name=subscenario, metric=metric)

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    scenario_data[scenario] = df.iloc[:, 0]
                else:
                    scenario_data[scenario] = df.iloc[:, 0] if len(df.columns) > 0 else df

        if len(scenario_data) < 2:
            return create_empty_figure("Not enough data for comparison")

        # Calculate differences (relative to first scenario)
        base_scenario = scenarios[0]
        base_data = scenario_data[base_scenario]

        fig = go.Figure()

        for scenario in scenarios[1:]:
            if scenario in scenario_data:
                diff = scenario_data[scenario] - base_data
                carriers = diff.index.get_level_values('carrier').tolist()

                fig.add_trace(go.Bar(
                    name=f"{scenario} - {base_scenario}",
                    x=carriers,
                    y=diff.values
                ))

        fig.update_layout(
            title=f"Difference Analysis: {metric} ({year}) - {subscenario}",
            xaxis_title="Carrier",
            yaxis_title=f"Difference from {base_scenario}",
            template="plotly_white",
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_comparison_table(year, subscenario, metric, scenarios, data_loader):
    """Create comparison table"""
    try:
        if not scenarios or not year or not metric or not subscenario:
            return html.P("Please select year, sub-scenario, metric, and scenarios")

        # Collect data from all scenarios
        table_data = []

        for scenario in scenarios:
            df = data_loader.get_data(scenario, year=year, scenario_name=subscenario, metric=metric)

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                data_series = df.iloc[:, 0]
            else:
                data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

            table_data.append({
                'Scenario': scenario,
                'Total': f"{data_series.sum():.2f}",
                'Mean': f"{data_series.mean():.2f}",
                'Max': f"{data_series.max():.2f}",
                'Carriers': len(data_series)
            })

        if not table_data:
            return html.P("No data available")

        # Create table
        df_table = pd.DataFrame(table_data)

        return html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in df_table.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df_table.iloc[i][col]) for col in df_table.columns
                ]) for i in range(len(df_table))
            ])
        ], style={'width': '100%', 'textAlign': 'center'})

    except Exception as e:
        return html.P(f"Error: {str(e)}")


def create_stacked_bar_all_years(scenario, scenario_name, metric, carriers, data_loader, color_mapper):
    """Create stacked bar plot comparing all years for a scenario"""
    try:
        stats = data_loader.get_summary_stats(scenario)
        years = stats.get('years', [])

        if not years or not scenario_name or not metric:
            return create_empty_figure("Select scenario and metric")

        fig = go.Figure()

        # Get data for each year
        for year in years:
            df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

            if df is None or df.empty:
                continue

            # Filter by carriers if specified
            if carriers:
                df = df[df.index.get_level_values(2).isin(carriers)]

            if isinstance(df.columns, pd.MultiIndex):
                data_series = df.iloc[:, 0]
            else:
                data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

            # Get carriers and values
            year_carriers = data_series.index.get_level_values(2).tolist()
            values = data_series.values

            # Add stacked bars for this year
            for i, carrier in enumerate(year_carriers):
                color = color_mapper.get_color(carrier, metric)

                fig.add_trace(go.Bar(
                    name=carrier,
                    x=[str(year)],
                    y=[values[i]],
                    marker_color=color,
                    legendgroup=carrier,
                    showlegend=(year == years[0]),  # Only show legend for first year
                    hovertemplate=f"{carrier}: %{{y:.2f}}<extra></extra>"
                ))

        fig.update_layout(
            title=f"{metric} - All Years Comparison",
            xaxis_title="Year",
            yaxis_title="Value",
            barmode='stack',
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                title="Carrier",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_year_comparison_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper):
    """Create grouped bar plot comparing carriers across years"""
    try:
        stats = data_loader.get_summary_stats(scenario)
        years = stats.get('years', [])

        if not years or not scenario_name or not metric:
            return create_empty_figure("Select scenario and metric")

        # Collect data for all years
        year_data = {}
        all_carriers = set()

        for year in years:
            df = data_loader.get_data(scenario, year=year, scenario_name=scenario_name, metric=metric)

            if df is None or df.empty:
                continue

            # Filter by carriers if specified
            if carriers:
                df = df[df.index.get_level_values(2).isin(carriers)]

            if isinstance(df.columns, pd.MultiIndex):
                data_series = df.iloc[:, 0]
            else:
                data_series = df.iloc[:, 0] if len(df.columns) > 0 else df

            year_carriers = data_series.index.get_level_values(2).tolist()
            all_carriers.update(year_carriers)
            year_data[year] = {carrier: val for carrier, val in zip(year_carriers, data_series.values)}

        # Create grouped bar plot
        fig = go.Figure()

        for year in years:
            carrier_vals = []
            for carrier in sorted(all_carriers):
                carrier_vals.append(year_data.get(year, {}).get(carrier, 0))

            fig.add_trace(go.Bar(
                name=str(year),
                x=sorted(list(all_carriers)),
                y=carrier_vals,
                hovertemplate=f"{year}: %{{y:.2f}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"{metric} - Year Comparison",
            xaxis_title="Carrier",
            yaxis_title="Value",
            barmode='group',
            template="plotly_white",
            hovermode='x unified',
            legend=dict(title="Year")
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_empty_figure(message):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        template="plotly_white"
    )
    return fig
