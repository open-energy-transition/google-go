"""
Callbacks for the dashboard
Handles all interactive components and updates
"""
from dash import Input, Output, State, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.colors import ColorMapper, format_scenario_name, format_main_scenario_name


def register_callbacks(app, data_loader):
    """Register all callbacks for the dashboard"""
    print("\n=== REGISTER_CALLBACKS CALLED ===\n")

    # Initialize color mapper
    color_mapper = ColorMapper(colors_csv_path="../results/colors.csv")

    # ==================== Single Scenario Callbacks ====================
    @app.callback(
        [Output('single-scenario-selector', 'options'),
         Output('single-scenario-selector', 'value')],
        [Input('single-main-scenario-selector', 'value')]
    )
    def update_single_subscenarios(main_scenario):
        """Update available sub-scenarios based on selected main scenario"""
        if not main_scenario:
            return [], None

        stats = data_loader.get_summary_stats(main_scenario)
        scenarios = stats.get('scenarios', [])
        options = [{'label': format_scenario_name(s), 'value': s} for s in scenarios]
        return options, scenarios[0] if scenarios else None

    @app.callback(
        [Output('single-carrier-selector', 'options'),
         Output('single-carrier-selector', 'value')],
        [Input('single-main-scenario-selector', 'value'),
         Input('single-metric-selector', 'value')]
    )
    def update_single_carriers(main_scenario, metric):
        """Update available carriers based on selected main scenario and metric"""
        if not main_scenario or not metric:
            return [], []

        carriers = data_loader.get_carriers_for_metric(main_scenario, metric)
        options = [{'label': c, 'value': c} for c in carriers]
        # Select all by default
        return options, carriers

    @app.callback(
        Output('single-main-plot', 'figure'),
        [Input('single-main-scenario-selector', 'value'),
         Input('single-year-selector', 'value'),
         Input('single-scenario-selector', 'value'),
         Input('single-metric-selector', 'value'),
         Input('single-plot-type-selector', 'value'),
         Input('single-carrier-selector', 'value')]
    )
    def update_single_main_plot(main_scenario, year, scenario, metric, plot_type, carriers):
        """Update main plot for single scenario analysis"""
        if not main_scenario:
            return create_empty_figure("Please select a main scenario")
        return create_plot(main_scenario, year, scenario, metric, plot_type,
                          carriers, data_loader, color_mapper)

    @app.callback(
        Output('single-secondary-plot', 'figure'),
        [Input('single-main-scenario-selector', 'value'),
         Input('single-year-selector', 'value'),
         Input('single-scenario-selector', 'value'),
         Input('single-metric-selector', 'value'),
         Input('single-carrier-selector', 'value')]
    )
    def update_single_secondary_plot(main_scenario, year, scenario, metric, carriers):
        """Update secondary plot showing carrier breakdown"""
        if not main_scenario:
            return create_empty_figure("Please select a main scenario")
        return create_pie_breakdown(main_scenario, year, scenario, metric,
                                   carriers, data_loader, color_mapper)

    @app.callback(
        Output('single-summary-stats', 'children'),
        [Input('single-main-scenario-selector', 'value'),
         Input('single-year-selector', 'value'),
         Input('single-scenario-selector', 'value'),
         Input('single-metric-selector', 'value'),
         Input('single-plot-type-selector', 'value')]
    )
    def update_single_summary(main_scenario, year, scenario, metric, plot_type):
        """Update summary statistics for single scenario analysis"""
        if not main_scenario:
            return html.P("Please select a main scenario")
        # Use multi-year stats for stacked_bar, year_comparison, and year_on_year_evolution
        if plot_type in ['stacked_bar', 'year_comparison', 'year_on_year_evolution']:
            return create_multi_year_summary_stats(main_scenario, scenario, metric, data_loader)
        return create_summary_stats(main_scenario, year, scenario, metric, data_loader)

    # ==================== Cross-Scenario Comparison Callbacks ====================
    # Helper to find which main scenario a sub-scenario belongs to
    def find_main_scenario(subscenario):
        """Find which main scenario contains this sub-scenario"""
        print(f"!!! FIND_MAIN_SCENARIO called for: {subscenario}")
        for main in ['CI_25', 'CI_50', 'CI_noadd']:
            stats = data_loader.get_summary_stats(main)
            if subscenario in stats.get('scenarios', []):
                print(f"Found {subscenario} in {main}")
                return main
        print(f"WARNING: {subscenario} not found in any main scenario!")
        return None

    @app.callback(
        Output('cross-comparison-plot', 'figure'),
        [Input('cross-year-selector', 'value'),
         Input('cross-metric-selector', 'value'),
         Input('cross-plot-type-selector', 'value'),
         Input('cross-subscenario1-selector', 'value'),
         Input('cross-subscenario2-selector', 'value')]
    )
    def update_cross_comparison(year, metric, plot_type, subscenario1, subscenario2):
        """Update cross-scenario comparison plot"""
        print(f"\n!!! UPDATE_CROSS_COMPARISON CALLED !!!")
        print(f"Params: {year}, {metric}, {plot_type}, {subscenario1}, {subscenario2}")

        main1 = find_main_scenario(subscenario1) if subscenario1 else None
        main2 = find_main_scenario(subscenario2) if subscenario2 else None
        print(f"Main scenarios: {subscenario1} -> {main1}, {subscenario2} -> {main2}")

        return create_cross_comparison_plot(main1 or main2, year, metric, plot_type, subscenario1, subscenario2,
                                            data_loader, color_mapper, main1, main2)

    @app.callback(
        Output('cross-difference-plot', 'figure'),
        [Input('cross-year-selector', 'value'),
         Input('cross-metric-selector', 'value'),
         Input('cross-subscenario1-selector', 'value'),
         Input('cross-subscenario2-selector', 'value')]
    )
    def update_cross_difference(year, metric, subscenario1, subscenario2):
        """Update cross-scenario difference plot"""
        main1 = find_main_scenario(subscenario1) if subscenario1 else None
        main2 = find_main_scenario(subscenario2) if subscenario2 else None
        return create_cross_difference_plot(main1 or main2, year, metric, subscenario1, subscenario2,
                                           data_loader, color_mapper, main1, main2)

    @app.callback(
        Output('cross-summary-stats', 'children'),
        [Input('cross-year-selector', 'value'),
         Input('cross-metric-selector', 'value'),
         Input('cross-subscenario1-selector', 'value'),
         Input('cross-subscenario2-selector', 'value')]
    )
    def update_cross_summary(year, metric, subscenario1, subscenario2):
        """Update cross-scenario summary statistics"""
        main1 = find_main_scenario(subscenario1) if subscenario1 else None
        main2 = find_main_scenario(subscenario2) if subscenario2 else None
        return create_cross_summary_stats(main1 or main2, year, metric, subscenario1, subscenario2,
                                          data_loader, main1, main2)


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
        elif plot_type == 'year_on_year_evolution':
            fig = create_year_on_year_evolution_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper)
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


def create_year_on_year_evolution_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper):
    """Create line plot showing year on year evolution of technologies"""
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

        # Create line plot - each carrier is a line
        fig = go.Figure()

        for carrier in sorted(all_carriers):
            carrier_years = []
            carrier_values = []

            for year in sorted(years):
                if carrier in year_data.get(year, {}):
                    carrier_years.append(year)
                    carrier_values.append(year_data[year][carrier])

            # Get color for this carrier
            color = color_mapper.get_color(carrier, metric) if color_mapper else None

            fig.add_trace(go.Scatter(
                x=carrier_years,
                y=carrier_values,
                mode='lines+markers',
                name=carrier,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                hovertemplate=f"{carrier}: %{{y:.2f}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"Year on Year Evolution: {metric}<br><sub>{format_main_scenario_name(scenario)} - {format_scenario_name(scenario_name)}</sub>",
            xaxis_title="Year",
            yaxis_title="Value",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(title="Technology", yanchor="top", y=0.99, xanchor="left", x=1.01)
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


# ==================== Helper Functions ====================

def create_cross_comparison_plot(main_scenario, year, metric, plot_type, subscenario1, subscenario2, data_loader, color_mapper, main1=None, main2=None):
    """Create comparison plot for two sub-scenarios"""
    try:
        print(f"\n=== Cross-Scenario Comparison Debug ===")
        print(f"Year: {year}")
        print(f"Metric: {metric}")
        print(f"Plot type: {plot_type}")
        print(f"Subscenario1: {subscenario1} (from {main1})")
        print(f"Subscenario2: {subscenario2} (from {main2})")

        if not all([metric, subscenario1, subscenario2]):
            print("Missing parameters!")
            return create_empty_figure("Please select all parameters")

        if subscenario1 == subscenario2:
            print("Same subscenarios selected!")
            return create_empty_figure("Please select two different sub-scenarios")

        # Handle different plot types
        if plot_type == 'year_comparison':
            return create_cross_year_comparison_plot(main1 or main_scenario, main2 or main_scenario,
                                                     metric, subscenario1, subscenario2,
                                                     data_loader, color_mapper, main1, main2)
        elif plot_type == 'year_on_year_evolution':
            return create_cross_evolution_plot(main1 or main_scenario, main2 or main_scenario,
                                               metric, subscenario1, subscenario2,
                                               data_loader, color_mapper, main1, main2)

        # Default: side-by-side comparison
        if not year:
            return create_empty_figure("Please select a year")

        fig = go.Figure()

        # Get data for both sub-scenarios from their respective main scenarios
        df1 = data_loader.get_data(main1 or main_scenario, year=year, scenario_name=subscenario1, metric=metric)
        df2 = data_loader.get_data(main2 or main_scenario, year=year, scenario_name=subscenario2, metric=metric)

        print(f"df1 shape: {df1.shape if df1 is not None else 'None'}")
        print(f"df2 shape: {df2.shape if df2 is not None else 'None'}")

        if df1 is None or df1.empty or df2 is None or df2.empty:
            print("Data is None or empty!")
            return create_empty_figure("No data available for selected parameters")

        # Extract data series
        data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
        data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]

        carriers = data1.index.get_level_values(2).tolist()
        values1 = data1.values
        values2 = data2.values

        # Add bars for both scenarios
        fig.add_trace(go.Bar(
            name=format_scenario_name(subscenario1),
            x=carriers,
            y=values1,
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            name=format_scenario_name(subscenario2),
            x=carriers,
            y=values2,
            marker_color='lightcoral'
        ))

        fig.update_layout(
            title=f"{format_main_scenario_name(main_scenario)}: {metric} ({year})<br><sub>{format_scenario_name(subscenario1)} vs {format_scenario_name(subscenario2)}</sub>",
            xaxis_title="Carrier",
            yaxis_title="Value",
            template="plotly_white",
            barmode='group',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_cross_difference_plot(main_scenario, year, metric, subscenario1, subscenario2, data_loader, color_mapper, main1=None, main2=None):
    """Create difference plot between two sub-scenarios"""
    try:
        if not all([year, metric, subscenario1, subscenario2]):
            return create_empty_figure("Please select all parameters")

        if subscenario1 == subscenario2:
            return create_empty_figure("Please select two different sub-scenarios")

        # Get data for both sub-scenarios from their respective main scenarios
        df1 = data_loader.get_data(main1 or main_scenario, year=year, scenario_name=subscenario1, metric=metric)
        df2 = data_loader.get_data(main2 or main_scenario, year=year, scenario_name=subscenario2, metric=metric)

        if df1 is None or df1.empty or df2 is None or df2.empty:
            return create_empty_figure("No data available for selected parameters")

        # Extract data series
        data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
        data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]

        carriers = data1.index.get_level_values(2).tolist()
        diff = data2.values - data1.values

        # Color bars based on positive/negative
        colors = ['green' if d > 0 else 'red' for d in diff]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=carriers,
            y=diff,
            marker_color=colors,
            text=[f"{d:+.2f}" for d in diff],
            textposition='outside'
        ))

        fig.update_layout(
            title=f"Difference: {format_scenario_name(subscenario2)} - {format_scenario_name(subscenario1)}<br><sub>{format_main_scenario_name(main_scenario)}: {metric} ({year})</sub>",
            xaxis_title="Carrier",
            yaxis_title="Difference (Positive = Higher in Scenario 2)",
            template="plotly_white",
            hovermode='x unified'
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_cross_summary_stats(main_scenario, year, metric, subscenario1, subscenario2, data_loader, main1=None, main2=None):
    """Create summary statistics comparing two sub-scenarios"""
    try:
        if not all([year, metric, subscenario1, subscenario2]):
            return html.P("Please select all parameters")

        if subscenario1 == subscenario2:
            return html.P("Please select two different sub-scenarios")

        # Get data for both sub-scenarios from their respective main scenarios
        df1 = data_loader.get_data(main1 or main_scenario, year=year, scenario_name=subscenario1, metric=metric)
        df2 = data_loader.get_data(main2 or main_scenario, year=year, scenario_name=subscenario2, metric=metric)

        if df1 is None or df1.empty or df2 is None or df2.empty:
            return html.P("No data available for selected parameters")

        # Extract data series
        data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
        data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]

        # Calculate basic statistics
        total1 = data1.sum()
        total2 = data2.sum()
        diff = total2 - total1
        pct_change = (diff / total1 * 100) if total1 != 0 else 0

        # Get carriers
        carriers1 = data1.index.get_level_values(2).tolist()
        values1 = data1.values
        values2 = data2.values

        # Find carriers with biggest absolute differences
        abs_diffs = np.abs(values2 - values1)
        top_diff_indices = np.argsort(abs_diffs)[-3:][::-1]  # Top 3

        # Find carriers with biggest percentage changes
        pct_changes = np.where(values1 != 0, ((values2 - values1) / values1) * 100, 0)

        # Filter to only carriers with significant absolute values (> 1% of total)
        significant_mask = np.abs(values1) > (abs(total1) * 0.01)
        significant_indices = np.where(significant_mask)[0]

        if len(significant_indices) > 0:
            significant_pct_changes = pct_changes[significant_indices]
            significant_carriers = [carriers1[i] for i in significant_indices]
            top_pct_indices = significant_indices[np.argsort(np.abs(significant_pct_changes))[-3:][::-1]]
        else:
            top_pct_indices = []

        # Count carriers
        num_carriers = len(carriers1)
        num_increased = np.sum(values2 > values1)
        num_decreased = np.sum(values2 < values1)
        num_unchanged = np.sum(values2 == values1)

        return html.Div([
            # Overall totals
            html.Div([
                html.Div([
                    html.H5(format_scenario_name(subscenario1), className="text-primary mb-1"),
                    html.H3(f"{total1:,.2f}", className="mb-0")
                ], className="col-md-4 text-center p-3", style={'backgroundColor': '#e3f2fd', 'borderRadius': '8px', 'margin': '5px'}),

                html.Div([
                    html.H5(format_scenario_name(subscenario2), className="text-success mb-1"),
                    html.H3(f"{total2:,.2f}", className="mb-0")
                ], className="col-md-4 text-center p-3", style={'backgroundColor': '#e8f5e9', 'borderRadius': '8px', 'margin': '5px'}),

                html.Div([
                    html.H5("Difference", className="mb-1", style={'color': '#666'}),
                    html.H3(f"{diff:+,.2f}", className="mb-0",
                           style={'color': 'green' if diff > 0 else 'red' if diff < 0 else 'gray'}),
                    html.P(f"({pct_change:+.1f}%)", className="mb-0", style={'fontSize': '14px'})
                ], className="col-md-4 text-center p-3", style={'backgroundColor': '#fff3e0', 'borderRadius': '8px', 'margin': '5px'}),
            ], className="row mb-3"),

            # Key insights as bullet points
            html.Div([
                html.H6("Key Insights:", style={'fontWeight': 'bold', 'marginBottom': '15px'}),
                html.Ul([
                    html.Li([
                        html.Strong("Overall change: "),
                        html.Span(f"{format_scenario_name(subscenario2)} is ", style={'color': '#666'}),
                        html.Span(f"{abs(pct_change):.1f}% {'higher' if pct_change > 0 else 'lower' if pct_change < 0 else 'the same'}",
                                 style={'color': 'green' if pct_change > 0 else 'red' if pct_change < 0 else 'gray', 'fontWeight': 'bold'}),
                        html.Span(f" than {format_scenario_name(subscenario1)}", style={'color': '#666'})
                    ]),
                    html.Li([
                        html.Strong("Carrier changes: "),
                        html.Span(f"{num_increased} increased", style={'color': 'green', 'marginRight': '10px'}),
                        html.Span(f"{num_decreased} decreased", style={'color': 'red', 'marginRight': '10px'}),
                        html.Span(f"{num_unchanged} unchanged", style={'color': 'gray'})
                    ]),
                    html.Li([
                        html.Strong("Largest absolute changes: "),
                        html.Ul([
                            html.Li([
                                html.Span(f"{carriers1[idx]}: ", style={'fontFamily': 'monospace'}),
                                html.Span(f"{values2[idx] - values1[idx]:+,.2f}",
                                         style={'color': 'green' if values2[idx] > values1[idx] else 'red' if values2[idx] < values1[idx] else 'gray'})
                            ])
                            for idx in top_diff_indices if abs(values2[idx] - values1[idx]) > 0.01
                        ], style={'marginTop': '5px', 'marginBottom': '5px'})
                    ]),
                    html.Li([
                        html.Strong("Largest percentage changes: "),
                        html.Ul([
                            html.Li([
                                html.Span(f"{carriers1[idx]}: ", style={'fontFamily': 'monospace'}),
                                html.Span(f"{pct_changes[idx]:+.1f}%",
                                         style={'color': 'green' if pct_changes[idx] > 0 else 'red' if pct_changes[idx] < 0 else 'gray'}),
                                html.Span(f" ({values1[idx]:.1f} → {values2[idx]:.1f})",
                                         style={'color': '#888', 'fontSize': '12px', 'marginLeft': '5px'})
                            ])
                            for idx in top_pct_indices if abs(pct_changes[idx]) > 0.1
                        ] if len(top_pct_indices) > 0 else [html.Li("No significant changes", style={'color': '#888'})],
                        style={'marginTop': '5px'})
                    ])
                ], style={'lineHeight': '1.8'})
            ], className="mt-3")
        ])

    except Exception as e:
        return html.P(f"Error: {str(e)}")


def create_cross_year_comparison_plot(main_scenario1, main_scenario2, metric, subscenario1, subscenario2, data_loader, color_mapper, main1=None, main2=None):
    """Create year comparison plot for two sub-scenarios (grouped bars across years)"""
    try:
        if not all([metric, subscenario1, subscenario2]):
            return create_empty_figure("Please select all parameters")

        if subscenario1 == subscenario2:
            return create_empty_figure("Please select two different sub-scenarios")

        # Get available years
        stats1 = data_loader.get_summary_stats(main1 or main_scenario1)
        years = stats1.get('years', [])

        if not years:
            return create_empty_figure("No years available")

        # Collect data for all years for both subscenarios
        all_carriers = set()
        year_data1 = {}
        year_data2 = {}

        for year in years:
            df1 = data_loader.get_data(main1 or main_scenario1, year=year, scenario_name=subscenario1, metric=metric)
            df2 = data_loader.get_data(main2 or main_scenario2, year=year, scenario_name=subscenario2, metric=metric)

            if df1 is not None and not df1.empty:
                data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
                carriers1 = data1.index.get_level_values(2).tolist()
                all_carriers.update(carriers1)
                year_data1[year] = {carrier: val for carrier, val in zip(carriers1, data1.values)}

            if df2 is not None and not df2.empty:
                data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]
                carriers2 = data2.index.get_level_values(2).tolist()
                all_carriers.update(carriers2)
                year_data2[year] = {carrier: val for carrier, val in zip(carriers2, data2.values)}

        if not all_carriers:
            return create_empty_figure("No data available")

        # Create grouped bar plot
        fig = go.Figure()

        # Add traces for subscenario1 across all years
        for year in years:
            carrier_vals = []
            for carrier in sorted(all_carriers):
                carrier_vals.append(year_data1.get(year, {}).get(carrier, 0))

            fig.add_trace(go.Bar(
                name=f"{format_scenario_name(subscenario1)} - {year}",
                x=sorted(list(all_carriers)),
                y=carrier_vals,
                legendgroup='sub1',
                marker_pattern_shape="",
                hovertemplate=f"{format_scenario_name(subscenario1)} ({year}): %{{y:.2f}}<extra></extra>"
            ))

        # Add traces for subscenario2 across all years
        for year in years:
            carrier_vals = []
            for carrier in sorted(all_carriers):
                carrier_vals.append(year_data2.get(year, {}).get(carrier, 0))

            fig.add_trace(go.Bar(
                name=f"{format_scenario_name(subscenario2)} - {year}",
                x=sorted(list(all_carriers)),
                y=carrier_vals,
                legendgroup='sub2',
                marker_pattern_shape="/",
                hovertemplate=f"{format_scenario_name(subscenario2)} ({year}): %{{y:.2f}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"Year Comparison: {metric}<br><sub>{format_scenario_name(subscenario1)} vs {format_scenario_name(subscenario2)}</sub>",
            xaxis_title="Carrier",
            yaxis_title="Value",
            barmode='group',
            template="plotly_white",
            hovermode='x unified',
            legend=dict(title="Scenario - Year")
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_cross_evolution_plot(main_scenario1, main_scenario2, metric, subscenario1, subscenario2, data_loader, color_mapper, main1=None, main2=None):
    """Create year on year evolution plot for two sub-scenarios (line plot over time)"""
    try:
        if not all([metric, subscenario1, subscenario2]):
            return create_empty_figure("Please select all parameters")

        if subscenario1 == subscenario2:
            return create_empty_figure("Please select two different sub-scenarios")

        # Get available years
        stats1 = data_loader.get_summary_stats(main1 or main_scenario1)
        years = stats1.get('years', [])

        if not years:
            return create_empty_figure("No years available")

        # Collect data for all years for both subscenarios
        all_carriers = set()
        year_data1 = {}
        year_data2 = {}

        for year in years:
            df1 = data_loader.get_data(main1 or main_scenario1, year=year, scenario_name=subscenario1, metric=metric)
            df2 = data_loader.get_data(main2 or main_scenario2, year=year, scenario_name=subscenario2, metric=metric)

            if df1 is not None and not df1.empty:
                data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
                carriers1 = data1.index.get_level_values(2).tolist()
                all_carriers.update(carriers1)
                year_data1[year] = {carrier: val for carrier, val in zip(carriers1, data1.values)}

            if df2 is not None and not df2.empty:
                data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]
                carriers2 = data2.index.get_level_values(2).tolist()
                all_carriers.update(carriers2)
                year_data2[year] = {carrier: val for carrier, val in zip(carriers2, data2.values)}

        if not all_carriers:
            return create_empty_figure("No data available")

        # Create line plot - each carrier gets two lines (one per subscenario)
        fig = go.Figure()

        for carrier in sorted(all_carriers):
            # Line for subscenario1
            carrier_years1 = []
            carrier_values1 = []
            for year in sorted(years):
                if carrier in year_data1.get(year, {}):
                    carrier_years1.append(year)
                    carrier_values1.append(year_data1[year][carrier])

            if carrier_years1:
                color = color_mapper.get_color(carrier, metric) if color_mapper else None
                fig.add_trace(go.Scatter(
                    x=carrier_years1,
                    y=carrier_values1,
                    mode='lines+markers',
                    name=f"{carrier} ({format_scenario_name(subscenario1)})",
                    line=dict(color=color, width=2, dash='solid'),
                    marker=dict(size=8, color=color),
                    legendgroup=carrier,
                    hovertemplate=f"{carrier} - {format_scenario_name(subscenario1)}: %{{y:.2f}}<extra></extra>"
                ))

            # Line for subscenario2
            carrier_years2 = []
            carrier_values2 = []
            for year in sorted(years):
                if carrier in year_data2.get(year, {}):
                    carrier_years2.append(year)
                    carrier_values2.append(year_data2[year][carrier])

            if carrier_years2:
                color = color_mapper.get_color(carrier, metric) if color_mapper else None
                fig.add_trace(go.Scatter(
                    x=carrier_years2,
                    y=carrier_values2,
                    mode='lines+markers',
                    name=f"{carrier} ({format_scenario_name(subscenario2)})",
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=8, color=color, symbol='diamond'),
                    legendgroup=carrier,
                    hovertemplate=f"{carrier} - {format_scenario_name(subscenario2)}: %{{y:.2f}}<extra></extra>"
                ))

        fig.update_layout(
            title=f"Year on Year Evolution: {metric}<br><sub>{format_scenario_name(subscenario1)} (solid) vs {format_scenario_name(subscenario2)} (dashed)</sub>",
            xaxis_title="Year",
            yaxis_title="Value",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(title="Technology", yanchor="top", y=0.99, xanchor="left", x=1.01)
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
