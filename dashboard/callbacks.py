"""
Callbacks for the dashboard
Handles all interactive components and updates
"""
from dash import Input, Output, State, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.colors import ColorMapper, format_scenario_name, format_main_scenario_name, clean_carrier_name


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
        [Output('single-plot-type-selector', 'options'),
         Output('single-plot-type-selector', 'value')],
        [Input('single-year-selector', 'value')]
    )
    def update_single_plot_types(year):
        """Update available plot types based on selected year"""
        if year == 'all':
            # Multi-year plots only
            options = [
                {'label': 'Stacked Bar (All Years)', 'value': 'stacked_bar'},
                {'label': 'Side by Side Bar Plot', 'value': 'year_comparison'},
                {'label': 'Year on Year Evolution', 'value': 'year_on_year_evolution'}
            ]
            return options, 'stacked_bar'
        else:
            # Single-year plots
            options = [
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Stacked Area', 'value': 'area'},
                {'label': 'Pie Chart', 'value': 'pie'}
            ]
            return options, 'bar'

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

    # ==================== Dead Zone Analysis Callbacks ====================
    @app.callback(
        [Output('deadzone-country-selector', 'options'),
         Output('deadzone-country-selector', 'value')],
        [Input('deadzone-main-scenario-selector', 'value'),
         Input('deadzone-year-selector', 'value')]
    )
    def update_deadzone_countries(main_scenario, year):
        """Update available countries for dead zone analysis"""
        if not main_scenario:
            return [{'label': 'All', 'value': 'all'}], ['EU']

        # Get available countries
        countries = data_loader.get_frontier_countries(main_scenario, year if year != 'all' else 2035)
        options = [{'label': 'All', 'value': 'all'}] + [{'label': c, 'value': c} for c in countries]
        return options, ['EU']

    @app.callback(
        Output('deadzone-plot', 'figure'),
        [Input('deadzone-main-scenario-selector', 'value'),
         Input('deadzone-year-selector', 'value'),
         Input('deadzone-scenario-selector', 'value'),
         Input('deadzone-country-selector', 'value')]
    )
    def update_deadzone_plot(main_scenario, year, scenario, country):
        """Update dead zone frontier plot - automatically infer comparison type"""
        if not main_scenario:
            return create_empty_figure("Please select a main scenario")

        # Infer comparison type from selections
        comparison_type = infer_comparison_type(year, scenario, country)

        return create_deadzone_frontier_plot(
            comparison_type, main_scenario, year, scenario, country, data_loader
        )

    @app.callback(
        Output('deadzone-summary', 'children'),
        [Input('deadzone-main-scenario-selector', 'value'),
         Input('deadzone-year-selector', 'value'),
         Input('deadzone-scenario-selector', 'value'),
         Input('deadzone-country-selector', 'value')]
    )
    def update_deadzone_summary(main_scenario, year, scenario, country):
        """Update dead zone summary with key takeaways"""
        if not main_scenario:
            return html.P("Select parameters to see analysis summary")

        # Infer comparison type
        comparison_type = infer_comparison_type(year, scenario, country)

        try:
            # Get frontier data to analyze
            stats = data_loader.get_summary_stats(main_scenario)
            all_years = stats.get('years', [])

            # Process scenarios
            if isinstance(scenario, list):
                scenarios_to_use = [s for s in scenario if s != 'all']
                if not scenarios_to_use or 'all' in scenario:
                    scenarios_to_use = stats.get('scenarios', [])
            else:
                scenarios_to_use = stats.get('scenarios', []) if scenario == 'all' else [scenario]

            # Process countries
            if isinstance(country, list):
                countries_to_use = [c for c in country if c != 'all']
                if not countries_to_use or 'all' in country:
                    countries_to_use = data_loader.get_frontier_countries(main_scenario, all_years[-1] if all_years else 2035)
            else:
                countries_to_use = data_loader.get_frontier_countries(main_scenario, all_years[-1] if all_years else 2035) if country == 'all' else [country]

            # Get some frontier data for insights
            year_for_analysis = all_years[-1] if year == 'all' and all_years else (year if year != 'all' else 2035)
            country_for_analysis = countries_to_use[0] if countries_to_use else 'EU'
            scenario_for_analysis = scenarios_to_use[0] if scenarios_to_use else 'baseline'

            frontier_dict = data_loader.get_frontier_data(main_scenario, year_for_analysis, country_for_analysis)
            frontier_values = frontier_dict.get(scenario_for_analysis, []) if frontier_dict else []

            # Calculate insights - handle numpy arrays
            if len(frontier_values) > 0:
                import numpy as np
                frontier_array = np.array(frontier_values)
                max_matching = float(np.max(frontier_array))
                avg_matching = float(np.mean(frontier_array))
                frontier_points = len(frontier_values)
            else:
                max_matching = 0
                avg_matching = 0
                frontier_points = 0

            # Emojis based on comparison type
            comparison_emojis = {
                'years': '📈',
                'scenarios_years': '🔄',
                'spatial': '🌍'
            }

            comparison_titles = {
                'years': 'Temporal Evolution',
                'scenarios_years': 'Policy Comparison',
                'spatial': 'Geographic Variation'
            }

            # Build year text
            year_text = f"Year: {year}" if year != 'all' else "All years"

            # Handle scenario as list or single value
            if isinstance(scenario, list):
                if 'all' in scenario:
                    scenario_text = "All scenarios"
                elif len(scenario) == 1:
                    scenario_text = f"{scenario[0]}"
                else:
                    scenario_text = f"{', '.join(scenario[:3])}"
            else:
                scenario_text = f"{scenario}" if scenario != 'all' else "All scenarios"

            # Handle country as list or single value
            if isinstance(country, list):
                if 'all' in country:
                    country_text = "All countries"
                elif len(country) == 1:
                    country_text = f"{country[0]}"
                else:
                    country_text = f"{', '.join(country[:3])}"
            else:
                country_text = f"{country}" if country != 'all' else "All countries"

            return html.Div([
                html.H4(f"Key Takeaways", style={'marginBottom': '20px', 'fontWeight': 'bold', 'color': '#1e3a8a'}),

                # Comparison type indicator
                html.Div([
                    html.Div([
                        html.Span("ANALYSIS TYPE", style={'fontSize': '12px', 'color': '#64748b', 'fontWeight': '600'}),
                        html.Div([
                            html.Span(comparison_emojis.get(comparison_type, '📊'), style={'fontSize': '20px', 'marginRight': '8px'}),
                            html.Label(comparison_titles.get(comparison_type, 'Frontier Comparison'),
                                     style={'fontSize': '14px', 'fontWeight': '500', 'color': '#1e293b'})
                        ], style={'marginTop': '5px', 'display': 'flex', 'alignItems': 'center'})
                    ], style={'flex': '1'}),

                    html.Div([
                        html.Span("CURRENT SELECTION", style={'fontSize': '12px', 'color': '#64748b', 'fontWeight': '600'}),
                        html.Div([
                            html.Span(year_text, style={'padding': '4px 12px', 'backgroundColor': '#3b82f6', 'color': 'white',
                                   'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': '500', 'marginRight': '5px'}),
                            html.Span(scenario_text, style={'padding': '4px 12px', 'backgroundColor': '#10b981', 'color': 'white',
                                   'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': '500', 'marginRight': '5px'}),
                            html.Span(country_text, style={'padding': '4px 12px', 'backgroundColor': '#f59e0b', 'color': 'white',
                                   'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': '500'})
                        ], style={'marginTop': '5px'})
                    ], style={'flex': '2'})
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),

                # Key metrics in cards
                html.Div([
                    html.Div([
                        # Max matching achievable
                        html.Div([
                            html.Div([
                                html.Span("🎯", style={'fontSize': '32px'}),
                                html.H6("Max matching achievable", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.H4(f"{max_matching:.1f}%",
                                   style={'color': '#10b981' if max_matching > 90 else '#f59e0b' if max_matching > 70 else '#ef4444',
                                          'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                            html.P(f"Sample: {format_scenario_name(scenario_for_analysis)}, {country_for_analysis}",
                                  style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                        ], className="col-md-3 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0'}),

                        # Average matching
                        html.Div([
                            html.Div([
                                html.Span("📊", style={'fontSize': '32px'}),
                                html.H6("Average matching", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.H4(f"{avg_matching:.1f}%",
                                   style={'color': '#3b82f6', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                            html.P(f"Across all frontier points",
                                  style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                        ], className="col-md-3 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'}),

                        # Frontier points
                        html.Div([
                            html.Div([
                                html.Span("⚡", style={'fontSize': '32px'}),
                                html.H6("Frontier points", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.H4(f"{frontier_points}",
                                   style={'color': '#8b5cf6', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                            html.P(f"Energy matching options",
                                  style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                        ], className="col-md-3 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'}),

                        # Comparison scope
                        html.Div([
                            html.Div([
                                html.Span("🔍", style={'fontSize': '32px'}),
                                html.H6("Comparison scope", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.Div([
                                html.P(f"{len(scenarios_to_use)} scenarios", style={'fontSize': '14px', 'margin': '4px 0', 'fontWeight': '600', 'color': '#3b82f6'}),
                                html.P(f"{len(countries_to_use)} countries", style={'fontSize': '14px', 'margin': '4px 0', 'fontWeight': '600', 'color': '#10b981'}),
                                html.P(f"{len(all_years)} years" if year == 'all' else f"Year {year}",
                                      style={'fontSize': '14px', 'margin': '4px 0', 'fontWeight': '600', 'color': '#f59e0b'})
                            ], style={'textAlign': 'center'})
                        ], className="col-md-3 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'})
                    ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '15px'}),

                    # Insights row
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span("💡", style={'fontSize': '32px'}),
                                html.H6("Key insights", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.Div([
                                html.P("📈 Higher matching requires more flexible energy systems" if comparison_type == 'years' else
                                      "🔄 Different policies create distinct frontier tradeoffs" if comparison_type == 'scenarios_years' else
                                      "🌍 Geography strongly influences achievable matching levels",
                                      style={'fontSize': '13px', 'margin': '8px 0', 'color': '#475569', 'lineHeight': '1.5'}),
                                html.P("⚡ Frontier curves show Pareto-optimal matching solutions",
                                      style={'fontSize': '13px', 'margin': '8px 0', 'color': '#475569', 'lineHeight': '1.5'}),
                                html.P("🎯 Points on the frontier represent different energy matching strategies",
                                      style={'fontSize': '13px', 'margin': '8px 0', 'color': '#475569', 'lineHeight': '1.5'})
                            ], style={'textAlign': 'left', 'padding': '0 20px'})
                        ], className="col-12 p-4", style={'backgroundColor': '#fffbeb', 'borderRadius': '12px', 'border': '1px solid #fcd34d'})
                    ], style={'display': 'flex', 'gap': '0px'})
                ])
            ])

        except Exception as e:
            return html.P(f"Error generating insights: {str(e)}", style={'color': '#ef4444'})

    # ==================== Cross-Scenario Comparison Callbacks ====================
    @app.callback(
        [Output('cross-plot-type-selector', 'options'),
         Output('cross-plot-type-selector', 'value')],
        [Input('cross-year-selector', 'value')]
    )
    def update_cross_plot_types(year):
        """Update available plot types based on selected year"""
        if year == 'all':
            # Multi-year plots only
            options = [
                {'label': 'Stacked Bar (All Years)', 'value': 'stacked_bar'},
                {'label': 'Side by Side Bar Plot', 'value': 'year_comparison'},
                {'label': 'Year on Year Evolution', 'value': 'year_on_year_evolution'}
            ]
            return options, 'stacked_bar'
        else:
            # Single-year plots
            options = [
                {'label': 'Side-by-Side', 'value': 'comparison'}
            ]
            return options, 'comparison'

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
        [Output('cross-carrier-selector', 'options'),
         Output('cross-carrier-selector', 'value')],
        [Input('cross-metric-selector', 'value'),
         Input('cross-subscenario1-selector', 'value')]
    )
    def update_cross_carriers(metric, subscenario1):
        """Update available carriers based on selected metric and first subscenario"""
        if not metric or not subscenario1:
            return [], []

        main_scenario = find_main_scenario(subscenario1)
        if not main_scenario:
            return [], []

        carriers = data_loader.get_carriers_for_metric(main_scenario, metric)
        options = [{'label': c, 'value': c} for c in carriers]
        # Select all by default
        return options, carriers

    @app.callback(
        Output('cross-comparison-plot', 'figure'),
        [Input('cross-year-selector', 'value'),
         Input('cross-metric-selector', 'value'),
         Input('cross-plot-type-selector', 'value'),
         Input('cross-subscenario1-selector', 'value'),
         Input('cross-subscenario2-selector', 'value'),
         Input('cross-subscenario3-selector', 'value'),
         Input('cross-subscenario4-selector', 'value'),
         Input('cross-carrier-selector', 'value'),
         Input('cross-grouping-selector', 'value')]
    )
    def update_cross_comparison(year, metric, plot_type, subscenario1, subscenario2, subscenario3, subscenario4, carriers, grouping):
        """Update cross-scenario comparison plot"""
        # Collect all selected subscenarios (filter out None)
        subscenarios = [s for s in [subscenario1, subscenario2, subscenario3, subscenario4] if s]

        if len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        # Find main scenarios for each subscenario
        main_scenarios = [find_main_scenario(s) if s else None for s in subscenarios]

        return create_cross_comparison_plot(main_scenarios[0], year, metric, plot_type, subscenarios,
                                            data_loader, color_mapper, main_scenarios, carriers, grouping)

    @app.callback(
        Output('cross-difference-plot', 'figure'),
        [Input('cross-year-selector', 'value'),
         Input('cross-metric-selector', 'value'),
         Input('diff-scenario-a-selector', 'value'),
         Input('diff-scenario-b-selector', 'value'),
         Input('cross-carrier-selector', 'value')]
    )
    def update_cross_difference(year, metric, subscenario1, subscenario2, carriers):
        """Update cross-scenario difference plot"""
        main1 = find_main_scenario(subscenario1) if subscenario1 else None
        main2 = find_main_scenario(subscenario2) if subscenario2 else None
        return create_cross_difference_plot(main1 or main2, year, metric, subscenario1, subscenario2,
                                           data_loader, color_mapper, main1, main2, carriers)

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

    # ==================== Timeseries Exploration Callbacks ====================
    @app.callback(
        [Output('ts-year-selector', 'options'),
         Output('ts-year-selector', 'value')],
        [Input('ts-main-scenario-selector', 'value')]
    )
    def update_ts_years(main_scenario):
        """Update available years for timeseries"""
        if not main_scenario:
            return [], None

        metadata = data_loader.get_timeseries_metadata(main_scenario)
        years = metadata.get('years', [])
        options = [{'label': str(y), 'value': y} for y in years]
        default_value = years[-1] if years else None
        return options, default_value

    @app.callback(
        [Output('ts-scenario-selector', 'options'),
         Output('ts-scenario-selector', 'value')],
        [Input('ts-main-scenario-selector', 'value')]
    )
    def update_ts_scenarios(main_scenario):
        """Update available scenarios for timeseries"""
        if not main_scenario:
            return [], []

        metadata = data_loader.get_timeseries_metadata(main_scenario)
        scenarios = metadata.get('scenarios', [])
        options = [{'label': s, 'value': s} for s in scenarios]
        default_value = [scenarios[0]] if scenarios else []
        return options, default_value

    @app.callback(
        [Output('ts-type-selector', 'options'),
         Output('ts-type-selector', 'value')],
        [Input('ts-main-scenario-selector', 'value')]
    )
    def update_ts_types(main_scenario):
        """Update available timeseries types"""
        if not main_scenario:
            return [], None

        metadata = data_loader.get_timeseries_metadata(main_scenario)
        types = metadata.get('types', [])
        options = [{'label': t, 'value': t} for t in types]
        default_value = types[0] if types else None
        return options, default_value

    @app.callback(
        [Output('ts-country-selector', 'options'),
         Output('ts-country-selector', 'value')],
        [Input('ts-main-scenario-selector', 'value')]
    )
    def update_ts_countries(main_scenario):
        """Update available countries for timeseries"""
        if not main_scenario:
            return [], None

        metadata = data_loader.get_timeseries_metadata(main_scenario)
        countries = metadata.get('countries', [])
        options = [{'label': c, 'value': c} for c in countries]
        # Try to find a reasonable default
        if 'Germany' in countries:
            default_value = 'Germany'
        elif 'France' in countries:
            default_value = 'France'
        elif countries:
            default_value = countries[0]
        else:
            default_value = None
        return options, default_value

    @app.callback(
        Output('ts-carrier-selector', 'options'),
        [Input('ts-main-scenario-selector', 'value')]
    )
    def update_ts_carriers(main_scenario):
        """Update available carriers for timeseries"""
        if not main_scenario:
            return []

        metadata = data_loader.get_timeseries_metadata(main_scenario)
        carriers = metadata.get('carriers', [])
        options = [{'label': c, 'value': c} for c in carriers]
        return options

    @app.callback(
        Output('ts-plot-type-selector', 'value'),
        [Input('ts-type-selector', 'value')],
        [State('ts-plot-type-selector', 'value')]
    )
    def update_ts_plot_type(ts_type, current_plot_type):
        """Auto-select line plot for Electricity Balance"""
        if ts_type == 'Electricity Balance':
            return 'line'
        return current_plot_type

    @app.callback(
        Output('ts-plot', 'figure'),
        [Input('ts-year-selector', 'value'),
         Input('ts-scenario-selector', 'value'),
         Input('ts-type-selector', 'value'),
         Input('ts-country-selector', 'value'),
         Input('ts-carrier-selector', 'value'),
         Input('ts-timerange-selector', 'value'),
         Input('ts-plot-type-selector', 'value')]
    )
    def update_ts_plot(year, scenarios, ts_type, country, carriers, time_range, plot_type):
        """Update timeseries plot"""
        if not year or not scenarios or not ts_type or not country:
            return create_empty_figure("Please select year, scenarios, type, and country")

        if not isinstance(scenarios, list):
            scenarios = [scenarios]

        return create_timeseries_plot(year, scenarios, ts_type, country, carriers,
                                     time_range, plot_type, data_loader)

    @app.callback(
        Output('ts-info', 'children'),
        [Input('ts-year-selector', 'value'),
         Input('ts-scenario-selector', 'value'),
         Input('ts-type-selector', 'value'),
         Input('ts-country-selector', 'value'),
         Input('ts-carrier-selector', 'value')]
    )
    def update_ts_info(year, scenarios, ts_type, country, carriers):
        """Update timeseries info panel"""
        if not year or not scenarios or not ts_type or not country:
            return html.P("Select parameters to see timeseries data")

        num_scenarios = len(scenarios) if isinstance(scenarios, list) else 1
        num_carriers = len(carriers) if carriers else "All"

        return html.Div([
            html.P(f"Year: {year}", style={'marginBottom': '5px'}),
            html.P(f"Scenarios: {num_scenarios} selected", style={'marginBottom': '5px'}),
            html.P(f"Type: {ts_type}", style={'marginBottom': '5px'}),
            html.P(f"Country: {country}", style={'marginBottom': '5px'}),
            html.P(f"Carriers: {num_carriers}", style={'marginBottom': '5px'}),
        ])


# ==================== Helper Functions ====================

def create_plot(scenario, year, scenario_name, metric, plot_type, carriers, data_loader, color_mapper):
    """Create a plot based on parameters"""
    try:
        # Handle multi-year plots (year='all')
        if year == 'all' or plot_type in ['stacked_bar', 'year_comparison', 'year_on_year_evolution']:
            if plot_type == 'stacked_bar':
                return create_stacked_bar_all_years(scenario, scenario_name, metric, carriers, data_loader, color_mapper)
            elif plot_type == 'year_comparison':
                return create_year_comparison_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper)
            elif plot_type == 'year_on_year_evolution':
                return create_year_on_year_evolution_plot(scenario, scenario_name, metric, carriers, data_loader, color_mapper)

        # Single year plots
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

    # Clean carrier names
    carriers_clean = [clean_carrier_name(c) for c in carriers]
    colors = [color_mapper.get_color(c, metric) for c in carriers]

    fig = go.Figure(data=[
        go.Bar(x=carriers_clean, y=values, marker_color=colors)
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

    # Clean carrier names
    carriers_clean = [clean_carrier_name(c) for c in carriers]
    colors = [color_mapper.get_color(carriers[i], metric) for i in range(len(carriers))]

    fig = go.Figure(data=[
        go.Pie(labels=carriers_clean, values=values, marker_colors=colors)
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
                carrier_clean = clean_carrier_name(carrier)

                fig.add_trace(go.Bar(
                    name=carrier_clean,
                    x=[str(year)],
                    y=[values[i]],
                    marker_color=color,
                    legendgroup=carrier,
                    showlegend=(year == years[0]),  # Only show legend for first year
                    hovertemplate=f"{carrier_clean}: %{{y:.2f}}<extra></extra>"
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
            carrier_clean = clean_carrier_name(carrier)

            fig.add_trace(go.Scatter(
                x=carrier_years,
                y=carrier_values,
                mode='lines+markers',
                name=carrier_clean,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                hovertemplate=f"{carrier_clean}: %{{y:.2f}}<extra></extra>"
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

def create_cross_comparison_plot(main_scenario, year, metric, plot_type, subscenarios, data_loader, color_mapper, main_scenarios=None, carriers=None, grouping='year'):
    """Create comparison plot for multiple sub-scenarios (2-4)"""
    try:
        # Handle both list and individual inputs for backwards compatibility
        if not isinstance(subscenarios, list):
            subscenarios = [subscenarios]
        if main_scenarios is None or not isinstance(main_scenarios, list):
            main_scenarios = [main_scenario] * len(subscenarios)

        if not metric or len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        # Check for duplicates
        if len(subscenarios) != len(set(subscenarios)):
            return create_empty_figure("Please select different sub-scenarios")

        # Handle multi-year plot types (these require year='all')
        if plot_type in ['stacked_bar', 'stacked_with_total', 'year_comparison', 'year_on_year_evolution']:
            if year != 'all':
                return create_empty_figure("Please select 'All' years for this plot type")

            if plot_type == 'stacked_bar':
                return create_cross_stacked_bar_plot(metric, subscenarios,
                                                    data_loader, color_mapper, main_scenarios, carriers, grouping)
            elif plot_type == 'stacked_with_total':
                return create_cross_stacked_with_total_plot(metric, subscenarios,
                                                           data_loader, color_mapper, main_scenarios, carriers, grouping)
            elif plot_type == 'year_comparison':
                return create_cross_year_comparison_plot(metric, subscenarios,
                                                         data_loader, color_mapper, main_scenarios, carriers, grouping)
            elif plot_type == 'year_on_year_evolution':
                return create_cross_evolution_plot(metric, subscenarios,
                                                   data_loader, color_mapper, main_scenarios, carriers, grouping)

        # Default: side-by-side comparison (single year required)
        if not year or year == 'all':
            return create_empty_figure("Please select a specific year for side-by-side comparison")

        fig = go.Figure()
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']

        # Get data for all sub-scenarios from their respective main scenarios
        all_data = []
        for i, (subscenario, main_sc) in enumerate(zip(subscenarios, main_scenarios)):
            df = data_loader.get_data(main_sc, year=year, scenario_name=subscenario, metric=metric)
            if df is not None and not df.empty:
                data = df.iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df.iloc[:, 0]
                # Filter by selected carriers if provided
                if carriers:
                    data = data[data.index.get_level_values(2).isin(carriers)]
                all_data.append((subscenario, data))

        if not all_data:
            return create_empty_figure("No data available for selected parameters")

        # Get carrier list from first dataset
        carrier_list = all_data[0][1].index.get_level_values(2).tolist()

        # Add bars for all scenarios
        for i, (subscenario, data) in enumerate(all_data):
            fig.add_trace(go.Bar(
                name=format_scenario_name(subscenario),
                x=carrier_list,
                y=data.values,
                marker_color=colors[i % len(colors)]
            ))

        scenario_names = ' vs '.join([format_scenario_name(s) for s in subscenarios])
        fig.update_layout(
            title=f"{metric} ({year})<br><sub>{scenario_names}</sub>",
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


def create_cross_difference_plot(main_scenario, year, metric, subscenario1, subscenario2, data_loader, color_mapper, main1=None, main2=None, carriers=None):
    """Create difference plot between two sub-scenarios"""
    try:
        # Handle 'all' year case - use latest year for difference plot
        display_year = year
        if year == 'all':
            stats = data_loader.get_summary_stats(main1 or main_scenario)
            years = stats.get('years', [])
            display_year = years[-1] if years else None
            if not display_year:
                return create_empty_figure("No year data available")

        if not all([display_year, metric, subscenario1, subscenario2]):
            return create_empty_figure("Please select all parameters")

        if subscenario1 == subscenario2:
            return create_empty_figure("Please select two different sub-scenarios")

        # Get data for both sub-scenarios from their respective main scenarios
        df1 = data_loader.get_data(main1 or main_scenario, year=display_year, scenario_name=subscenario1, metric=metric)
        df2 = data_loader.get_data(main2 or main_scenario, year=display_year, scenario_name=subscenario2, metric=metric)

        if df1 is None or df1.empty or df2 is None or df2.empty:
            return create_empty_figure("No data available for selected parameters")

        # Extract data series
        data1 = df1.iloc[:, 0] if isinstance(df1.columns, pd.MultiIndex) else df1.iloc[:, 0]
        data2 = df2.iloc[:, 0] if isinstance(df2.columns, pd.MultiIndex) else df2.iloc[:, 0]

        # Filter by selected carriers if provided
        if carriers:
            data1 = data1[data1.index.get_level_values(2).isin(carriers)]
            data2 = data2[data2.index.get_level_values(2).isin(carriers)]

        carrier_list = data1.index.get_level_values(2).tolist()
        diff = data2.values - data1.values

        # Color bars based on positive/negative
        colors = ['green' if d > 0 else 'red' for d in diff]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=carrier_list,
            y=diff,
            marker_color=colors,
            text=[f"{d:+.2f}" for d in diff],
            textposition='outside'
        ))

        year_label = f"Year {display_year}" if year == 'all' else display_year

        fig.update_layout(
            title=f"Difference: {format_scenario_name(subscenario2)} - {format_scenario_name(subscenario1)}<br><sub>{format_main_scenario_name(main_scenario)}: {metric} ({year_label})</sub>",
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
    """Create summary statistics comparing two sub-scenarios with 6 key takeaways"""
    try:
        # Handle 'all' year case - use latest year for summary stats
        display_year = year
        if year == 'all':
            stats = data_loader.get_summary_stats(main1 or main_scenario)
            years = stats.get('years', [])
            display_year = years[-1] if years else None
            if not display_year:
                return html.P("No year data available")

        if not all([display_year, metric, subscenario1, subscenario2]):
            return html.P("Please select all parameters")

        if subscenario1 == subscenario2:
            return html.P("Please select two different sub-scenarios")

        # Get data for both sub-scenarios from their respective main scenarios
        df1 = data_loader.get_data(main1 or main_scenario, year=display_year, scenario_name=subscenario1, metric=metric)
        df2 = data_loader.get_data(main2 or main_scenario, year=display_year, scenario_name=subscenario2, metric=metric)

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

        # Fetch emissions data separately
        emissions_metric = "(j) CO2 emissions"
        df1_emissions = data_loader.get_data(main1 or main_scenario, year=display_year, scenario_name=subscenario1, metric=emissions_metric)
        df2_emissions = data_loader.get_data(main2 or main_scenario, year=display_year, scenario_name=subscenario2, metric=emissions_metric)

        emissions1 = 0
        emissions2 = 0
        emissions_diff = 0
        emissions_pct = 0
        has_emissions = False

        if df1_emissions is not None and not df1_emissions.empty and df2_emissions is not None and not df2_emissions.empty:
            emissions_data1 = df1_emissions.iloc[:, 0] if isinstance(df1_emissions.columns, pd.MultiIndex) else df1_emissions.iloc[:, 0]
            emissions_data2 = df2_emissions.iloc[:, 0] if isinstance(df2_emissions.columns, pd.MultiIndex) else df2_emissions.iloc[:, 0]
            emissions1 = emissions_data1.sum()
            emissions2 = emissions_data2.sum()
            emissions_diff = emissions2 - emissions1
            emissions_pct = (emissions_diff / emissions1 * 100) if emissions1 != 0 else 0
            has_emissions = True

        # Calculate various metrics
        abs_diffs = values2 - values1
        renewables_mask = np.array(['solar' in c.lower() or 'wind' in c.lower() for c in carriers1])
        clean_mask = np.array(['nuclear' in c.lower() or 'carbon' in c.lower() for c in carriers1])

        renewables_build1 = values1[renewables_mask].sum() if renewables_mask.any() else 0
        renewables_build2 = values2[renewables_mask].sum() if renewables_mask.any() else 0
        renewables_diff = renewables_build2 - renewables_build1

        clean_firm1 = values1[clean_mask].sum() if clean_mask.any() else 0
        clean_firm2 = values2[clean_mask].sum() if clean_mask.any() else 0
        clean_diff = clean_firm2 - clean_firm1

        # Find top movers (resource shifts)
        top_increase_idx = np.argmax(abs_diffs) if len(abs_diffs) > 0 else 0
        top_decrease_idx = np.argmin(abs_diffs) if len(abs_diffs) > 0 else 0

        # Create 6-category key takeaways styled like the screenshot
        return html.Div([
            html.H4(f"Key Takeaways", style={'marginBottom': '20px', 'fontWeight': 'bold', 'color': '#1e3a8a'}),

            # Year indicator if showing 'all'
            html.P(f"Year {display_year}", style={'marginBottom': '20px', 'color': '#64748b', 'fontSize': '14px'}) if year == 'all' else html.Div(),

            html.Div([
                html.Div([
                    html.Span("SCENARIO A", style={'fontSize': '12px', 'color': '#64748b', 'fontWeight': '600'}),
                    html.Div([
                        html.Label(format_scenario_name(subscenario1),
                                 style={'fontSize': '14px', 'fontWeight': '500', 'color': '#1e293b'})
                    ], style={'marginTop': '5px'})
                ], style={'flex': '1'}),

                html.Div([
                    html.Span("SCENARIO B", style={'fontSize': '12px', 'color': '#64748b', 'fontWeight': '600'}),
                    html.Div([
                        html.Label(format_scenario_name(subscenario2),
                                 style={'fontSize': '14px', 'fontWeight': '500', 'color': '#1e293b'})
                    ], style={'marginTop': '5px'})
                ], style={'flex': '1'}),

                html.Div([
                    html.Span("COMPARING", style={'fontSize': '12px', 'color': '#64748b', 'fontWeight': '600'}),
                    html.Div([
                        html.Span(format_scenario_name(subscenario1),
                                style={'padding': '4px 12px', 'backgroundColor': '#3b82f6', 'color': 'white',
                                       'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': '500', 'marginRight': '5px'}),
                        html.Span(format_scenario_name(subscenario2),
                                style={'padding': '4px 12px', 'backgroundColor': '#10b981', 'color': 'white',
                                       'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': '500'})
                    ], style={'marginTop': '5px'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),

            # 6 Key categories in 2 rows of 3
            html.Div([
                # Row 1
                html.Div([
                    # Total generation
                    html.Div([
                        html.Div([
                            html.Span("⚡", style={'fontSize': '32px'}),
                            html.H6("Total generation", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4(f"{diff:+.0f} {metric.split('(')[-1].split(')')[0] if '(' in metric else 'units'}",
                               style={'color': '#10b981' if diff > 0 else '#ef4444' if diff < 0 else '#64748b', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                        html.P(f"({pct_change:+.0f}%)", style={'fontSize': '14px', 'color': '#64748b', 'textAlign': 'center', 'margin': '0'}),
                        html.P(f"{format_scenario_name(subscenario1)}: {total1:.0f} → {format_scenario_name(subscenario2)}: {total2:.0f}",
                              style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0'}),

                    # Emissions
                    html.Div([
                        html.Div([
                            html.Span("📉", style={'fontSize': '32px'}),
                            html.H6("Emissions", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4(f"{emissions_diff:+.0f} Mt CO₂" if has_emissions else "N/A",
                               style={'color': '#10b981' if emissions_diff < 0 else '#ef4444' if emissions_diff > 0 else '#64748b', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                        html.P(f"({emissions_pct:+.0f}%)" if has_emissions else "Emissions data not available",
                              style={'fontSize': '14px', 'color': '#64748b', 'textAlign': 'center', 'margin': '0'}),
                        html.P(f"{format_scenario_name(subscenario1)}: {emissions1:.0f} Mt → {format_scenario_name(subscenario2)}: {emissions2:.0f} Mt" if has_emissions else "",
                              style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'}),

                    # Clean share (% of clean/renewable)
                    html.Div([
                        html.Div([
                            html.Span("🌱", style={'fontSize': '32px'}),
                            html.H6("Clean share", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4(f"{(renewables_build2 + clean_firm2) / total2 * 100 - (renewables_build1 + clean_firm1) / total1 * 100:+.1f}pp" if total1 > 0 and total2 > 0 else "N/A",
                               style={'color': '#10b981' if (renewables_build2 + clean_firm2) / total2 > (renewables_build1 + clean_firm1) / total1 else '#ef4444', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                        html.P(f"{format_scenario_name(subscenario1)}: {(renewables_build1 + clean_firm1) / total1 * 100:.0f}% → {format_scenario_name(subscenario2)}: {(renewables_build2 + clean_firm2) / total2 * 100:.0f}%" if total1 > 0 and total2 > 0 else "",
                              style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'})
                ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '15px'}),

                # Row 2
                html.Div([
                    # Renewables build
                    html.Div([
                        html.Div([
                            html.Span("☀️", style={'fontSize': '32px'}),
                            html.H6("Renewables build", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4(f"{renewables_diff:+.0f} {metric.split('(')[-1].split(')')[0] if '(' in metric else 'units'}",
                               style={'color': '#10b981' if renewables_diff > 0 else '#ef4444' if renewables_diff < 0 else '#64748b', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                        html.P(f"({renewables_diff / renewables_build1 * 100:+.0f}%)" if renewables_build1 > 0 else "(new capacity)",
                              style={'fontSize': '14px', 'color': '#64748b', 'textAlign': 'center', 'margin': '0'}),
                        html.P(f"Solar + Wind", style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0'}),

                    # Clean firm
                    html.Div([
                        html.Div([
                            html.Span("⚛️", style={'fontSize': '32px'}),
                            html.H6("Clean firm", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H4(f"{clean_diff:+.0f} {metric.split('(')[-1].split(')')[0] if '(' in metric else 'units'}",
                               style={'color': '#10b981' if clean_diff > 0 else '#ef4444' if clean_diff < 0 else '#64748b', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '5px'}),
                        html.P(f"({clean_diff / clean_firm1 * 100:+.0f}%)" if clean_firm1 > 0 else "(new capacity)",
                              style={'fontSize': '14px', 'color': '#64748b', 'textAlign': 'center', 'margin': '0'}),
                        html.P(f"Nuclear + CCS", style={'fontSize': '12px', 'color': '#94a3b8', 'textAlign': 'center', 'marginTop': '8px'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'}),

                    # Top moves (resource shifts)
                    html.Div([
                        html.Div([
                            html.Span("🔍", style={'fontSize': '32px'}),
                            html.H6("Top moves", style={'marginTop': '10px', 'fontWeight': '600', 'fontSize': '14px', 'color': '#475569'})
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.H6("Resource shifts", style={'color': '#3b82f6', 'fontSize': '16px', 'fontWeight': '600', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Div([
                            html.P(f"Largest build: {carriers1[top_increase_idx]} (+{abs_diffs[top_increase_idx]:.0f})",
                                  style={'fontSize': '12px', 'margin': '4px 0'}),
                            html.P(f"Largest reduction: {carriers1[top_decrease_idx]} ({abs_diffs[top_decrease_idx]:.0f})",
                                  style={'fontSize': '12px', 'margin': '4px 0'})
                        ], style={'textAlign': 'center', 'color': '#64748b'})
                    ], className="col-md-4 p-4", style={'backgroundColor': '#f8fafc', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginLeft': '10px'})
                ], style={'display': 'flex', 'gap': '0px'})
            ])
        ])

    except Exception as e:
        return html.P(f"Error: {str(e)}")


def create_cross_stacked_bar_plot(metric, subscenarios, data_loader, color_mapper, main_scenarios=None, carriers=None, grouping='year'):
    """Create stacked bar plot comparing multiple sub-scenarios (2-4) across all years"""
    try:
        if not metric or len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        # Get available years
        stats = data_loader.get_summary_stats(main_scenarios[0])
        years = stats.get('years', [])

        if not years:
            return create_empty_figure("No years available")

        # Collect data for all years for all subscenarios
        all_carriers = set()
        year_data = {}  # Dict of dicts: {subscenario: {year: {carrier: value}}}

        for subscenario, main_sc in zip(subscenarios, main_scenarios):
            year_data[subscenario] = {}
            for year in years:
                df = data_loader.get_data(main_sc, year=year, scenario_name=subscenario, metric=metric)
                if df is not None and not df.empty:
                    data = df.iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df.iloc[:, 0]
                    carrier_list = data.index.get_level_values(2).tolist()
                    all_carriers.update(carrier_list)
                    year_data[subscenario][year] = {carrier: val for carrier, val in zip(carrier_list, data.values)}

        if not all_carriers:
            return create_empty_figure("No data available")

        # Filter carriers if selection is provided
        if carriers:
            all_carriers = all_carriers.intersection(set(carriers))

        if not all_carriers:
            return create_empty_figure("No carriers selected")

        fig = go.Figure()
        opacities = [1.0, 0.75, 0.6, 0.45]  # Different opacities for up to 4 scenarios

        # Create bars based on grouping mode
        for carrier in sorted(all_carriers):
            x_labels = []
            y_values = []
            opacity_values = []

            if grouping == 'year':
                # Group by year: all scenarios for each year appear together
                for year in years:
                    for i, subscenario in enumerate(subscenarios):
                        x_labels.append(f"{year} - {format_scenario_name(subscenario)}")
                        y_values.append(year_data[subscenario].get(year, {}).get(carrier, 0))
                        opacity_values.append(opacities[i % len(opacities)])
            else:  # grouping == 'scenario'
                # Group by scenario: all years for each scenario appear together
                for i, subscenario in enumerate(subscenarios):
                    for year in years:
                        x_labels.append(f"{format_scenario_name(subscenario)} - {year}")
                        y_values.append(year_data[subscenario].get(year, {}).get(carrier, 0))
                        opacity_values.append(opacities[i % len(opacities)])

            color = color_mapper.get_color(carrier, metric) if color_mapper else None

            # Create a trace for each carrier with interleaved data
            carrier_clean = clean_carrier_name(carrier)
            for i, (x, y, opacity) in enumerate(zip(x_labels, y_values, opacity_values)):
                fig.add_trace(go.Bar(
                    name=carrier_clean,
                    x=[x],
                    y=[y],
                    marker=dict(color=color, opacity=opacity),
                    legendgroup=carrier,
                    showlegend=(i == 0),  # Only show legend for first occurrence
                    hovertemplate=f"{carrier_clean}: %{{y:.2f}}<extra></extra>"
                ))

        scenario_names = ' vs '.join([format_scenario_name(s) for s in subscenarios])
        group_label = "Grouped by Year" if grouping == 'year' else "Grouped by Scenario"
        fig.update_layout(
            title=f"Stacked Bar - All Years: {metric}<br><sub>{scenario_names} ({group_label})</sub>",
            xaxis_title="Year and Scenario" if grouping == 'year' else "Scenario and Year",
            yaxis_title="Value",
            barmode='stack',
            bargap=0.3,
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


def create_cross_stacked_with_total_plot(metric, subscenarios, data_loader, color_mapper, main_scenarios=None, carriers=None, grouping='year'):
    """Create stacked bar plot with total line overlay comparing multiple sub-scenarios across all years"""
    try:
        # For now, only use first 2 scenarios for this plot type
        if len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        subscenario1, subscenario2 = subscenarios[0], subscenarios[1]
        main1, main2 = main_scenarios[0], main_scenarios[1]

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

        # Filter carriers if selection is provided
        if carriers:
            all_carriers = all_carriers.intersection(set(carriers))

        if not all_carriers:
            return create_empty_figure("No carriers selected")

        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Interleave bars by year so both scenarios for each year appear together
        for carrier in sorted(all_carriers):
            x_labels = []
            y_values = []
            opacities = []

            for year in years:
                # Add subscenario1 bar for this year
                x_labels.append(f"{year} - {format_scenario_name(subscenario1)}")
                y_values.append(year_data1.get(year, {}).get(carrier, 0))
                opacities.append(1.0)

                # Add subscenario2 bar for this year (right after subscenario1)
                x_labels.append(f"{year} - {format_scenario_name(subscenario2)}")
                y_values.append(year_data2.get(year, {}).get(carrier, 0))
                opacities.append(0.6)

            color = color_mapper.get_color(carrier, metric) if color_mapper else None

            # Create a trace for each carrier with interleaved data
            carrier_clean = clean_carrier_name(carrier)
            for i, (x, y, opacity) in enumerate(zip(x_labels, y_values, opacities)):
                fig.add_trace(go.Bar(
                    name=carrier_clean,
                    x=[x],
                    y=[y],
                    marker=dict(color=color, opacity=opacity),
                    legendgroup=carrier,
                    showlegend=(i == 0),
                    hovertemplate=f"{carrier_clean}: %{{y:.2f}}<extra></extra>"
                ), secondary_y=False)

        # Calculate totals for line plot
        totals_x = []
        totals_y = []

        for year in years:
            # Total for subscenario1
            total1 = sum(year_data1.get(year, {}).get(carrier, 0) for carrier in all_carriers)
            totals_x.append(f"{year} - {format_scenario_name(subscenario1)}")
            totals_y.append(total1)

            # Total for subscenario2
            total2 = sum(year_data2.get(year, {}).get(carrier, 0) for carrier in all_carriers)
            totals_x.append(f"{year} - {format_scenario_name(subscenario2)}")
            totals_y.append(total2)

        # Add total line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=totals_x,
            y=totals_y,
            mode='lines+markers',
            name='Total',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black'),
            hovertemplate='Total: %{y:.2f}<extra></extra>',
            text=[f"{(y/max(totals_y)*100):.1f}%" for y in totals_y],
            textposition='top center',
            textfont=dict(size=10)
        ), secondary_y=True)

        fig.update_layout(
            title=f"Stacked Bar with Total: {metric}<br><sub>{format_scenario_name(subscenario1)} (solid) vs {format_scenario_name(subscenario2)} (lighter)</sub>",
            xaxis_title="Year and Scenario",
            barmode='stack',
            bargap=0.3,
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

        # Set y-axes titles
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text="Total (%)", secondary_y=True)

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_cross_year_comparison_plot(metric, subscenarios, data_loader, color_mapper, main_scenarios=None, carriers=None, grouping='year'):
    """Create year comparison plot for multiple sub-scenarios (grouped bars across years)"""
    try:
        # For now, only use first 2 scenarios for this plot type
        if len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        subscenario1, subscenario2 = subscenarios[0], subscenarios[1]
        main1, main2 = main_scenarios[0], main_scenarios[1]
        main_scenario1, main_scenario2 = main1, main2

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

        # Filter carriers if selection is provided
        if carriers:
            all_carriers = all_carriers.intersection(set(carriers))

        if not all_carriers:
            return create_empty_figure("No carriers selected")

        # Create grouped bar plot with year-carrier combinations on x-axis
        fig = go.Figure()

        # Create x-axis labels combining year and carrier
        x_labels = []
        for year in years:
            for carrier in sorted(all_carriers):
                x_labels.append(f"{year}<br>{carrier}")

        # Collect all values for subscenario1
        y_vals1 = []
        for year in years:
            for carrier in sorted(all_carriers):
                y_vals1.append(year_data1.get(year, {}).get(carrier, 0))

        # Collect all values for subscenario2
        y_vals2 = []
        for year in years:
            for carrier in sorted(all_carriers):
                y_vals2.append(year_data2.get(year, {}).get(carrier, 0))

        # Add traces for both scenarios
        fig.add_trace(go.Bar(
            name=format_scenario_name(subscenario1),
            x=x_labels,
            y=y_vals1,
            marker_pattern_shape="",
            hovertemplate=f"{format_scenario_name(subscenario1)}: %{{y:.2f}}<extra></extra>"
        ))

        fig.add_trace(go.Bar(
            name=format_scenario_name(subscenario2),
            x=x_labels,
            y=y_vals2,
            marker_pattern_shape="/",
            hovertemplate=f"{format_scenario_name(subscenario2)}: %{{y:.2f}}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Side by Side Bar Plot: {metric}<br><sub>{format_scenario_name(subscenario1)} vs {format_scenario_name(subscenario2)}</sub>",
            xaxis_title="Year - Carrier",
            yaxis_title="Value",
            barmode='group',
            template="plotly_white",
            hovermode='x unified',
            legend=dict(title="Scenario")
        )

        return fig

    except Exception as e:
        return create_empty_figure(f"Error: {str(e)}")


def create_cross_evolution_plot(metric, subscenarios, data_loader, color_mapper, main_scenarios=None, carriers=None, grouping='year'):
    """Create year on year evolution plot for multiple sub-scenarios (line plot over time)"""
    try:
        # For now, only use first 2 scenarios for this plot type
        if len(subscenarios) < 2:
            return create_empty_figure("Please select at least 2 sub-scenarios")

        subscenario1, subscenario2 = subscenarios[0], subscenarios[1]
        main1, main2 = main_scenarios[0], main_scenarios[1]
        main_scenario1, main_scenario2 = main1, main2

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

        # Filter carriers if selection is provided
        if carriers:
            all_carriers = all_carriers.intersection(set(carriers))

        if not all_carriers:
            return create_empty_figure("No carriers selected")

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


def infer_comparison_type(year, scenario, country):
    """
    Automatically infer comparison type based on selections:
    - If country is 'all' or list with 'all' -> spatial comparison
    - If scenario is 'all' or list with 'all' -> scenarios_years comparison
    - If year is 'all' OR multiple years implied -> years comparison
    - Default: years comparison
    """
    # Handle country as list or single value
    if isinstance(country, list):
        has_all_countries = 'all' in country
        num_countries = len([c for c in country if c != 'all'])
    else:
        has_all_countries = country == 'all'
        num_countries = 0 if has_all_countries else 1

    # Handle scenario as list or single value
    if isinstance(scenario, list):
        has_all_scenarios = 'all' in scenario
        num_scenarios = len([s for s in scenario if s != 'all'])
    else:
        has_all_scenarios = scenario == 'all'
        num_scenarios = 0 if has_all_scenarios else 1

    # Spatial comparison: comparing multiple countries
    if has_all_countries:
        return 'spatial'

    # Scenarios & Years comparison: comparing multiple scenarios
    if has_all_scenarios or num_scenarios > 1:
        return 'scenarios_years'

    # Years comparison: comparing across years (default when year='all')
    if year == 'all':
        return 'years'

    # If single year and single scenario but multiple countries -> years comparison
    if num_countries > 1 and year != 'all' and num_scenarios == 1:
        return 'years'  # Treat as years comparison with multiple countries

    # Default: years comparison
    return 'years'


def create_deadzone_frontier_plot(comparison_type, main_scenario, year, scenario, country, data_loader):
    """
    Create dead zone frontier plot based on comparison type

    Types:
    - 'years': Fixed spatial scope & scenario, compare years
    - 'scenarios_years': Fixed spatial scope, compare scenarios & years
    - 'spatial': Fixed scenario & year, compare spatial scopes

    country can be a single value or a list (max 2 for types 1 & 2)
    """
    try:
        fig = go.Figure()

        # Color palette
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

        # Define line styles
        line_styles_base = ['solid', 'dash', 'dot', 'dashdot']

        stats = data_loader.get_summary_stats(main_scenario)
        all_years = stats.get('years', [])
        all_scenarios = stats.get('scenarios', [])

        # Handle country as list or single value
        if isinstance(country, list):
            countries = country[:2]  # Limit to 2 countries
        else:
            countries = [country]

        trace_idx = 0

        # Handle scenario as list or single value
        if isinstance(scenario, list):
            scenarios_to_use = [s for s in scenario if s != 'all']
            if not scenarios_to_use or 'all' in scenario:
                scenarios_to_use = all_scenarios
        else:
            scenarios_to_use = all_scenarios if scenario == 'all' else [scenario]

        if comparison_type == 'years':
            # Type 1: Fixed spatial scope & scenario, compare years
            # Can now compare 1-2 countries and 1-2 scenarios
            years_to_plot = all_years if year == 'all' else [year]
            scenarios_to_plot = scenarios_to_use[:2]  # Limit to 2 scenarios

            for scen in scenarios_to_plot:
                for ctry in countries:
                    for yr in years_to_plot:
                        frontier_dict = data_loader.get_frontier_data(main_scenario, yr, ctry)
                        if scen in frontier_dict:
                            frontier_values = frontier_dict[scen]
                            x_values = np.arange(len(frontier_values))
                            x_values_extended = np.append(x_values, len(frontier_values))
                            frontier_values_extended = np.append(frontier_values, 0)

                            # Build label based on what's being compared
                            label_parts = []
                            if len(scenarios_to_plot) > 1:
                                label_parts.append(format_scenario_name(scen))
                            if len(countries) > 1:
                                label_parts.append(ctry)
                            label_parts.append(str(yr))
                            label = " - ".join(label_parts)

                            fig.add_trace(go.Scatter(
                                x=x_values_extended,
                                y=frontier_values_extended,
                                mode='lines',
                                name=label,
                                line=dict(
                                    color=colors[trace_idx % len(colors)],
                                    dash=line_styles_base[trace_idx % len(line_styles_base)],
                                    width=2.5
                                ),
                                hovertemplate=f"{label}<br>Point: %{{x}}<br>Hourly Matching: %{{y:.2f}}%<extra></extra>"
                            ))
                            trace_idx += 1

            # Build title
            title_parts = []
            if len(countries) == 1:
                title_parts.append(countries[0])
            else:
                title_parts.append(f"{countries[0]} vs {countries[1]}")

            if len(scenarios_to_plot) == 1:
                title_parts.append(format_scenario_name(scenarios_to_plot[0]))
            else:
                title_parts.append(f"{format_scenario_name(scenarios_to_plot[0])} vs {format_scenario_name(scenarios_to_plot[1])}")

            title = f"Frontier Evolution Over Time<br><sub>{' - '.join(title_parts)}</sub>"

        elif comparison_type == 'scenarios_years':
            # Type 2: Fixed spatial scope, compare scenarios & years
            # Can now compare 1 or 2 countries
            years_to_plot = all_years if year == 'all' else [year]
            scenarios_to_plot = scenarios_to_use  # Use processed scenarios

            for ctry in countries:
                for yr in years_to_plot:
                    frontier_dict = data_loader.get_frontier_data(main_scenario, yr, ctry)
                    for scen in scenarios_to_plot:
                        if scen in frontier_dict:
                            frontier_values = frontier_dict[scen]
                            x_values = np.arange(len(frontier_values))
                            x_values_extended = np.append(x_values, len(frontier_values))
                            frontier_values_extended = np.append(frontier_values, 0)

                            # Label includes country if comparing 2
                            if len(countries) == 1:
                                label = f"{format_scenario_name(scen)} ({yr})"
                            else:
                                label = f"{ctry} - {format_scenario_name(scen)} ({yr})"

                            fig.add_trace(go.Scatter(
                                x=x_values_extended,
                                y=frontier_values_extended,
                                mode='lines',
                                name=label,
                                line=dict(
                                    color=colors[trace_idx % len(colors)],
                                    dash=line_styles_base[trace_idx % len(line_styles_base)],
                                    width=2
                                ),
                                hovertemplate=f"{label}<br>Point: %{{x}}<br>Hourly Matching: %{{y:.2f}}%<extra></extra>"
                            ))
                            trace_idx += 1

            country_label = countries[0] if len(countries) == 1 else f"{countries[0]} vs {countries[1]}"
            title = f"Frontier Comparison Across Scenarios and Years<br><sub>{country_label}</sub>"

        elif comparison_type == 'spatial':
            # Type 3: Fixed scenario & year, compare spatial scopes
            year_to_use = year if year != 'all' else all_years[-1]
            scenario_to_use = scenarios_to_use[0] if scenarios_to_use else 'baseline'

            # Get all countries if 'all' is selected
            if isinstance(country, list) and 'all' in country:
                spatial_countries = data_loader.get_frontier_countries(main_scenario, year_to_use)
            elif isinstance(country, list):
                spatial_countries = [c for c in country if c != 'all']
            else:
                spatial_countries = [country] if country != 'all' else data_loader.get_frontier_countries(main_scenario, year_to_use)

            for ctry in spatial_countries:
                frontier_dict = data_loader.get_frontier_data(main_scenario, year_to_use, ctry)
                if scenario_to_use in frontier_dict:
                    frontier_values = frontier_dict[scenario_to_use]
                    x_values = np.arange(len(frontier_values))
                    x_values_extended = np.append(x_values, len(frontier_values))
                    frontier_values_extended = np.append(frontier_values, 0)

                    fig.add_trace(go.Scatter(
                        x=x_values_extended,
                        y=frontier_values_extended,
                        mode='lines',
                        name=ctry,
                        line=dict(
                            color=colors[trace_idx % len(colors)],
                            dash=line_styles_base[trace_idx % len(line_styles_base)],
                            width=2
                        ),
                        hovertemplate=f"{ctry}<br>Point: %{{x}}<br>Hourly Matching: %{{y:.2f}}%<extra></extra>"
                    ))
                    trace_idx += 1

            title = f"Results Frontier Comparison<br><sub>{format_scenario_name(scenario_to_use)} - {year_to_use}</sub>"

        else:
            return create_empty_figure("Invalid comparison type")

        if trace_idx == 0:
            return create_empty_figure("No data available for selected parameters")

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Energy Matching Point",
            yaxis_title="Hourly Matching (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[40, 100]),
            template="plotly_white",
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)"
            ),
            margin=dict(l=60, r=40, t=100, b=120)
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_empty_figure(f"Error creating frontier plot: {str(e)}")


def create_timeseries_plot(year, scenarios, ts_type, country, carriers, time_range, plot_type, data_loader):
    """Create timeseries plot for selected parameters - matching notebook style"""
    try:
        # Load the timeseries data
        result = data_loader.load_timeseries_data(year, scenarios, ts_type, country, carriers, time_range)

        if result is None:
            return create_empty_figure("No timeseries data available for selected parameters")

        df, timestamp_cols = result

        if df.empty:
            return create_empty_figure("No data found for selected parameters")

        # Create figure
        fig = go.Figure()

        # Color palette - using Plotly colors
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

        # Group by scenario
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_data = df[df['scenario'] == scenario]

            if plot_type == 'area':
                # Separate positive and negative values for stacked area (like notebook)
                # Collect all carrier data
                carrier_data = {}
                y_label = None

                for _, row in scenario_data.iterrows():
                    carrier = row['carrier']
                    y_label = row['y_label']
                    values = row[timestamp_cols].values.astype(float)
                    carrier_data[carrier] = values

                # Create traces for positive values (generation)
                trace_idx = scenario_idx * 20  # Offset for multiple scenarios
                positive_carriers = []
                negative_carriers = []

                for carrier, values in carrier_data.items():
                    # Separate into positive and negative
                    positive_values = np.maximum(values, 0)
                    negative_values = np.minimum(values, 0)

                    # Add positive trace
                    if positive_values.sum() > 0:
                        label = f"{scenario} - {carrier}" if len(scenarios) > 1 else carrier
                        fig.add_trace(go.Scatter(
                            x=list(range(len(timestamp_cols))),
                            y=positive_values,
                            mode='lines',
                            name=label,
                            stackgroup='positive' + str(scenario_idx),
                            line=dict(width=0.5, color=colors[trace_idx % len(colors)]),
                            fillcolor=colors[trace_idx % len(colors)],
                            hovertemplate=f"{label}<br>Value: %{{y:.2f}} GW<extra></extra>"
                        ))
                        positive_carriers.append(carrier)
                        trace_idx += 1

                    # Add negative trace (charging/consumption)
                    if negative_values.sum() < 0:
                        label = f"{scenario} - {carrier}" if len(scenarios) > 1 else carrier
                        fig.add_trace(go.Scatter(
                            x=list(range(len(timestamp_cols))),
                            y=negative_values,
                            mode='lines',
                            name=label,
                            stackgroup='negative' + str(scenario_idx),
                            line=dict(width=0.5, color=colors[trace_idx % len(colors)]),
                            fillcolor=colors[trace_idx % len(colors)],
                            showlegend=False,  # Don't show twice in legend
                            hovertemplate=f"{label}<br>Value: %{{y:.2f}} GW<extra></extra>"
                        ))
                        negative_carriers.append(carrier)

                # Add demand line (negative of 'electricity' carrier)
                if 'electricity' in carrier_data:
                    electricity_values = carrier_data['electricity']
                    demand_values = -electricity_values  # Flip sign to show demand
                    fig.add_trace(go.Scatter(
                        x=list(range(len(timestamp_cols))),
                        y=demand_values,
                        mode='lines',
                        name='Demand (electricity)',
                        line=dict(color='black', width=2, dash='dash'),
                        hovertemplate="Demand<br>Value: %{y:.2f} GW<extra></extra>"
                    ))

            else:  # Line plot
                trace_idx = scenario_idx * 20
                electricity_values = None

                for _, row in scenario_data.iterrows():
                    carrier = row['carrier']
                    y_label = row['y_label']
                    values = row[timestamp_cols].values.astype(float)

                    # Skip electricity carrier - we'll plot it as demand line instead
                    if carrier == 'electricity':
                        electricity_values = values
                        continue

                    label = f"{scenario} - {carrier}" if len(scenarios) > 1 else carrier

                    fig.add_trace(go.Scatter(
                        x=list(range(len(timestamp_cols))),
                        y=values,
                        mode='lines',
                        name=label,
                        line=dict(color=colors[trace_idx % len(colors)], width=2),
                        hovertemplate=f"{label}<br>Value: %{{y:.2f}}<extra></extra>"
                    ))
                    trace_idx += 1

                # Add demand line (negative of 'electricity' carrier) for line plots
                if electricity_values is not None:
                    demand_values = -electricity_values  # Flip sign to show demand
                    fig.add_trace(go.Scatter(
                        x=list(range(len(timestamp_cols))),
                        y=demand_values,
                        mode='lines',
                        name='Demand',
                        line=dict(color='black', width=2, dash='dash'),
                        hovertemplate="Demand<br>Value: %{y:.2f} GW<extra></extra>"
                    ))

        # Update layout
        # Create x-axis labels (sample every N timestamps to avoid crowding)
        num_labels = min(10, len(timestamp_cols))
        step = max(1, len(timestamp_cols) // num_labels)
        tickvals = list(range(0, len(timestamp_cols), step))
        # Format as month-day only (remove year)
        ticktext = []
        for i in tickvals:
            ts = timestamp_cols[i]
            # Handle both string and Timestamp objects
            if isinstance(ts, str):
                # CSV format: "2013-01-01 00:00:00" -> "01-01"
                ticktext.append('-'.join(ts.split()[0].split('-')[1:]))
            else:
                # Parquet Timestamp object -> "01-01"
                ticktext.append(f"{ts.month:02d}-{ts.day:02d}")

        # Get y_label
        y_label = df['y_label'].iloc[0] if len(df) > 0 else "Value"

        fig.update_layout(
            title=f"{ts_type} - {country} ({year})<br><sub>{', '.join(scenarios[:3])}{'...' if len(scenarios) > 3 else ''}</sub>",
            xaxis_title="Time",
            yaxis_title=y_label,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=-45
            ),
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            margin=dict(l=60, r=150, t=100, b=100)
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_empty_figure(f"Error creating timeseries plot: {str(e)}")
