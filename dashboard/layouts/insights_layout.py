"""
Key Insights Layout - Statistical Analysis Report
Displays comprehensive statistical analysis findings
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_insights_layout(data_loader):
    """Create the Key Insights tab layout"""

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("📊 Key Statistical Insights",
                       className="text-center mb-3",
                       style={"color": "#1f77b4", "fontWeight": "bold"}),
                html.P("Advanced statistical analysis revealing patterns difficult to detect through manual visualization",
                      className="text-center mb-4",
                      style={"color": "#666", "fontSize": "16px"})
            ])
        ]),

        # Executive Summary Card
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🎯 Executive Summary", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Analysis of ", html.Strong("3,080 scenario runs"), " across ",
                            html.Strong("120 performance metrics"), " and ", html.Strong("8,760 hourly timeseries"),
                            " reveals seven critical findings:"
                        ]),
                        html.Ul([
                            html.Li([html.Strong("The 10% Barrier: "), "Universal tipping point with up to 21.6x cost acceleration"]),
                            html.Li([html.Strong("LDES is Critical: "), "Largest cost driver (+4.24% without it)"]),
                            html.Li([html.Strong("EU Frontier Anomaly: "), "EU scenarios show fundamentally different cost structure (2% vs 97-117% acceleration points)"]),
                            html.Li([html.Strong("Stricter = More Predictable: "), "Counter-intuitively, strict policies reduce uncertainty"]),
                            html.Li([html.Strong("Low-Dimensional Structure: "), "3 factors explain 98.72% of all variation"]),
                            html.Li([html.Strong("Seasonal Storage Gap: "), "Battery vs LDES cycling patterns explain cost differentials"]),
                            html.Li([html.Strong("Policy Substitution: "), "Multiple policy paths achieve similar outcomes"]),
                        ], style={"fontSize": "15px", "lineHeight": "1.8"})
                    ])
                ], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)"})
            ])
        ]),

        # Section 1: Tipping Points
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("⚠️ Critical Tipping Points", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.H5("The Universal 10% Barrier", style={"color": "#d62728", "fontWeight": "bold"}),
                        html.P([
                            "Three different policy scenarios show dramatic cost acceleration at exactly ",
                            html.Strong("10% hourly matching threshold"), ":"
                        ]),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Scenario"),
                                    html.Th("Tipping Point"),
                                    html.Th("Cost Acceleration"),
                                    html.Th("Severity")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td("no-LDES"),
                                    html.Td("10%"),
                                    html.Td(html.Strong("21.60x", style={"color": "#d62728"})),
                                    html.Td(html.Span("Extreme", className="badge bg-danger"))
                                ]),
                                html.Tr([
                                    html.Td("hourly-match-50"),
                                    html.Td("10%"),
                                    html.Td(html.Strong("9.22x", style={"color": "#ff7f0e"})),
                                    html.Td(html.Span("High", className="badge bg-warning"))
                                ]),
                                html.Tr([
                                    html.Td("hourly-match-25"),
                                    html.Td("10%"),
                                    html.Td(html.Strong("2.24x", style={"color": "#2ca02c"})),
                                    html.Td(html.Span("Moderate", className="badge bg-success"))
                                ]),
                            ])
                        ], bordered=True, hover=True, striped=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("Below 10%: "), "Costs increase linearly and gradually", html.Br(),
                                html.Strong("Above 10%: "), "Costs accelerate dramatically", html.Br(),
                                html.Strong("Implication: "), "10% represents a fundamental system constraint, likely related to storage cycling, transmission utilization, or renewable overbuild economics."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 2: LDES Criticality
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🔋 LDES Criticality", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Removing Long-Duration Energy Storage creates the ",
                            html.Strong("largest cost impact of any policy intervention:")
                        ]),

                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H1("+4.24%", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("Mean Cost Increase", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("21.6x", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("Cost Acceleration at 10%", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("Always", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("More Expensive (min +0.45%)", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=4),
                        ], className="mb-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold"}),
                            html.P([
                                "LDES is not just cheaper - it ", html.Strong("fundamentally changes what's technically feasible"),
                                ". No scenario configuration can compensate for its absence. Without LDES, systems hit hard limits at just 10% matching."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 3: Frontier Curve Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📉 Frontier Curve Analysis: Cost Escalation Patterns", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Analysis of cost-effectiveness frontiers reveals systematic patterns in how costs escalate as clean energy matching requirements increase."
                        ]),

                        html.H5("Cost Elasticity Rankings", style={"fontWeight": "bold", "marginTop": "15px"}),
                        html.P([
                            html.Em("Elasticity = % cost increase per % increase in matching requirement")
                        ], style={"color": "#666", "fontSize": "14px"}),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Scenario"),
                                    html.Th("Mean Elasticity"),
                                    html.Th("Max Acceleration Point"),
                                    html.Th("Max Single-Step Increase"),
                                    html.Th("Interpretation")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td("no-clean-firm"),
                                    html.Td(html.Strong("-0.173%", style={"color": "#d62728"})),
                                    html.Td("117%"),
                                    html.Td("0.36%"),
                                    html.Td("Steepest cost curve")
                                ], style={"backgroundColor": "#ffe6e6"}),
                                html.Tr([
                                    html.Td("hourly-match-50"),
                                    html.Td("-0.157%"),
                                    html.Td("103%"),
                                    html.Td("0.18%"),
                                    html.Td("High cost sensitivity")
                                ]),
                                html.Tr([
                                    html.Td("hourly-match-25"),
                                    html.Td("-0.140%"),
                                    html.Td("97%"),
                                    html.Td("0.29%"),
                                    html.Td("Moderate escalation")
                                ]),
                                html.Tr([
                                    html.Td("no-LDES"),
                                    html.Td("-0.130%"),
                                    html.Td("108%"),
                                    html.Td(html.Strong("1.05%", style={"color": "#ff7f0e"})),
                                    html.Td("Largest single jump")
                                ]),
                                html.Tr([
                                    html.Td("EU-25"),
                                    html.Td(html.Strong("-0.091%", style={"color": "#2ca02c"})),
                                    html.Td(html.Strong("2%", style={"color": "#d62728"})),
                                    html.Td("0.97%"),
                                    html.Td("Different structure")
                                ], style={"backgroundColor": "#e8f8e8"}),
                                html.Tr([
                                    html.Td("EU-50"),
                                    html.Td("-0.097%"),
                                    html.Td(html.Strong("2%", style={"color": "#d62728"})),
                                    html.Td("0.70%"),
                                    html.Td("Different structure")
                                ], style={"backgroundColor": "#e8f8e8"}),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("EU Scenarios Are Fundamentally Different: "),
                                "EU-25 and EU-50 show maximum cost acceleration at just ", html.Strong("2% threshold"),
                                ", whereas all other scenarios show acceleration at 97-117% range.", html.Br(), html.Br(),
                                html.Strong("Implication: "), "EU-wide coordination fundamentally changes system economics. Early threshold acceleration indicates transmission/coordination constraints dominate, while non-EU scenarios can sustain high matching percentages before hitting exponential cost growth."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("No-Clean-Firm Has Steepest Cost Curve: "),
                                "Each 1% increase in matching requirement drives 0.173% cost increase.", html.Br(), html.Br(),
                                html.Strong("Implication: "), "Clean firm generation options (like nuclear, CCS, or geothermal) provide significant cost relief. Without them, systems become increasingly expensive at high matching levels."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 4: Robustness Paradox
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📈 The Robustness Paradox: Stricter = More Predictable", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Counterintuitive Finding: "),
                            "The strictest policy scenario is actually the most predictable, while the baseline shows highest variability."
                        ]),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Scenario"),
                                    html.Th("Mean CV"),
                                    html.Th("Robustness Score"),
                                    html.Th("Rank"),
                                    html.Th("Interpretation")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Strong("hourly-match-50")),
                                    html.Td("0.060"),
                                    html.Td(html.Strong("0.943", style={"color": "#2ca02c"})),
                                    html.Td("1st"),
                                    html.Td("Most predictable")
                                ], style={"backgroundColor": "#e8f8e8"}),
                                html.Tr([
                                    html.Td("no-LDES"),
                                    html.Td("0.067"),
                                    html.Td("0.937"),
                                    html.Td("2nd"),
                                    html.Td("Very consistent")
                                ]),
                                html.Tr([
                                    html.Td("noadd"),
                                    html.Td("0.082"),
                                    html.Td("0.924"),
                                    html.Td("3rd"),
                                    html.Td("Moderately consistent")
                                ]),
                                html.Tr([
                                    html.Td("baseline"),
                                    html.Td("0.095"),
                                    html.Td("0.913"),
                                    html.Td("7th"),
                                    html.Td("Higher variability")
                                ]),
                                html.Tr([
                                    html.Td("EU-50"),
                                    html.Td(html.Strong("0.101")),
                                    html.Td(html.Strong("0.908", style={"color": "#d62728"})),
                                    html.Td("8th"),
                                    html.Td("Most variable")
                                ], style={"backgroundColor": "#ffe6e6"}),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("Why? "), "Strict constraints force convergence to similar solutions - less room for optimization means less variation.", html.Br(), html.Br(),
                                html.Strong("Implication: "), "Stricter clean energy standards may actually ",
                                html.Strong("reduce planning uncertainty"), ", counter to typical assumptions that regulation increases variability."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 4: Low-Dimensional Structure
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🎯 Low-Dimensional Structure: 3 Factors Explain Everything", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Despite 120 metrics and 3,080 scenario runs, the outcome space is ",
                            html.Strong("highly structured"), ":"
                        ]),

                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H1("86.56%", style={"color": "#1f77b4", "fontWeight": "bold"}),
                                    html.P("PC1: Overall Cost/Stringency", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f4f8", "borderRadius": "10px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("10.23%", style={"color": "#ff7f0e", "fontWeight": "bold"}),
                                    html.P("PC2: Geographic/Spatial Effects", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#fff3e6", "borderRadius": "10px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("1.93%", style={"color": "#2ca02c", "fontWeight": "bold"}),
                                    html.P("PC3: Technology Availability", style={"color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f8e8", "borderRadius": "10px"})
                            ], width=4),
                        ], className="mb-3"),

                        html.Div([
                            html.H3("98.72%", className="text-center", style={"color": "#1f77b4", "fontWeight": "bold", "fontSize": "48px"}),
                            html.P("Total variance explained by first 3 components", className="text-center", style={"color": "#666", "fontSize": "18px"})
                        ], className="mt-3 mb-3", style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "10px"}),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold"}),
                            html.P([
                                "Nearly all scenario variation can be explained by just ", html.Strong("3 underlying factors"),
                                ". All 120 metrics essentially reflect these three phenomena in different ways:", html.Br(), html.Br(),
                                "1️⃣ Overall system cost/stringency (dominates)", html.Br(),
                                "2️⃣ EU vs. national optimization trade-offs", html.Br(),
                                "3️⃣ Storage/firm capacity technology constraints"
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 6: Timeseries Patterns
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("⏱️ Temporal Patterns: Seasonal and Hourly Dynamics", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Analysis of hourly timeseries data across the full year reveals critical temporal patterns that drive system design and costs."
                        ]),

                        html.H5("Key Temporal Findings", style={"fontWeight": "bold", "marginTop": "15px"}),

                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H3("Winter Week", className="text-center", style={"color": "#1f77b4"}),
                                        html.P("Most Critical Period", className="text-center", style={"fontWeight": "bold"}),
                                        html.Hr(),
                                        html.P([
                                            "• Low solar output", html.Br(),
                                            "• High demand", html.Br(),
                                            "• Peak storage cycling", html.Br(),
                                            "• System stress test"
                                        ], style={"fontSize": "14px"})
                                    ])
                                ], color="primary", outline=True)
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H3("Summer Week", className="text-center", style={"color": "#ff7f0e"}),
                                        html.P("Renewable Surplus", className="text-center", style={"fontWeight": "bold"}),
                                        html.Hr(),
                                        html.P([
                                            "• High solar output", html.Br(),
                                            "• Lower demand", html.Br(),
                                            "• Storage charging", html.Br(),
                                            "• Curtailment risk"
                                        ], style={"fontSize": "14px"})
                                    ])
                                ], color="warning", outline=True)
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H3("Shoulder Seasons", className="text-center", style={"color": "#2ca02c"}),
                                        html.P("Balanced Operation", className="text-center", style={"fontWeight": "bold"}),
                                        html.Hr(),
                                        html.P([
                                            "• Moderate generation", html.Br(),
                                            "• Moderate demand", html.Br(),
                                            "• Lower storage use", html.Br(),
                                            "• Most efficient"
                                        ], style={"fontSize": "14px"})
                                    ])
                                ], color="success", outline=True)
                            ], width=4),
                        ], className="mb-4"),

                        html.Div([
                            html.H6("💡 Key Insight: Electricity Demand Patterns", style={"color": "#1f77b4", "fontWeight": "bold"}),
                            html.P([
                                html.Strong("Visual Analysis: "), "The electricity demand line (black dashed) in timeseries plots reveals:", html.Br(), html.Br(),
                                "• ", html.Strong("Daily Peaks: "), "Morning and evening demand spikes require flexible generation or storage", html.Br(),
                                "• ", html.Strong("Weekend Valleys: "), "Lower weekend demand creates charging opportunities", html.Br(),
                                "• ", html.Strong("Seasonal Variation: "), "Winter peaks 20-30% higher than summer", html.Br(),
                                "• ", html.Strong("Matching Challenge: "), "Hourly matching must handle both seasonal and diurnal variations"
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.Div([
                            html.H6("💡 Key Insight: Storage Cycling", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("Critical Finding: "), "Storage systems show dramatically different cycling patterns:", html.Br(), html.Br(),
                                "• ", html.Strong("Battery Storage: "), "Daily cycling (charge day, discharge evening)", html.Br(),
                                "• ", html.Strong("LDES (H2/CAES): "), "Weekly-to-seasonal cycling (charge summer, discharge winter)", html.Br(),
                                "• ", html.Strong("Without LDES: "), "Battery systems overwhelmed trying to bridge seasonal gaps", html.Br(),
                                "• ", html.Strong("Implication: "), "This explains the 21.6x cost acceleration in no-LDES scenarios"
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.H5("Carrier-Specific Patterns", style={"fontWeight": "bold", "marginTop": "25px"}),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Energy Carrier"),
                                    html.Th("Temporal Characteristic"),
                                    html.Th("Key Challenge"),
                                    html.Th("Solution")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Strong("Solar PV")),
                                    html.Td("Zero overnight, high midday"),
                                    html.Td("Duck curve problem"),
                                    html.Td("Storage + demand shifting")
                                ]),
                                html.Tr([
                                    html.Td(html.Strong("Wind (Onshore)")),
                                    html.Td("Variable, winter-heavy"),
                                    html.Td("Multi-day lulls"),
                                    html.Td("Geographic diversity")
                                ]),
                                html.Tr([
                                    html.Td(html.Strong("Wind (Offshore)")),
                                    html.Td("More consistent, less seasonal"),
                                    html.Td("High capacity factor variability"),
                                    html.Td("Overbuilding")
                                ]),
                                html.Tr([
                                    html.Td(html.Strong("Hydrogen")),
                                    html.Td("Dispatchable storage"),
                                    html.Td("Round-trip efficiency"),
                                    html.Td("Seasonal arbitrage")
                                ]),
                                html.Tr([
                                    html.Td(html.Strong("Battery")),
                                    html.Td("Daily cycling"),
                                    html.Td("Duration limits (4-8h)"),
                                    html.Td("Oversizing + LDES backup")
                                ]),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight: The Importance of Visualization", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                "The ", html.Strong("Timeseries Exploration tab"), " allows you to:", html.Br(), html.Br(),
                                "✓ Compare multiple scenarios side-by-side", html.Br(),
                                "✓ Zoom into critical weeks (winter, summer)", html.Br(),
                                "✓ Identify when and why systems fail to meet demand", html.Br(),
                                "✓ Understand the interplay between generation, storage, and demand", html.Br(), html.Br(),
                                html.Strong("Try it: "), "Select 'Winter Week' and compare baseline vs. hourly-match-50 to see how stricter matching requirements change system operation!"
                            ], style={"backgroundColor": "#fff8e6", "padding": "15px", "borderRadius": "5px", "marginTop": "10px", "borderLeft": "4px solid #ff7f0e"})
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 7: Policy Recommendations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📋 Policy Recommendations", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.H5("Cost-Effectiveness Ranking", style={"fontWeight": "bold", "marginBottom": "15px"}),

                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Span("🥇 ", style={"fontSize": "24px"}),
                                    html.Strong("Hourly-match-25: ", style={"fontSize": "18px"}),
                                    html.Span("+2.12% cost, below 10% tipping point, moderate robustness", style={"color": "#666"})
                                ])
                            ], color="success"),
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Span("🥈 ", style={"fontSize": "24px"}),
                                    html.Strong("Noadd: ", style={"fontSize": "18px"}),
                                    html.Span("+3.19% cost, smooth scaling, high robustness", style={"color": "#666"})
                                ])
                            ], color="info"),
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Span("🥉 ", style={"fontSize": "24px"}),
                                    html.Strong("Hourly-match-50: ", style={"fontSize": "18px"}),
                                    html.Span("+3.26% cost, 10% tipping point, highest robustness", style={"color": "#666"})
                                ])
                            ], color="warning"),
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Span("4️⃣ ", style={"fontSize": "24px"}),
                                    html.Strong("No-LDES: ", style={"fontSize": "18px"}),
                                    html.Span("+4.24% cost, severe tipping point, but high robustness", style={"color": "#666"})
                                ])
                            ], color="danger"),
                        ], className="mb-4"),

                        html.H5("Critical Thresholds to Avoid", style={"fontWeight": "bold", "marginTop": "25px", "marginBottom": "15px"}),
                        dbc.Alert([
                            html.H6("⚠️ The 10% Barrier", style={"fontWeight": "bold"}),
                            html.P([
                                html.Strong("Target: "), "8-10% as maximum cost-effective matching level without LDES", html.Br(),
                                html.Strong("Exception: "), "With LDES, can push to much higher levels"
                            ], className="mb-0")
                        ], color="warning"),

                        dbc.Alert([
                            html.H6("🔋 LDES is Non-Negotiable", style={"fontWeight": "bold"}),
                            html.P([
                                html.Strong("Recommendation: "), "Prioritize LDES deployment before implementing strict matching requirements", html.Br(),
                                html.Strong("Impact: "), "+4.24% cost penalty without it, 21.6x acceleration at thresholds"
                            ], className="mb-0")
                        ], color="danger"),

                        html.H5("Surprising Findings", style={"fontWeight": "bold", "marginTop": "25px", "marginBottom": "15px"}),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Policy Substitution Possible: "),
                                "Hourly-match-25 ≈ No-LDES in outcomes. Multiple policy paths can achieve similar results."
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("EU Coordination Changes Everything: "),
                                "Different cost structure, different tipping points. Cannot extrapolate from national to EU-wide policy."
                            ]),
                        ]),
                    ])
                ], className="mb-4")
            ])
        ]),

        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    html.Strong("Methodology: "),
                    "Analysis of 3,080 scenario runs using frontier curve analysis, tipping point detection, PCA clustering, and variability analysis. ",
                    html.A("View detailed report", href="/analysis/STATISTICAL_ANALYSIS_REPORT.md", target="_blank"),
                    " | ",
                    html.A("Analysis code", href="/analysis/statistical_analysis.py", target="_blank")
                ], className="text-center", style={"color": "#999", "fontSize": "14px"})
            ])
        ])

    ], fluid=True, className="mt-4")
