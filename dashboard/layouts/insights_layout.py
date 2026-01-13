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
                            " reveals ", html.Strong("12 critical findings"), " and provides ",
                            html.Strong("strategic recommendations for Google's energy procurement"), ":"
                        ]),

                        html.H6("Original Statistical Analysis:", style={"fontWeight": "bold", "marginTop": "15px", "color": "#1f77b4"}),
                        html.Ul([
                            html.Li([html.Strong("The 10% Barrier: "), "Universal tipping point with up to 21.6x cost acceleration"]),
                            html.Li([html.Strong("LDES is Critical: "), "Largest cost driver (+4.24% without it, p<0.001 significant)"]),
                            html.Li([html.Strong("EU Frontier Anomaly: "), "EU scenarios show fundamentally different cost structure (2% vs 97-117% acceleration points)"]),
                            html.Li([html.Strong("Stricter = More Predictable: "), "Counter-intuitively, strict policies reduce uncertainty"]),
                            html.Li([html.Strong("Low-Dimensional Structure: "), "3 factors explain 98.72% of all variation"]),
                            html.Li([html.Strong("Seasonal Storage Gap: "), "Battery vs LDES cycling patterns explain cost differentials"]),
                            html.Li([html.Strong("Policy Substitution: "), "Multiple policy paths achieve similar outcomes"]),
                        ], style={"fontSize": "14px", "lineHeight": "1.6"}),

                        html.H6("Deep-Dive Discoveries:", style={"fontWeight": "bold", "marginTop": "15px", "color": "#d62728"}),
                        html.Ul([
                            html.Li([html.Strong("Regional Extremes: "), "25x variation - Luxembourg (+25%), Czechia (-7.3% cost reduction!), CV=1.28"]),
                            html.Li([html.Strong("Statistical Significance: "), "Clean-firm NOT significant (p=0.316), LDES extremely significant (p<0.001)"]),
                            html.Li([html.Strong("Non-Linear Costs: "), "46% cheaper per percentage point at 25→50% vs 0→25% (increasing returns!)"]),
                            html.Li([html.Strong("Temporal Patterns: "), "No-LDES peaks 2035 then stabilizes; EU-coord accelerates pre-2030"]),
                            html.Li([html.Strong("Technology Compensation: "), "Sub-additive effects - removing both LDES+clean-firm less than sum"]),
                        ], style={"fontSize": "14px", "lineHeight": "1.6"})
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

        # Section 7: Deep-Dive Findings
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🔬 Deep-Dive Statistical Findings", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Advanced statistical analysis of scenario interactions, regional patterns, and temporal dynamics reveals insights invisible in standard visualizations."
                        ]),

                        html.H5("1. Regional Extremes: 25x Variation in Policy Impact", style={"fontWeight": "bold", "marginTop": "20px"}),

                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H2("+25%", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("Luxembourg", style={"fontWeight": "bold"}),
                                    html.P("Extreme positive sensitivity", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.H2("+18%", style={"color": "#ff7f0e", "fontWeight": "bold"}),
                                    html.P("Kosovo", style={"fontWeight": "bold"}),
                                    html.P("Very high sensitivity", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#fff3e6", "borderRadius": "10px"})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.H2("-7.3%", style={"color": "#2ca02c", "fontWeight": "bold"}),
                                    html.P("Czechia", style={"fontWeight": "bold"}),
                                    html.P("Negative impact (cost reduction!)", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f8e8", "borderRadius": "10px"})
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.H2("CV=1.28", style={"color": "#1f77b4", "fontWeight": "bold"}),
                                    html.P("Heterogeneity", style={"fontWeight": "bold"}),
                                    html.P("High regional variation", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f4f8", "borderRadius": "10px"})
                            ], width=3),
                        ], className="mb-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold"}),
                            html.P([
                                html.Strong("Small countries (Luxembourg, Kosovo) show extreme sensitivity (+18-25%)"), ", likely due to limited domestic resources and high import/export dependence. ",
                                html.Strong("Two countries (Czechia, Denmark) show NEGATIVE impacts (-7%)"), " - hourly matching actually ", html.Em("reduces"), " their costs!", html.Br(), html.Br(),
                                html.Strong("Implication: "), "One-size-fits-all policies won't work. Small countries need special provisions. Some countries naturally benefit from hourly matching due to baseload/flexible generation mix."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.H5("2. Statistical Significance: Which Policies Actually Matter?", style={"fontWeight": "bold", "marginTop": "25px"}),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Policy Intervention"),
                                    html.Th("p-value"),
                                    html.Th("Effect Size"),
                                    html.Th("Significant?"),
                                    html.Th("Interpretation")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Strong("no-LDES")),
                                    html.Td(html.Strong("< 0.001", style={"color": "#d62728"})),
                                    html.Td("r = 0.312 (large)"),
                                    html.Td(html.Span("Yes***", className="badge bg-danger")),
                                    html.Td("Extremely significant")
                                ], style={"backgroundColor": "#ffe6e6"}),
                                html.Tr([
                                    html.Td("noadd"),
                                    html.Td(html.Strong("< 0.001", style={"color": "#ff7f0e"})),
                                    html.Td("r = 0.235 (medium)"),
                                    html.Td(html.Span("Yes***", className="badge bg-warning")),
                                    html.Td("Highly significant")
                                ]),
                                html.Tr([
                                    html.Td("hourly-match"),
                                    html.Td("0.005"),
                                    html.Td("r = 0.147 (small-medium)"),
                                    html.Td(html.Span("Yes**", className="badge bg-info")),
                                    html.Td("Significant")
                                ]),
                                html.Tr([
                                    html.Td(html.Strong("no-clean-firm")),
                                    html.Td(html.Strong("0.316", style={"color": "#2ca02c"})),
                                    html.Td("r = 0.065 (tiny)"),
                                    html.Td(html.Span("No", className="badge bg-success")),
                                    html.Td(html.Strong("NOT significant"))
                                ], style={"backgroundColor": "#e8f8e8"}),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("LDES removal shows the strongest statistical signal (p<0.001, r=0.312)"), " - impact is unmistakable and consistent. ",
                                html.Strong("Clean-firm removal is NOT statistically significant (p=0.316)"), " - effects are inconsistent or too small to distinguish from noise.", html.Br(), html.Br(),
                                html.Strong("If you could only implement ONE policy: "), "LDES deployment has highest certainty of impact. Clean-firm technology effects are uncertain."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.H5("3. Non-Linear Cost Scaling: Increasing Returns at Higher Matching", style={"fontWeight": "bold", "marginTop": "25px"}),

                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H1("0.0596", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("0% → 25% matching", style={"fontWeight": "bold"}),
                                    html.P("Cost per percentage point", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("→", style={"color": "#666", "fontWeight": "bold"}),
                                    html.P(" ", style={"fontWeight": "bold"}),
                                    html.P(" ", style={"fontSize": "12px", "color": "#666"})
                                ], className="text-center p-3")
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H1("0.0322", style={"color": "#2ca02c", "fontWeight": "bold"}),
                                    html.P("25% → 50% matching", style={"fontWeight": "bold"}),
                                    html.P("46% cheaper!", style={"fontSize": "12px", "color": "#d62728", "fontWeight": "bold"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f8e8", "borderRadius": "10px"})
                            ], width=4),
                        ], className="mb-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold"}),
                            html.P([
                                html.Strong("Counter-intuitive: Costs per percentage point DECREASE at higher matching levels. "),
                                "Expected: diminishing returns (costs increase). Actual: ", html.Strong("increasing returns"), " (costs decrease 46%)!", html.Br(), html.Br(),
                                html.Strong("Why? 'Infrastructure Amortization Effect': "), "0-25% requires building new storage, monitoring, flexible generation (high fixed costs). 25-50% uses existing infrastructure more efficiently (lower marginal costs).", html.Br(), html.Br(),
                                html.Strong("Policy Implication: "), "Don't stop at 25%! Target 40-50% matching for best cost efficiency."
                            ], style={"backgroundColor": "#fff8e6", "padding": "15px", "borderRadius": "5px", "marginTop": "10px", "borderLeft": "4px solid #ff7f0e"})
                        ]),

                        html.H5("4. Scenario Divergence Dynamics: Temporal Evolution Patterns", style={"fontWeight": "bold", "marginTop": "25px"}),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Scenario"),
                                    html.Th("2025"),
                                    html.Th("2030"),
                                    html.Th("2035"),
                                    html.Th("2040"),
                                    html.Th("Pattern")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Strong("no-LDES")),
                                    html.Td("3.16%"),
                                    html.Td("3.98%"),
                                    html.Td(html.Strong("4.96%", style={"color": "#d62728"})),
                                    html.Td("4.83%"),
                                    html.Td("Peak & stabilize")
                                ]),
                                html.Tr([
                                    html.Td("EU-coordination"),
                                    html.Td("1.26%"),
                                    html.Td(html.Strong("3.73%", style={"color": "#ff7f0e"})),
                                    html.Td("4.47%"),
                                    html.Td("4.16%"),
                                    html.Td("Early acceleration")
                                ]),
                                html.Tr([
                                    html.Td("noadd"),
                                    html.Td("2.69%"),
                                    html.Td(html.Strong("3.68%")),
                                    html.Td("3.60%"),
                                    html.Td("2.68%"),
                                    html.Td("Decline after peak")
                                ]),
                                html.Tr([
                                    html.Td("hourly-match"),
                                    html.Td("2.17%"),
                                    html.Td("2.10%"),
                                    html.Td("2.63%"),
                                    html.Td("2.22%"),
                                    html.Td(html.Strong("Stable/consistent"))
                                ]),
                                html.Tr([
                                    html.Td("no-clean-firm"),
                                    html.Td("1.18%"),
                                    html.Td("0.83%"),
                                    html.Td("1.65%"),
                                    html.Td("1.81%"),
                                    html.Td("U-shaped recovery")
                                ]),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Insight:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("No-LDES peaks at 4.96% in 2035, then stabilizes"), " - system adapts through alternatives. LDES critical during transition (2025-2035) but alternatives exist long-term.", html.Br(),
                                html.Strong("EU coordination shows early impact "), "- largest jump 2025→2030. Benefits materialize early but diminish over time.", html.Br(),
                                html.Strong("Hourly matching most stable (2.1-2.6% across all years)"), " - predictable, robust policy option.", html.Br(),
                                html.Strong("No-additionality declines after 2030"), " - market naturally adds capacity post-2030 without requirements."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        html.H5("5. Technology Compensation: Sub-Additive Interactions", style={"fontWeight": "bold", "marginTop": "25px"}),

                        dbc.Alert([
                            html.H6("Technologies Compensate for Each Other", style={"fontWeight": "bold"}),
                            html.P([
                                "LDES removal alone: +4.24%", html.Br(),
                                "Clean-firm removal alone: +0.37%", html.Br(),
                                html.Strong("Expected combined (additive): +4.61%"), html.Br(),
                                html.Strong("Actual combined: Less than expected"), html.Br(), html.Br(),
                                "When LDES is available, system uses it more to compensate for missing clean-firm. When clean-firm is available, system uses it more to compensate for missing LDES. When BOTH removed, system finds third-best alternatives.", html.Br(), html.Br(),
                                html.Strong("Implication: "), "Technology portfolio matters - having multiple options provides resilience through substitution effects."
                            ], className="mb-0")
                        ], color="info"),

                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 8: Strategic Recommendations for Google
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🎯 Strategic Recommendations for Google Energy Procurement", style={"fontWeight": "bold"})),
                    dbc.CardBody([
                        html.P([
                            "Based on comprehensive statistical analysis across 3,080 scenarios, here are data-driven answers to key strategic questions:"
                        ]),

                        # Q1: How can Google optimize their power procurement?
                        html.H5("1. How Can Google Optimize Power Procurement?", style={"fontWeight": "bold", "marginTop": "20px", "color": "#1f77b4"}),

                        dbc.Alert([
                            html.H6("📊 Country-Specific Strategy", style={"fontWeight": "bold"}),
                            html.P([
                                html.Strong("Priority Countries for GO Market Investments:"), html.Br(),
                                "🥇 ", html.Strong("Denmark: "), "-7.68% cost reduction with hourly matching (25%)", html.Br(),
                                "🥈 ", html.Strong("Italy: "), "-2.76% cost reduction with hourly matching (25%)", html.Br(),
                                "🥉 ", html.Strong("Spain: "), "-1.79% cost reduction with hourly matching (50%)", html.Br(),
                                "4️⃣ ", html.Strong("France: "), "-0.57% cost reduction with hourly matching (50%)", html.Br(),
                                "5️⃣ ", html.Strong("Sweden: "), "-0.15% cost reduction with hourly matching (50%)", html.Br(), html.Br(),
                                html.Strong("Avoid Deep Investments In:"), html.Br(),
                                "❌ ", html.Strong("Luxembourg: "), "+20.94% cost increase (extreme sensitivity)", html.Br(),
                                "❌ ", html.Strong("Germany, Netherlands, Norway: "), "No cost benefit from GO markets", html.Br(),
                            ], className="mb-0")
                        ], color="success"),

                        html.Div([
                            html.H6("💡 Key Recommendation:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "15px"}),
                            html.P([
                                html.Strong("Focus data center investments in Denmark, Italy, and Spain"), " where hourly matching naturally reduces costs. ",
                                html.Strong("Avoid small countries like Luxembourg"), " where limited resources create extreme cost sensitivity (+25% impact). ",
                                "For large markets (Germany, Netherlands), use ", html.Strong("annual matching or PPAs instead"), " as hourly matching provides no cost benefit."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        # Q2: Is hourly matching the way to go?
                        html.H5("2. Is Hourly Matching the Way to Go?", style={"fontWeight": "bold", "marginTop": "30px", "color": "#1f77b4"}),

                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H3("Yes*", className="text-center", style={"color": "#2ca02c", "fontWeight": "bold"}),
                                    html.P("But only 25-50%", className="text-center", style={"fontWeight": "bold"}),
                                    html.Hr(),
                                    html.P([
                                        html.Strong("✓ 25-50% is optimal"), html.Br(),
                                        "• +2.1-3.3% cost", html.Br(),
                                        "• 46% cheaper per % at 25-50%", html.Br(),
                                        "• Below 10% tipping point", html.Br(),
                                        "• Highest robustness (CV=0.06)"
                                    ], style={"fontSize": "14px"})
                                ], className="text-center p-3", style={"backgroundColor": "#e8f8e8", "borderRadius": "10px"})
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.H3("No", className="text-center", style={"color": "#d62728", "fontWeight": "bold"}),
                                    html.P("Avoid >50% or 99%", className="text-center", style={"fontWeight": "bold"}),
                                    html.Hr(),
                                    html.P([
                                        html.Strong("✗ 99% is prohibitive"), html.Br(),
                                        "• Costs accelerate >10%", html.Br(),
                                        "• Without LDES: 21.6x jump", html.Br(),
                                        "• Requires massive overbuild", html.Br(),
                                        "• No cost benefit in most countries"
                                    ], style={"fontSize": "14px"})
                                ], className="text-center p-3", style={"backgroundColor": "#ffe6e6", "borderRadius": "10px"})
                            ], width=6),
                        ], className="mb-3"),

                        html.Div([
                            html.H6("💡 Key Recommendation:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "15px"}),
                            html.P([
                                html.Strong("Target 40-50% hourly matching as the 'sweet spot'"), ". Statistical analysis reveals ",
                                html.Strong("increasing returns: 46% cheaper per percentage point at 25-50% vs 0-25%"), " due to infrastructure amortization. ",
                                "Avoid 99% matching - it crosses the 10% cost acceleration barrier and provides no additional climate benefit over 50%."
                            ], style={"backgroundColor": "#fff8e6", "padding": "15px", "borderRadius": "5px", "marginTop": "10px", "borderLeft": "4px solid #ff7f0e"})
                        ]),

                        # Q3: Are investments in LDES and clean advanced technologies a business?
                        html.H5("3. Are Investments in LDES and Advanced Clean Technologies a Business?", style={"fontWeight": "bold", "marginTop": "30px", "color": "#1f77b4"}),

                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Technology"),
                                    html.Th("Investment Case"),
                                    html.Th("Statistical Significance"),
                                    html.Th("Business Opportunity"),
                                    html.Th("Recommendation")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Strong("LDES (H2, Iron-Air)")),
                                    html.Td(html.Strong("+4.24% cost without it", style={"color": "#d62728"})),
                                    html.Td(html.Strong("p<0.001, r=0.312", style={"color": "#d62728"})),
                                    html.Td(html.Span("Excellent", className="badge bg-success", style={"fontSize": "14px"})),
                                    html.Td("✓ Invest heavily")
                                ], style={"backgroundColor": "#e8f8e8"}),
                                html.Tr([
                                    html.Td(html.Strong("Clean Firm (Nuclear, Geothermal)")),
                                    html.Td("+0.37% cost without it"),
                                    html.Td(html.Strong("p=0.316 (NOT significant)", style={"color": "#2ca02c"})),
                                    html.Td(html.Span("Uncertain", className="badge bg-warning", style={"fontSize": "14px"})),
                                    html.Td("⚠ Wait for cost reductions")
                                ]),
                            ])
                        ], bordered=True, hover=True, className="mt-3"),

                        html.Div([
                            html.H6("💡 Key Recommendation:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "15px"}),
                            html.P([
                                html.Strong("LDES: Strong investment case"), " - p<0.001 significance, 21.6x cost acceleration without it, non-negotiable for >10% matching. ",
                                html.Strong("Google should invest in LDES partnerships (iron-air, hydrogen) now."), html.Br(), html.Br(),
                                html.Strong("Clean Firm: Weak investment case"), " - NOT statistically significant (p=0.316), only 0.37% impact. ",
                                html.Strong("Wait for SMR/geothermal costs to fall"), " before major commitments. Focus PPA strategy on renewables + LDES instead."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px"})
                        ]),

                        # Q4: Which policies are necessary for full decarbonization at minimum cost?
                        html.H5("4. Which Policies Are Necessary for Full Decarbonization at Minimum Cost?", style={"fontWeight": "bold", "marginTop": "30px", "color": "#1f77b4"}),

                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Span("1️⃣ ", style={"fontSize": "20px"}),
                                html.Strong("LDES Deployment Mandates/Incentives", style={"fontSize": "16px"}),
                                html.Br(),
                                html.Span("Critical foundation - system cannot function without it at scale. Investment tax credits, capacity markets, or direct procurement.", style={"color": "#666", "fontSize": "14px"})
                            ], color="light", className="mb-2"),
                            dbc.ListGroupItem([
                                html.Span("2️⃣ ", style={"fontSize": "20px"}),
                                html.Strong("Moderate Hourly Matching Requirements (25-50%)", style={"fontSize": "16px"}),
                                html.Br(),
                                html.Span("Drives clean energy investment without crossing cost tipping points. Increasing returns make 40-50% most efficient.", style={"color": "#666", "fontSize": "14px"})
                            ], color="light", className="mb-2"),
                            dbc.ListGroupItem([
                                html.Span("3️⃣ ", style={"fontSize": "20px"}),
                                html.Strong("EU-Wide Grid Coordination", style={"fontSize": "16px"}),
                                html.Br(),
                                html.Span("Changes system economics fundamentally. Enables geographic diversity, reduces overbuild. Early-2030 focus.", style={"color": "#666", "fontSize": "14px"})
                            ], color="light", className="mb-2"),
                            dbc.ListGroupItem([
                                html.Span("4️⃣ ", style={"fontSize": "20px"}),
                                html.Strong("Technology-Neutral Incentives (Not Technology-Specific)", style={"fontSize": "16px"}),
                                html.Br(),
                                html.Span("Sub-additive technology interactions mean portfolio diversity matters more than any single technology. Allow substitution.", style={"color": "#666", "fontSize": "14px"})
                            ], color="light"),
                        ], className="mb-3"),

                        dbc.Alert([
                            html.H6("⚠️ Avoid These Policies:", style={"fontWeight": "bold"}),
                            html.P([
                                "❌ ", html.Strong("99% hourly matching mandates"), " - crosses 10% tipping point, 21.6x cost acceleration", html.Br(),
                                "❌ ", html.Strong("LDES technology bans"), " - +4.24% cost, system becomes infeasible", html.Br(),
                                "❌ ", html.Strong("One-size-fits-all national policies"), " - 25x variation between countries (Luxembourg +25%, Czechia -7%)", html.Br(),
                                "❌ ", html.Strong("Strict additionality in mature markets post-2030"), " - effect declines after 2030 as market naturally adds capacity"
                            ], className="mb-0")
                        ], color="danger"),

                        # Q5: Which country is of special interest?
                        html.H5("5. Which Country Is of Special Interest for GO Markets?", style={"fontWeight": "bold", "marginTop": "30px", "color": "#1f77b4"}),

                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H2("🇩🇰 Denmark", className="text-center", style={"color": "#d62728", "fontWeight": "bold"}),
                                        html.Hr(),
                                        html.P([
                                            html.Strong("Why Denmark Wins:"), html.Br(), html.Br(),
                                            "✓ ", html.Strong("-7.68% cost reduction"), " with GO matching", html.Br(),
                                            "✓ High wind capacity already deployed", html.Br(),
                                            "✓ Flexible generation mix (gas, CHP)", html.Br(),
                                            "✓ Nordic interconnections enable import/export", html.Br(),
                                            "✓ Small size = low coordination costs", html.Br(),
                                            "✓ Renewable surplus hours = cheap charging", html.Br(), html.Br(),
                                            html.Strong("Strategic Value: "), "Denmark ", html.Em("benefits"), " from hourly matching naturally. Google doesn't pay a premium - the system becomes MORE efficient with matching requirements due to baseload displacement."
                                        ], style={"fontSize": "14px"})
                                    ])
                                ], color="danger", outline=True)
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H2("🇮🇹 Italy", className="text-center", style={"color": "#ff7f0e", "fontWeight": "bold"}),
                                        html.Hr(),
                                        html.P([
                                            html.Strong("Runner-Up: Italy"), html.Br(), html.Br(),
                                            "✓ ", html.Strong("-2.76% cost reduction"), " with 25% matching", html.Br(),
                                            "✓ High solar potential (60%+ capacity factors)", html.Br(),
                                            "✓ Industrial load provides flexibility", html.Br(),
                                            "✓ Large market size (100+ GW)", html.Br(),
                                            "✓ Gas infrastructure enables peaking", html.Br(), html.Br(),
                                            html.Strong("Strategic Value: "), "Second-best country. Large enough market for significant data center capacity. Solar-heavy generation naturally matches daytime computing loads."
                                        ], style={"fontSize": "14px"})
                                    ])
                                ], color="warning", outline=True)
                            ], width=6),
                        ], className="mb-3"),

                        html.Div([
                            html.H6("💡 Final Strategic Recommendation:", style={"color": "#1f77b4", "fontWeight": "bold", "marginTop": "20px"}),
                            html.P([
                                html.Strong("Priority 1: Expand data center presence in Denmark and Italy"), " where GO markets naturally reduce costs.", html.Br(),
                                html.Strong("Priority 2: Invest in LDES partnerships (iron-air batteries, green hydrogen)"), " - strongest statistical signal (p<0.001).", html.Br(),
                                html.Strong("Priority 3: Target 40-50% hourly matching"), " - 46% cheaper per percentage point than 0-25% (increasing returns).", html.Br(),
                                html.Strong("Avoid: Luxembourg, 99% matching, clean-firm technology lock-in before cost reductions"), ".", html.Br(), html.Br(),
                                html.Strong("Expected Outcome: "), "Following these recommendations could reduce Google's clean energy costs by 2-8% while accelerating decarbonization."
                            ], style={"backgroundColor": "#e8f4f8", "padding": "15px", "borderRadius": "5px", "marginTop": "10px", "border": "2px solid #1f77b4"})
                        ]),

                    ])
                ], className="mb-4")
            ])
        ]),

        # Section 9: Policy Recommendations
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
