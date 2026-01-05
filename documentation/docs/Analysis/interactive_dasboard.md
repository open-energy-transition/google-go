# Interactive Dashboard

## Overview

The Google-Go Interactive Dashboard is a web-based visualization tool built with Python Dash for exploring energy system modeling results. It provides an intuitive interface to analyze multiple scenarios, compare results across different policy configurations, and explore both aggregated metrics and detailed hourly timeseries data.

The dashboard is designed to handle large-scale energy system analysis results with millions of data points, providing fast, interactive visualizations with intelligent caching and data management.

---

## Key Features

The dashboard provides five main analysis tabs:

### 1. **Single Scenario Analysis**
- Visualize individual scenario results in detail
- Multiple plot types: bar charts, stacked bars, area charts, pie charts
- Year-over-year evolution tracking
- Carrier-level filtering and analysis

### 2. **Cross-Scenario Comparison**
- Compare up to 4 scenarios side-by-side
- Side-by-side bar charts with scenario grouping
- Year-over-year evolution comparisons
- Summary statistics and difference analysis

### 3. **Dead Zone Analysis**
- Frontier curve visualization across spatial scopes
- Compare cost-effectiveness frontiers
- Multi-year and multi-scenario frontier analysis
- Country-level dead zone identification

### 4. **Timeseries Exploration**
- Explore hourly timeseries data (8,760 hours/year)
- Interactive time range selection (week, month, full year)
- Multi-scenario overlay comparisons
- Carrier-level filtering for detailed analysis

### 5. **Key Insights**
- Statistical analysis findings (12 critical insights)
- Strategic recommendations for energy procurement
- Regional analysis and policy implications
- Interactive summary cards with detailed explanations

---

## Technical Architecture

### Technology Stack

**Core Framework:**
- **Dash (Plotly)**: Web application framework for Python
- **Plotly**: Interactive visualization library
- **Dash Bootstrap Components**: UI component library

**Data Processing:**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Parquet/CSV**: Data storage formats

**Deployment:**
- **Flask**: WSGI web server (built into Dash)
- **Gunicorn**: Production WSGI server (optional)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (User Interface)                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Tab 1    │ │ Tab 2    │ │ Tab 3    │ │ Tab 4    │ ...  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────┴────────────────────────────────────┐
│                   Dash Application (app.py)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Callback Management Layer                │  │
│  │  • User input handlers                                │  │
│  │  • Plot generation                                    │  │
│  │  • Data filtering                                     │  │
│  └────────┬─────────────────────────────────────┬────────┘  │
└───────────┼─────────────────────────────────────┼───────────┘
            │                                     │
┌───────────┴───────────┐           ┌────────────┴────────────┐
│   Layout Components   │           │   Utility Modules       │
│   (layouts/)          │           │   (utils/)              │
│                       │           │                         │
│ • single_scenario     │           │ • DataLoader            │
│ • cross_scenario      │           │   - CSV/Parquet I/O    │
│ • deadzone            │           │   - Caching            │
│ • timeseries          │           │   - Data filtering     │
│ • insights            │           │                         │
└───────────────────────┘           │ • ColorMapper           │
                                    │   - Carrier colors     │
                                    │   - Consistent themes  │
                                    └────────┬────────────────┘
                                             │
┌────────────────────────────────────────────┴────────────────┐
│                    Data Layer (results/)                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results.csv (Consolidated)                           │  │
│  │  • Multi-level headers (year, scenario, scope)        │  │
│  │  • Multi-level index (metric, y-label, carrier)       │  │
│  │  • ~145 rows × ~400 columns                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results_frontier.csv                                 │  │
│  │  • Frontier analysis data                             │  │
│  │  • Multiple scenarios, years, countries               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results_time_series.parquet                          │  │
│  │  • Hourly data (8,760 hours/year)                     │  │
│  │  • ~millions of data points                           │  │
│  │  • Chunked loading with caching                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  colors.csv                                           │  │
│  │  • Carrier color mappings                             │  │
│  │  • Ensures visual consistency                         │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Startup**: `DataLoader` loads all consolidated results into memory
2. **User Selection**: User selects year, scenario, metric via dropdowns
3. **Callback Trigger**: Dash detects input changes and fires registered callbacks
4. **Data Filtering**: `DataLoader` filters data based on user selections
5. **Plot Generation**: Callback creates Plotly figure with filtered data
6. **Color Mapping**: `ColorMapper` applies consistent colors to carriers
7. **Rendering**: Plotly renders interactive visualization in browser

### File Structure

```
dashboard/
├── app.py                          # Main application entry point
├── callbacks.py                    # All callback functions (17 callbacks)
├── layouts/                        # Tab-specific layouts
│   ├── __init__.py
│   ├── single_scenario_layout.py   # Single scenario analysis
│   ├── cross_scenario_layout.py    # Multi-scenario comparison
│   ├── deadzone_layout.py          # Frontier analysis
│   ├── timeseries_layout.py        # Hourly timeseries
│   └── insights_layout.py          # Statistical insights (static)
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and caching
│   └── colors.py                   # Color mapping utilities
└── assets/                         # Static assets (CSS, images)
    └── custom.css                  # Custom styling
```

---

## Data Structure

### Consolidated Results Format

The dashboard uses a consolidated data structure where all scenarios are stored in a single file, eliminating redundancy and simplifying data management.

#### results.csv

**Multi-level column headers:**
```
Level 0: year          | 2025              | 2025              | 2030 ...
Level 1: scenario      | baseline          | energy-match-25   | baseline ...
Level 2: scope         | system            | system            | system ...
```

**Multi-level row index:**
```
Level 0: Results       | (a) Energy mix    | (a) Energy mix    | (b) Capacity ...
Level 1: y_label       | Net generation... | Net generation... | Capacity ...
Level 2: carrier       | solar             | onwind            | solar ...
```

**Example data snippet:**
```csv
year,,,2025,2025,2025,2030,2030,2030
scenario,,,baseline,energy-match-25,hourly-match-25-90,baseline,energy-match-25,hourly-match-25-90
scope,,,system,system,system,system,system,system
Results,y_label,carrier,,,,,,,
(a) Energy mix,Net generation (TWh),solar,245.3,267.8,289.1,312.5,345.7,378.2
(a) Energy mix,Net generation (TWh),onwind,456.2,478.9,501.3,534.8,567.1,599.4
(a) Energy mix,Net generation (TWh),CCGT,151.1,119.3,48.4,104.5,31.7,14.9
```

**Key characteristics:**
- **Dimensions**: ~145 rows × ~400 columns
- **Scenarios**: All policy scenarios in one file
- **Years**: 2025, 2030, 2035, 2040
- **Metrics**: 13+ categories (energy mix, capacity, costs, emissions, etc.)
- **Carriers**: 15-20 energy carriers (solar, wind, gas, hydrogen, etc.)

#### results_frontier.csv

Stores frontier analysis data for dead zone visualization:

**Structure:**
```
Row 0: Years (metadata)
Row 1: Countries (metadata)
Row 2+: Frontier points (matching percentages and costs)
Columns: scenario | year | country triplets
```

**Use case:** Identifying cost-effectiveness frontiers and "dead zones" where policies become prohibitively expensive.

#### results_time_series.parquet

**Format**: Parquet (compressed columnar format) or CSV fallback

**Structure:**
- **Multi-index**: scenario, year, Results, y_label, country, type, carrier
- **Columns**: 8,760 hourly timestamps (full year)
- **Size**: ~50-100MB compressed, ~500MB-1GB uncompressed

**Performance optimizations:**
- Parquet format provides 10-20x faster loading than CSV
- Chunked reading for memory efficiency
- Intelligent caching (50 most recent queries)
- Multi-index for fast filtering

**Example query:**
```python
# Load electricity balance for baseline scenario, 2035, EU, summer week
data = data_loader.load_timeseries_data(
    year=2035,
    scenarios=['baseline'],
    ts_type='Electricity Balance',
    country='EU',
    carriers=['solar', 'onwind', 'battery_discharge'],
    time_range='week_summer'
)
```

#### colors.csv

Maps energy carriers to consistent colors across all visualizations:

```csv
Results,carrier,color
(a) Energy mix,solar,#f9d002
(a) Energy mix,onwind,#235ebc
(a) Energy mix,offwind-ac,#6895dd
(a) Energy mix,offwind-dc,#74c6f2
(a) Energy mix,CCGT,#a85522
(a) Energy mix,hydrogen,#ea048a
(a) Energy mix,battery_discharge,#b474de
```

---

## Available Scenarios

The consolidated dataset includes all policy scenarios:

### Baseline
- **baseline**: Base case without policy constraints

### Energy Matching
- **energy-match-25**: 25% annual energy matching
- **energy-match-50**: 50% annual energy matching

### Hourly Matching (25% additionality)
- **hourly-match-25-90**: 25% additionality, 90% hourly matching
- **hourly-match-25-95**: 25% additionality, 95% hourly matching
- **hourly-match-25-98**: 25% additionality, 98% hourly matching
- **hourly-match-25-99**: 25% additionality, 99% hourly matching

### Hourly Matching (50% additionality)
- **hourly-match-50-90**: 50% additionality, 90% hourly matching
- **hourly-match-50-95**: 50% additionality, 95% hourly matching
- **hourly-match-50-98**: 50% additionality, 98% hourly matching
- **hourly-match-50-99**: 50% additionality, 99% hourly matching

### No Additionality
- **hourly-match-noadd-10-99**: No additionality requirement, 10% target, 99% hourly
- **hourly-match-noadd-50-99**: No additionality requirement, 50% target, 99% hourly
- **hourly-match-noadd-90-99**: No additionality requirement, 90% target, 99% hourly

### Policy Variants
- **baseline-co2-price25**: Baseline with €25/ton CO2 price
- **no-LDES**: Scenarios without long-duration energy storage
- **no-clean-firm**: Scenarios without clean firm capacity
- **EU-25**: EU-wide coordination at 25% matching
- **EU-50**: EU-wide coordination at 50% matching

---

## Installation and Setup

### Requirements

**Python version**: 3.8+

**Core dependencies:**
```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0  # For parquet support
```

### Installation Steps

1. **Clone or navigate to the repository:**
```bash
cd /path/to/google-go
```

2. **Install dependencies:**
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyarrow
```

3. **Verify data structure:**
```bash
ls -lh results/
# Should show:
# results.csv
# results_frontier.csv
# results_time_series.parquet (or .csv)
# colors.csv
```

4. **Launch the dashboard:**
```bash
cd dashboard
python app.py
```

5. **Access in browser:**
```
http://localhost:8050
```

### Production Deployment

For production environments, use Gunicorn:

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050 --workers 4 --timeout 300
```

**Configuration options:**
- `--workers 4`: Use 4 worker processes
- `--timeout 300`: 5-minute timeout for large data queries
- `-b 0.0.0.0:8050`: Bind to all network interfaces

---

## User Guide

### Tab 1: Single Scenario Analysis

**Purpose**: Deep dive into a single scenario's results

**Controls:**
- **Year**: Select 2025, 2030, 2035, 2040, or "All"
- **Scenario**: Choose from available policy scenarios
- **Metric**: Select analysis category (energy mix, capacity, costs, etc.)
- **Plot Type**: Choose visualization style

**Plot Types:**
1. **Bar Chart**: Vertical bars showing carrier breakdown
2. **Stacked Bar (All Years)**: Multi-year comparison in one chart
3. **Stacked Area**: Cumulative area chart over years
4. **Pie Chart**: Proportional breakdown
5. **Year Comparison**: Side-by-side bars for all years
6. **Year on Year Evolution**: Line chart showing trends

**Example Workflow:**

*Question: How does the energy mix change from 2025 to 2040 under hourly-match-50-90?*

1. Navigate to **Single Scenario Analysis** tab
2. Select **Scenario**: hourly-match-50-90
3. Select **Metric**: (a) Energy mix
4. Select **Plot Type**: Year on Year Evolution
5. View the line chart showing how each carrier evolves

**Tips:**
- Use "Year on Year Evolution" to identify trends
- Use "Stacked Bar" to see proportional changes
- Click legend items to show/hide specific carriers
- Hover over bars for exact values

### Tab 2: Cross-Scenario Comparison

**Purpose**: Compare up to 4 scenarios side-by-side

**Controls:**
- **Year**: Select specific year or "All"
- **Metric**: Choose analysis category
- **Plot Type**: Choose comparison style
- **Group By**: Group by year or scenario
- **Scenario 1-4**: Select up to 4 scenarios

**Plot Types:**
1. **Side-by-Side**: Grouped bar chart
2. **Stacked Bar (All Years)**: Multi-year stacked comparison
3. **Stacked Bar + Total Line**: Stacked bars with total overlay
4. **Year Comparison**: Compare specific years
5. **Year on Year Evolution**: Evolution lines for multiple scenarios

**Example Workflow:**

*Question: How much more solar capacity is needed at 95% vs 90% hourly matching?*

1. Navigate to **Cross-Scenario Comparison** tab
2. Select **Year**: 2040
3. Select **Metric**: (c) Capacity mix
4. Select **Scenario 1**: hourly-match-50-90
5. Select **Scenario 2**: hourly-match-50-95
6. Select **Plot Type**: Side-by-Side
7. Compare the solar bars between scenarios

**Tips:**
- Compare baseline against policy scenarios to measure impact
- Use "Group By: Year" to see temporal evolution
- Use "Group By: Scenario" to see scenario differences at each year
- Summary statistics show total changes and percentages

### Tab 3: Dead Zone Analysis

**Purpose**: Visualize cost-effectiveness frontiers and identify dead zones

**Controls:**
- **Year**: Select specific year or "All"
- **Scenarios**: Select up to 5 scenarios (or "All")
- **Countries**: Select up to 5 countries (or "All")

**What is a Dead Zone?**
A "dead zone" is a region on the frontier where small increases in clean energy matching requirements lead to disproportionately large cost increases, indicating diminishing returns.

**Frontier Curve Interpretation:**
- **X-axis**: Clean energy matching percentage (0-100%)
- **Y-axis**: Total system cost increase (%)
- **Steep sections**: Dead zones (avoid these targets)
- **Flat sections**: Cost-effective operating ranges
- **Tipping points**: Where curve suddenly steepens

**Example Workflow:**

*Question: At what matching percentage does the no-LDES scenario become prohibitively expensive?*

1. Navigate to **Dead Zone Analysis** tab
2. Select **Year**: 2035
3. Select **Scenarios**: no-LDES, baseline (for comparison)
4. Select **Countries**: EU
5. Observe the frontier curves - look for steep increases
6. **Finding**: no-LDES shows 21.6x cost acceleration at 10% matching

**Tips:**
- Compare multiple scenarios to identify which policies create dead zones
- Use "All" years to see how dead zones shift over time
- Country-level analysis reveals regional sensitivities
- Look for "tipping points" where curves suddenly steepen

### Tab 4: Timeseries Exploration

**Purpose**: Explore hourly operational data (8,760 hours/year)

**Controls:**
- **Year**: Select year to analyze
- **Scenarios**: Multi-select scenarios to overlay
- **Timeseries Type**: Choose data type (Electricity Balance, Storage Levels, etc.)
- **Country**: Select geographic scope
- **Carrier**: Optional filter for specific carriers
- **Time Range**: Select temporal zoom level

**Time Range Options:**
- **Full Year**: All 8,760 hours
- **Week 1**: First week of January
- **Winter Week**: Representative winter week (Jan 15-21)
- **Summer Week**: Representative summer week (July 1-7)
- **Specific Month**: 2013-01, 2013-02, etc.

**Example Workflow:**

*Question: How does battery storage cycle during a typical winter week under hourly-match-50-90?*

1. Navigate to **Timeseries Exploration** tab
2. Select **Year**: 2035
3. Select **Scenarios**: hourly-match-50-90
4. Select **Timeseries Type**: Storage Levels
5. Select **Country**: EU
6. Select **Carrier**: battery
7. Select **Time Range**: Winter Week
8. Observe daily charging/discharging cycles

**Tips:**
- Start with weekly views to identify patterns
- Expand to full year to see seasonal trends
- Compare multiple scenarios to see policy impacts on operations
- Look for correlation between demand, solar/wind, and storage
- Winter weeks show system stress, summer weeks show surplus

### Tab 5: Key Insights

**Purpose**: View comprehensive statistical analysis and strategic recommendations

**Content:**
- **Executive Summary**: 12 critical findings from 3,080 scenario runs
- **Tipping Points**: Universal 10% barrier analysis
- **LDES Criticality**: Statistical significance of long-duration storage
- **Frontier Curve Analysis**: Cost elasticity rankings
- **Robustness Paradox**: Stricter policies → more predictable outcomes
- **Low-Dimensional Structure**: 3 factors explain 98.72% of variation
- **Temporal Patterns**: Seasonal and hourly dynamics
- **Deep-Dive Findings**: Regional extremes, non-linear costs, scenario divergence
- **Strategic Recommendations**: Data-driven procurement strategies
- **Policy Recommendations**: Cost-effectiveness rankings

**Key Findings Highlight:**

1. **10% Tipping Point**: All scenarios show dramatic cost acceleration at 10% hourly matching threshold
2. **LDES Non-Negotiable**: +4.24% cost without it, p<0.001 significance
3. **EU Frontier Anomaly**: EU scenarios accelerate at 2% vs 97-117% for national scenarios
4. **Increasing Returns**: 46% cheaper per percentage point at 25-50% matching vs 0-25%
5. **Regional Extremes**: 25x variation - Luxembourg +25%, Czechia -7.3%

**Tips:**
- Read this tab first to understand key findings
- Reference specific insights when exploring other tabs
- Use findings to guide scenario comparisons
- Share strategic recommendations with stakeholders

---

## Customization Guide

### Changing Analyzed Scenarios

The dashboard automatically detects all scenarios present in `results.csv`. To add or remove scenarios:

1. **Modify results.csv**:
   - Add new columns with format: `year | scenario | scope`
   - Ensure new scenarios follow existing data structure

2. **Restart dashboard**:
   ```bash
   python app.py
   ```

3. **Verify**: New scenarios appear in dropdown menus

**No code changes required** - the dashboard dynamically loads all available scenarios.

### Adding Custom Metrics

To add new metrics to analysis:

1. **Add data to results.csv**:
   ```csv
   (n) New Metric,Y-axis Label,carrier,value1,value2,value3,...
   ```

2. **Add color mappings** (optional) in `colors.csv`:
   ```csv
   (n) New Metric,carrier1,#color1
   (n) New Metric,carrier2,#color2
   ```

3. **Restart dashboard** - new metric appears in dropdown

### Customizing Colors

Edit `results/colors.csv`:

```csv
Results,carrier,color
(a) Energy mix,solar,#FFD700  # Change to gold
(a) Energy mix,onwind,#00BFFF  # Change to deep sky blue
```

**Color format**: Hex codes (#RRGGBB)

**Apply changes**: Restart dashboard

### Modifying Plot Types

To add new visualizations, edit `dashboard/callbacks.py`:

```python
def create_plot(data, metric, plot_type, color_mapper):
    if plot_type == 'my_custom_plot':
        fig = go.Figure()
        # Add your custom plot logic
        for carrier in data.index:
            fig.add_trace(go.Scatter(
                x=years,
                y=data.loc[carrier],
                name=carrier,
                marker=dict(color=color_mapper.get_color(metric, carrier))
            ))
        return fig
```

Then add to dropdown in `layouts/single_scenario_layout.py`:

```python
options=[
    {'label': 'Bar Chart', 'value': 'bar'},
    {'label': 'My Custom Plot', 'value': 'my_custom_plot'},
    # ...
]
```

### Adjusting Data Caching

Edit `dashboard/utils/data_loader.py`:

```python
# Change cache size (default: 50 entries)
if len(self.timeseries_cache) > 100:  # Increase to 100
    self.timeseries_cache.pop(next(iter(self.timeseries_cache)))
```

**Trade-off**: Larger cache → more memory usage, faster repeated queries

---

## Performance Optimization

### Memory Management

**Typical memory usage:**
- Dashboard base: ~100-200 MB
- Consolidated results: ~50-100 MB
- Frontier data: ~10-20 MB
- Timeseries cache: ~500 MB (50 queries)
- **Total**: ~1-2 GB for typical usage

**For large datasets:**
1. Use Parquet format (10-20x compression vs CSV)
2. Limit timeseries cache size
3. Use time range filtering instead of loading full year
4. Deploy with adequate RAM (4GB+ recommended)

### Loading Speed

**Startup time:**
- Consolidated results.csv: ~1-2 seconds
- Frontier data: ~0.5 seconds
- Timeseries metadata: ~5-10 seconds (parquet), ~30-60 seconds (CSV)

**Query response time:**
- Aggregated plots: ~0.1-0.5 seconds
- Timeseries plots (cached): ~0.2-0.5 seconds
- Timeseries plots (uncached): ~2-10 seconds (parquet), ~10-30 seconds (CSV)

**Optimization recommendations:**
1. **Convert timeseries to Parquet**:
   ```python
   import pandas as pd
   df = pd.read_csv('results_time_series.csv')
   df.to_parquet('results_time_series.parquet', compression='snappy')
   ```

2. **Increase cache size** for repeated queries
3. **Use shorter time ranges** for exploratory analysis
4. **Deploy with SSD** for faster I/O

---

## Troubleshooting

### Common Issues

#### 1. Dashboard won't start

**Error**: `ModuleNotFoundError: No module named 'dash'`

**Solution**:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyarrow
```

#### 2. No data showing

**Error**: Blank plots or "No data available"

**Possible causes**:
- `results.csv` not found or incorrectly formatted
- Column headers don't match expected structure
- Missing year/scenario/metric in data

**Solution**:
1. Verify file exists: `ls -lh results/results.csv`
2. Check first few lines: `head -5 results/results.csv`
3. Verify multi-level headers are correct
4. Check dashboard logs for specific errors

#### 3. Timeseries loading very slow

**Issue**: Timeseries queries take 30+ seconds

**Solution**:
1. Convert CSV to Parquet:
   ```python
   import pandas as pd
   df = pd.read_csv('results/results_time_series.csv')
   df.to_parquet('results/results_time_series.parquet', compression='snappy')
   ```
2. Verify parquet file is detected:
   ```bash
   ls -lh results/results_time_series.parquet
   ```
3. Restart dashboard

#### 4. Colors not matching

**Issue**: Carriers show default colors instead of custom colors

**Possible causes**:
- `colors.csv` not found
- Carrier names don't match between `results.csv` and `colors.csv`
- Incorrect CSV format

**Solution**:
1. Verify file: `cat results/colors.csv | head -10`
2. Check carrier name spelling (case-sensitive)
3. Ensure format: `Results,carrier,color`

#### 5. Port already in use

**Error**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Option 1: Kill existing process
lsof -ti:8050 | xargs kill -9

# Option 2: Use different port
# Edit app.py, change:
app.run_server(debug=True, host='0.0.0.0', port=8051)
```

#### 6. Memory errors

**Error**: `MemoryError` or dashboard crashes

**Cause**: Insufficient RAM for large datasets

**Solutions**:
1. Reduce timeseries cache size (edit `data_loader.py`)
2. Use time range filtering instead of full year
3. Deploy on machine with more RAM (4GB+ recommended)
4. Use Parquet format for better compression

---

## Advanced Topics

### Callback Architecture

The dashboard uses Dash's callback system for interactivity:

```python
@app.callback(
    Output('plot-id', 'figure'),      # What to update
    [Input('dropdown-id', 'value')]   # What triggers the update
)
def update_plot(selected_value):
    # Filter data based on input
    data = data_loader.get_data(selected_value)
    # Generate plot
    fig = create_plot(data)
    return fig
```

**17 callbacks** handle all dashboard interactivity:
- 5 for Single Scenario Analysis
- 1 for Cross-Scenario Comparison
- 3 for Dead Zone Analysis
- 5 for Timeseries Exploration
- 3 for dynamic dropdown population

### Multi-Index Data Handling

Pandas MultiIndex is used extensively for efficient data organization:

```python
# Column MultiIndex
columns = pd.MultiIndex.from_tuples([
    (2025, 'baseline', 'system'),
    (2025, 'energy-match-25', 'system'),
    # ...
], names=['year', 'scenario', 'scope'])

# Row MultiIndex
index = pd.MultiIndex.from_tuples([
    ('(a) Energy mix', 'Net generation (TWh)', 'solar'),
    ('(a) Energy mix', 'Net generation (TWh)', 'onwind'),
    # ...
], names=['Results', 'y_label', 'carrier'])

# Fast filtering with IndexSlice
idx = pd.IndexSlice
data = df.loc[idx['(a) Energy mix', :, :], idx[2025, 'baseline', :]]
```

### Adding New Tabs

To add a new analysis tab:

1. **Create layout** in `layouts/my_new_tab.py`:
```python
def create_my_tab_layout(data_loader):
    return dbc.Container([
        html.H3("My New Analysis"),
        dcc.Graph(id='my-plot'),
        # Add controls...
    ])
```

2. **Register in app.py**:
```python
from layouts import my_new_tab

# Add tab
dcc.Tab(label='My Analysis', value='my-tab')

# Add layout
html.Div(my_new_tab.create_my_tab_layout(data_loader),
         id='my-content', style={'display': 'none'})
```

3. **Add callback** in `callbacks.py`:
```python
@app.callback(
    Output('my-plot', 'figure'),
    [Input('my-selector', 'value')]
)
def update_my_plot(value):
    # Your plot logic
    return fig
```

4. **Update tab visibility** callback in `app.py`

---

## Best Practices

### Data Exploration Workflow

1. **Start with Key Insights tab** to understand high-level findings
2. **Use Single Scenario Analysis** to explore individual scenarios
3. **Use Cross-Scenario Comparison** to measure policy impacts
4. **Use Dead Zone Analysis** to identify cost tipping points
5. **Use Timeseries Exploration** to understand operational details

### Performance Tips

1. **Use Parquet format** for timeseries data (10-20x faster)
2. **Start with smaller time ranges** (week/month) before loading full year
3. **Use caching** - repeated queries are instant
4. **Limit scenarios** in multi-scenario comparisons (max 4)
5. **Close unused browser tabs** to free memory

### Presentation Tips

1. **Use consistent plot types** across related analyses
2. **Include baseline** in comparisons for reference
3. **Use "Year on Year Evolution"** to show trends clearly
4. **Export plots** via browser's screenshot or save feature
5. **Cite specific insights** from Key Insights tab

---

## API Reference

### DataLoader Class

**Location**: `dashboard/utils/data_loader.py`

**Methods:**

```python
# Load all data at startup
data_loader.load_all_data()

# Get summary statistics
stats = data_loader.get_summary_stats()
# Returns: {'years': [...], 'scenarios': [...], 'metrics': [...]}

# Get filtered data
data = data_loader.get_data(
    year=2035,              # int or None
    scenario_name='baseline',  # str or None
    metric='(a) Energy mix'     # str or None
)

# Get carriers for a metric
carriers = data_loader.get_carriers_for_metric('(a) Energy mix')

# Get frontier data
frontier = data_loader.get_frontier_data(
    year=2035,
    country='EU'
)

# Get frontier countries
countries = data_loader.get_frontier_countries(year=2035)

# Get timeseries metadata
metadata = data_loader.get_timeseries_metadata()

# Load timeseries data
data, timestamps = data_loader.load_timeseries_data(
    year=2035,
    scenarios=['baseline', 'energy-match-25'],
    ts_type='Electricity Balance',
    country='EU',
    carriers=['solar', 'onwind'],
    time_range='week_winter'
)
```

### ColorMapper Class

**Location**: `dashboard/utils/colors.py`

**Methods:**

```python
# Initialize with colors.csv
color_mapper = ColorMapper('../results/colors.csv')

# Get color for carrier in metric
color = color_mapper.get_color('(a) Energy mix', 'solar')

# Get all colors for a metric
colors = color_mapper.get_colors_for_metric('(a) Energy mix')

# Format scenario names for display
display_name = format_scenario_name('hourly-match-50-90')
# Returns: "Hourly 90% (CI 50%)"
```

---

## Contributing

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to all functions
- Comment complex logic

### Testing Changes

1. Test with full dataset
2. Verify all tabs load correctly
3. Check all dropdown combinations
4. Test edge cases (empty data, single year, etc.)
5. Verify timeseries loading with both Parquet and CSV

### Submitting Updates

1. Document all changes
2. Update this guide if adding features
3. Test on clean Python environment
4. Ensure backward compatibility

---

## Further Resources

- **Dash Documentation**: https://dash.plotly.com/
- **Plotly Python**: https://plotly.com/python/
- **Pandas MultiIndex**: https://pandas.pydata.org/docs/user_guide/advanced.html
- **Parquet Format**: https://parquet.apache.org/

---

## Version History

- **v1.0** (Dec 2024): Initial release with separate CI_25/CI_50/CI_noadd files
- **v2.0** (Jan 2025): Consolidated data structure, added Key Insights tab, improved performance
- **Current**: v2.0

---

## Support

For questions, issues, or feature requests:
1. Check this guide's Troubleshooting section
2. Review dashboard logs for error messages
3. Verify data structure matches expected format
4. Contact the development team with specific questions

**Dashboard Status**: Production-ready, actively maintained
