# Interactive Dashboard

## Overview

The Google-Go Interactive Dashboard is a web-based visualization tool for exploring the project results. It provides an intuitive interface to analyze multiple scenarios, compare results across different policy configurations, and explore both aggregated metrics and detailed hourly timeseries data.

The dashboard handles large-scale energy system analysis results with millions of data points, providing fast, interactive visualizations with intelligent caching and data management.

---

## Installation and Setup

### Requirements

**Python version**: 3.8+

**Core dependencies:**
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyarrow
```

### Installation Steps

1. **Navigate to the repository:**

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
    ```
    
    Should show:
    
    - `results.csv`
    - `results_frontier.csv`
    - `results_time_series.parquet` (or `.csv`)
    - `colors.csv`

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

## Data Structure

The dashboard uses a consolidated data structure where all scenarios are stored in a single file.

### Required Files

```
results/
├── colors.csv                    # Color mappings for carriers
├── results.csv                   # Consolidated results for all scenarios
├── results-*.csv                 # Country-specific results (optional)
├── results_frontier.csv          # Frontier analysis data
├── results_time_series.parquet   # Timeseries data (parquet format, preferred)
└── results_time_series.csv       # Timeseries data (CSV fallback)
```

### results.csv Format

The consolidated `results.csv` file has a **multi-level header structure**:

**Column headers:**
```
Level 0: year          | 2025              | 2025              | 2030 ...
Level 1: scenario      | baseline          | energy-match-25   | baseline ...
Level 2: scope         | system            | system            | system ...
```

**Row index:**
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
- **Years**: 2025, 2030, 2035, 2040
- **Metrics**: 13+ categories (energy mix, capacity, costs, emissions, etc.)
- **Carriers**: 15-20 energy carriers (solar, wind, gas, hydrogen, etc.)

### colors.csv Format

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

## User Guide

The dashboard provides five main analysis tabs for exploring energy system results.

### Tab 1: Single Scenario Analysis

**Purpose**: Deep dive into a single scenario's results to understand detailed energy system characteristics.

**What You Can Do:**

- Visualize individual scenario results in detail
- View multiple plot types: bar charts, stacked bars, area charts, pie charts
- Track year-over-year evolution
- Filter by specific energy carriers

**Controls:**

- **Year**: Select 2025, 2030, 2035, 2040, or "All"
- **Scenario**: Choose from available policy scenarios
- **Metric**: Select analysis category (energy mix, capacity, costs, etc.)
- **Plot Type**: Choose visualization style

**Available Plot Types:**

**1. Bar Chart**: Vertical bars showing carrier breakdown for a specific year

- Best for: Quick overview of a single year
- Shows: Individual carrier contributions

**2. Stacked Bar (All Years)**: Multi-year comparison in one chart

- Best for: Seeing how the mix evolves over time
- Shows: All years side-by-side with stacked carriers

**3. Stacked Area**: Cumulative area chart over years

- Best for: Visualizing cumulative contributions
- Shows: Smooth evolution of carrier mix

**4. Pie Chart**: Proportional breakdown

- Best for: Understanding relative proportions
- Shows: Percentage contribution of each carrier

**5. Year Comparison**: Side-by-side bars for all years

- Best for: Comparing specific carriers across years
- Shows: How each carrier changes year-to-year

**6. Year on Year Evolution**: Line chart showing trends

- Best for: Identifying growth/decline trends
- Shows: Line trajectories for each carrier

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

---

### Tab 2: Cross-Scenario Comparison

**Purpose**: Compare up to 4 scenarios side-by-side to understand how different policies affect the energy system.

**What You Can Do:**

- Compare multiple scenarios simultaneously
- View side-by-side bar charts with scenario grouping
- Analyze year-over-year evolution across scenarios
- Review summary statistics and differences

**Controls:**

- **Year**: Select specific year or "All"
- **Metric**: Choose analysis category
- **Plot Type**: Choose comparison style
- **Group By**: Group by year or scenario
- **Scenario 1-4**: Select up to 4 scenarios to compare

**Available Plot Types:**

**1. Side-by-Side**: Grouped bar chart comparing scenarios

- Best for: Direct scenario comparisons
- Shows: Scenarios grouped for each carrier

**2. Stacked Bar (All Years)**: Multi-year stacked comparison

- Best for: Seeing evolution across scenarios and years
- Shows: Stacked carriers for each scenario/year combination

**3. Stacked Bar + Total Line**: Stacked bars with total overlay

- Best for: Comparing totals while seeing composition
- Shows: Stacked bars with line showing total

**4. Year Comparison**: Compare specific years across scenarios

- Best for: Year-specific analysis
- Shows: How scenarios differ in specific years

**5. Year on Year Evolution**: Evolution lines for multiple scenarios

- Best for: Trend comparison
- Shows: Multiple scenario trajectories on one plot

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

---

### Tab 3: Dead Zone Analysis

**Purpose**: Visualize cost-effectiveness frontiers and identify "dead zones" where policies become prohibitively expensive.

**What You Can Do:**

- View frontier curves showing cost vs. clean energy matching
- Identify tipping points where costs accelerate dramatically
- Compare frontiers across multiple scenarios and countries
- Analyze regional sensitivities to policy constraints

**Controls:**

- **Year**: Select specific year or "All"
- **Scenarios**: Select up to 5 scenarios (or "All")
- **Countries**: Select up to 5 countries (or "All")

**What is a Dead Zone?**

A "dead zone" is a region on the frontier where small increases in clean energy matching requirements lead to disproportionately large cost increases, indicating diminishing returns.

**How to Read Frontier Curves:**

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

---

### Tab 4: Timeseries Exploration

**Purpose**: Explore hourly operational data (8,760 hours/year) to understand how the energy system operates in detail.

**What You Can Do:**

- Explore hourly timeseries data for the full year
- Select specific time ranges (week, month, season)
- Overlay multiple scenarios for comparison
- Filter by specific energy carriers

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

---

### Tab 5: Key Insights

**Purpose**: View comprehensive statistical analysis and strategic recommendations derived from 3,080 scenario runs.

**What You'll Find:**

**Executive Summary**: 12 critical findings including:

- Universal 10% tipping point analysis
- LDES (Long-Duration Energy Storage) criticality
- EU frontier anomaly
- Regional extremes (25x variation)
- Non-linear cost patterns

**Key Sections:**

**1. Critical Tipping Points**

- The universal 10% barrier
- Cost acceleration patterns
- Technology-specific thresholds

**2. LDES Criticality**

- Statistical significance (p<0.001)
- Cost impact (+4.24% without LDES)
- System feasibility analysis

**3. Frontier Curve Analysis**

- Cost elasticity rankings
- Maximum acceleration points
- Scenario-specific patterns

**4. Robustness Paradox**

- Stricter policies → more predictable outcomes
- Variability analysis across scenarios
- Planning uncertainty implications

**5. Regional Analysis**

- Country-specific sensitivities
- 25x variation (Luxembourg +25%, Czechia -7.3%)
- Recommended focus countries

**6. Strategic Recommendations**

- Data-driven procurement strategies
- Optimal matching percentages (40-50%)
- Technology investment priorities
- Policy effectiveness rankings

**Key Findings Highlight:**

1. **10% Tipping Point**: All scenarios show dramatic cost acceleration at 10% hourly matching threshold
2. **LDES Non-Negotiable**: +4.24% cost without it, statistically significant (p<0.001)
3. **EU Frontier Anomaly**: EU scenarios accelerate at 2% vs 97-117% for national scenarios
4. **Increasing Returns**: 46% cheaper per percentage point at 25-50% matching vs 0-25%
5. **Regional Extremes**: 25x variation - Luxembourg +25%, Czechia -7.3%

**Tips:**

- Read this tab first to understand key findings
- Reference specific insights when exploring other tabs
- Use findings to guide scenario comparisons
- Share strategic recommendations with stakeholders

---

## Scenario Configuration

**Format**: `[type]-[additionality]-[matching]`

**Examples:**

- `energy-match-25` = Energy matching with 25% additionality
- `hourly-match-50-99` = Hourly matching with 50% additionality at 99% matching level
- `hourly-match-noadd-50-99` = Hourly matching with NO additionality, 50% target, 99% matching

**Key Parameters:**

**Additionality Level:**

- `25` = 25% of clean energy must be new/additional
- `50` = 50% of clean energy must be new/additional
- `noadd` = No additionality requirement (can use existing clean energy)

**Matching Level:**

- `90` = 90% hourly matching requirement
- `95` = 95% hourly matching requirement
- `98` = 98% hourly matching requirement
- `99` = 99% hourly matching requirement

---

## Customization

### Changing Analyzed Scenarios

The dashboard automatically detects all scenarios in `results.csv`. To add or remove scenarios:

**1. Modify results.csv**:

- Add new columns with format: `year | scenario | scope`
- Ensure new scenarios follow existing data structure

**2. Restart dashboard**:

   ```bash
   python app.py
   ```

**3. Verify**: New scenarios appear in dropdown menus

**No code changes required** - the dashboard dynamically loads all available scenarios.

### Adding Custom Metrics

To add new metrics:

**1. Add data to results.csv**:

   ```csv
   (n) New Metric,Y-axis Label,carrier,value1,value2,value3,...
   ```

**2. Add color mappings** (optional) in `colors.csv`:

   ```csv
   (n) New Metric,carrier1,#color1
   (n) New Metric,carrier2,#color2
   ```

**3. Restart dashboard** - new metric appears in dropdown

### Customizing Colors

Edit `results/colors.csv`:

```csv
Results,carrier,color
(a) Energy mix,solar,#FFD700  # Change to gold
(a) Energy mix,onwind,#00BFFF  # Change to deep sky blue
```

**Color format**: Hex codes (#RRGGBB)

**Apply changes**: Restart dashboard

---

## Troubleshooting

### Common Issues

#### Dashboard won't start

**Error**: `ModuleNotFoundError: No module named 'dash'`

**Solution**:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyarrow
```

#### No data showing

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

#### Timeseries loading very slow

**Issue**: Timeseries queries take 30+ seconds

**Solution**:

**1.** Convert CSV to Parquet:

```python
import pandas as pd
df = pd.read_csv('results/results_time_series.csv')
df.to_parquet('results/results_time_series.parquet', compression='snappy')
```

**2.** Verify parquet file is detected:

```bash
ls -lh results/results_time_series.parquet
```

**3.** Restart dashboard

#### Colors not matching

**Issue**: Carriers show default colors instead of custom colors

**Possible causes**:

- `colors.csv` not found
- Carrier names don't match between `results.csv` and `colors.csv`
- Incorrect CSV format

**Solution**:

1. Verify file: `cat results/colors.csv | head -10`
2. Check carrier name spelling (case-sensitive)
3. Ensure format: `Results,carrier,color`

#### Port already in use

**Error**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Option 1: Kill existing process
lsof -ti:8050 | xargs kill -9

# Option 2: Use different port
# Edit app.py, change:
app.run_server(debug=True, host='0.0.0.0', port=8051)
```

#### Memory errors

**Error**: `MemoryError` or dashboard crashes

**Cause**: Insufficient RAM for large datasets

**Solutions**:

1. Use Parquet format for timeseries (better compression)
2. Use time range filtering instead of full year
3. Deploy on machine with more RAM (4GB+ recommended)

---

## Best Practices

### Data Exploration Workflow

1. **Start with Key Insights tab** to understand high-level findings
2. **Use Single Scenario Analysis** to explore individual scenarios
3. **Use Cross-Scenario Comparison** to measure policy impacts
4. **Use Dead Zone Analysis** to identify cost tipping points
5. **Use Timeseries Exploration** to understand operational details

### Performance Tips

1. **Use Parquet format** for timeseries data (10-20x faster than CSV)
2. **Start with smaller time ranges** (week/month) before loading full year
3. **Use caching** - repeated queries are instant
4. **Limit scenarios** in multi-scenario comparisons (max 4 recommended)
5. **Close unused browser tabs** to free memory

### Presentation Tips

1. **Use consistent plot types** across related analyses
2. **Include baseline** in comparisons for reference
3. **Use "Year on Year Evolution"** to show trends clearly
4. **Export plots** via browser's screenshot or Plotly's built-in save feature
5. **Cite specific insights** from Key Insights tab in presentations

---

## Browser Compatibility

Tested and working on:

- Chrome 90+
- Firefox 85+
- Safari 14+
- Edge 90+

---

## Support

For questions, issues, or feature requests:

1. Check this guide's Troubleshooting section
2. Review dashboard logs for error messages
3. Verify data structure matches expected format
4. Contact the development team with specific questions

**Dashboard Status**: Production-ready, actively maintained
