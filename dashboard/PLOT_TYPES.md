# Plot Types in Google-Go Analysis Dashboard

## Overview

The dashboard now includes all the major plot types found in the notebooks, providing comprehensive visualization options for exploring the energy system results.

## Available Plot Types

### 1. **Bar Chart** (Single Year)
- **Description**: Simple bar chart showing carrier-wise breakdown for a selected year and scenario
- **Use Case**: Quick overview of energy mix, capacity mix, or other metrics for a specific year
- **Features**:
  - Color-coded by carrier (from colors.csv)
  - Interactive hover tooltips
  - Can filter specific carriers

### 2. **Stacked Bar (All Years)**
- **Description**: Stacked bar chart comparing all available years (2025, 2030, 2035, 2040) side by side
- **Use Case**: See how carrier contributions evolve across all years for a given scenario
- **Features**:
  - Each year shown as a separate stacked bar
  - Same carriers use consistent colors across years
  - Tooltips show carrier name and value
  - Legend shows all carriers (from first year)
- **Similar to**: `plot_bar` function in notebooks with vertical lines between years

### 3. **Year Comparison** (Grouped Bars)
- **Description**: Grouped bar chart showing how each carrier changes across years
- **Use Case**: Compare a specific carrier's values across different years
- **Features**:
  - Each year is a different colored group
  - X-axis shows carriers
  - Bars grouped by year
  - Excellent for tracking trends for specific carriers

### 4. **Pie Chart**
- **Description**: Circular chart showing proportional breakdown of carriers
- **Use Case**: Understand relative contributions of different carriers at a glance
- **Features**:
  - Color-coded by carrier
  - Shows percentages
  - Filters out negative values automatically

### 5. **Stacked Area**
- **Description**: Area plot showing carrier contributions over time
- **Use Case**: Time series visualization (when applicable)
- **Features**:
  - Currently simplified version
  - Can be extended for hourly/time series data
- **Similar to**: Time series plots in `plot_time_series.ipynb`

## Plot Types from Notebooks

### Coverage Status

| Plot Type | Notebook | Covered | Dashboard Implementation |
|-----------|----------|---------|-------------------------|
| Stacked Bar Charts | `country_comparison.ipynb`, `time_comparison.ipynb` | ✅ | "Stacked Bar (All Years)" |
| Simple Bar Charts | All notebooks | ✅ | "Bar Chart" |
| Pie Charts | Various | ✅ | "Pie Chart" |
| Year Comparison | `cap_vs_max_cap.ipynb` | ✅ | "Year Comparison" |
| Geographic Maps | `country_comparison.ipynb` | ⚠️ | Not yet implemented (requires geographic data) |
| Frontier Plots | `plot_frontier.ipynb` | ⚠️ | Not yet implemented (requires network data) |
| Time Series Area | `plot_time_series.ipynb` | ⚠️ | Partially implemented |

### Not Yet Implemented

The following specialized plot types from notebooks are not yet in the dashboard:

1. **Geographic Maps** (`plot_map` function)
   - Requires: Network data with bus locations, geographic coordinates
   - Shows: Spatial distribution on European map
   - Reason: Needs network (.nc) files, not just results CSV

2. **Energy Procurement Frontier Plots** (`plot_hourly_energy_matrix`)
   - Requires: Hourly matching vs energy matching data
   - Shows: Frontier curves for different scenarios
   - Reason: Requires specialized calculations from network data

3. **Time Series with Load Overlay** (`get_time_series`)
   - Requires: Hourly time series data
   - Shows: Generation/consumption over specific time periods with demand curve
   - Reason: Requires time-resolved data not in summary CSV

## How to Use Each Plot Type

### In Each Tab (CI_25, CI_50, CI_noadd):

1. Select **Year** (or leave unselected for year comparison plots)
2. Select **Scenario** (baseline, energy-match, hourly-match, etc.)
3. Select **Metric** (Energy mix, Capacity mix, System costs, etc.)
4. Select **Plot Type** from dropdown:
   - **Bar Chart**: For single year analysis
   - **Stacked Bar (All Years)**: To see all years together
   - **Year Comparison**: To compare years side-by-side
   - **Pie Chart**: For proportional breakdown
   - **Stacked Area**: For time-based visualization
5. Use **carrier checkboxes** to filter specific carriers

### In Comparison Tab:

- Compare the same metric across CI_25, CI_50, and CI_noadd scenarios
- See differences and similarities in approach
- Grouped bar charts automatically compare scenarios

## Plot Customization

All plots use:
- **Colors from colors.csv**: Consistent color scheme across all visualizations
- **Interactive tooltips**: Hover to see exact values
- **Responsive design**: Works on different screen sizes
- **Export capability**: Built-in Plotly toolbar to download as PNG/SVG

## Future Enhancements

Potential additions:
- [ ] Add geographic map visualization (requires network data integration)
- [ ] Add frontier plot for energy/hourly matching analysis
- [ ] Add full time series functionality with hourly data
- [ ] Add scenario comparison on frontier plots
- [ ] Add heatmaps for multi-dimensional comparisons
- [ ] Add box plots for distribution analysis

## Technical Details

### Data Flow

1. User selects filters (year, scenario, metric)
2. `data_loader.get_data()` retrieves filtered data from CSV
3. Plot function receives data and creates Plotly figure
4. `color_mapper.get_color()` assigns colors based on carrier
5. Interactive figure rendered in browser

### Color Consistency

All plots use the same color mapping:
- Defined in `/results/colors.csv`
- Mapped by metric category and carrier name
- Ensures visual consistency across all plots

### Performance

- Data pre-loaded at startup
- Plot generation on-demand
- Lightweight Plotly figures
- Client-side interactivity (zoom, pan, hover)
