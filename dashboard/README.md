# Google-Go Analysis Dashboard

Interactive dashboard for visualizing energy system results from all scenarios.

## Features

- **Five Main Tabs:**
  - **Single Scenario Analysis**: Visualize individual scenario results in detail
  - **Cross-Scenario Comparison**: Compare up to 4 scenarios side-by-side
  - **Dead Zone Analysis**: Frontier curve analysis across different spatial scopes
  - **Timeseries Exploration**: Explore hourly timeseries data across scenarios
  - **Key Insights**: View statistical analysis and strategic recommendations

- **Interactive Controls:**
  - **Year selection**: Choose from 2025, 2030, 2035, 2040
  - **Scenario selection**: baseline, energy-match, hourly-match variants
  - **Metric selection**: 13 different metrics including:
    - Energy mix
    - Capacity mix
    - Storage capacity
    - System costs
    - CO2 emissions
    - GO Market revenue
    - And more...
  - **Carrier filtering**: Select specific carriers to visualize
  - **Multiple plot types**: Bar charts, stacked area, pie charts, time series

- **Visualizations:**
  - Main plots with customizable chart types
  - Secondary breakdown plots (pie charts)
  - Summary statistics panels
  - Cross-scenario comparisons with grouped bars
  - Difference analysis plots
  - Summary comparison tables

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Using the run script (recommended)

```bash
cd dashboard
./run_dashboard.sh
```

### Option 2: Direct Python execution

```bash
cd dashboard
python app.py
```

Then open your browser to `http://localhost:8050`

## Data Structure

The dashboard expects data in the following consolidated structure:

```
results/
├── colors.csv                    # Color mappings for carriers
├── results.csv                   # Consolidated results for all scenarios
├── results-*.csv                 # Country-specific results (optional)
├── results_frontier.csv          # Frontier analysis data
├── results_time_series.parquet   # Timeseries data (parquet format, preferred)
└── results_time_series.csv       # Timeseries data (CSV fallback)
```

### Consolidated Results CSV Format

The consolidated `results.csv` file should have a **multi-level header structure**:
- **Level 0**: Year (2025, 2030, 2035, 2040)
- **Level 1**: Scenario name (baseline, energy-match-25, hourly-match-25-90, etc.)
- **Level 2**: Scope (system, country-specific)

And a **multi-level index**:
- **Level 0**: Result category (e.g., "(a) Energy mix", "(c) Capacity mix")
- **Level 1**: Y-axis label (e.g., "Net generation (TWh)", "Capacity (GW)")
- **Level 2**: Carrier (e.g., "solar", "onwind", "CCGT")

All scenarios are now included in one file, eliminating the need for separate CI_25, CI_50, and CI_noadd directories.

### Colors CSV Format

The colors.csv file should have three columns:
- **Column 0**: Result category
- **Column 1**: Carrier name
- **Column 2**: Hex color code

Example:
```
Results,carrier,color
(a) Energy mix,solar,#f9d002
(a) Energy mix,onwind,#235ebc
(a) Energy mix,CCGT,#a85522
```

## File Structure

```
dashboard/
├── app.py                          # Main application file
├── callbacks.py                    # All callback functions and plot generation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── run_dashboard.sh                # Convenient run script
├── layouts/                        # Tab layouts
│   ├── __init__.py
│   ├── single_scenario_layout.py   # Single scenario analysis tab
│   ├── cross_scenario_layout.py    # Cross-scenario comparison tab
│   ├── deadzone_layout.py          # Dead zone/frontier analysis tab
│   ├── timeseries_layout.py        # Timeseries exploration tab
│   └── insights_layout.py          # Key insights tab (static)
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and filtering utilities
│   └── colors.py                   # Color mapping from colors.csv
└── assets/                         # Static assets
    └── custom.css                  # Custom styling
```

## How It Works

1. **Data Loading**: On startup, the dashboard loads all CSV files from the results directory
2. **Interactive Selection**: Users select year, scenario, and metric from dropdowns
3. **Dynamic Filtering**: Data is filtered based on user selections
4. **Real-time Plotting**: Plotly generates interactive visualizations
5. **Color Consistency**: All plots use colors from colors.csv for consistency

## Plot Types

### Bar Charts
Vertical bar charts showing carrier-wise breakdown with color coding

### Stacked Area
Time series visualization (for applicable data)

### Pie Charts
Circular charts showing proportional breakdown by carrier

### Grouped Bars (Comparison Tab)
Side-by-side comparison of multiple scenarios

## Customization

### Adding New Plot Types

To add new plot types, modify the `create_plot()` function in `callbacks.py`:

```python
elif plot_type == 'my_new_type':
    fig = create_my_new_plot(data_series, metric, color_mapper)
```

### Adding New Metrics

The dashboard automatically detects metrics from the data. Simply ensure your results CSV includes the metric in the multi-level index.

### Styling

Add custom CSS to `assets/custom.css`. Dash will automatically load all CSS files in this directory.

### Color Schemes

Edit `results/colors.csv` to change colors for specific carriers. The dashboard will automatically use the updated colors on next startup.

## Performance Notes

- The dashboard loads all data at startup for better performance
- Large datasets may require increased memory (estimated ~500MB-1GB for full results)
- Data is cached in memory - refresh the app to reload data from disk
- For large datasets, consider filtering carriers to reduce plot complexity

## Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050 --workers 4
```

## Troubleshooting

### Dashboard won't start

- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the results directory structure matches the expected format
- Check console output for specific error messages

### No data showing

- Ensure CSV files are in the correct format with multi-level headers
- Check the browser console (F12) for JavaScript errors
- Verify the relative path `../results` points to your results directory
- Try running from the `dashboard/` directory

### Colors not matching

- Ensure `colors.csv` exists in the results directory
- Check carrier names match between results CSV and colors.csv
- The first column should be the metric category, not the carrier

### Port already in use

If port 8050 is already in use, modify `app.py`:

```python
app.run_server(debug=True, host='0.0.0.0', port=8051)  # Change port
```

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 85+
- Safari 14+
- Edge 90+

## Development

To modify the dashboard:

1. Edit layout files in `layouts/` to change the UI
2. Edit `callbacks.py` to modify plot generation logic
3. Edit `utils/data_loader.py` to change data loading behavior
4. Edit `assets/custom.css` to update styling

The app runs in debug mode by default, so changes will auto-reload.

## Credits

This dashboard is inspired by the plots in the notebooks directory and provides an interactive way to explore the large amounts of data generated by the Google-Go energy system analysis.
