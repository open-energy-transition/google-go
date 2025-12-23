# Jupyter Notebooks

The project includes several Jupyter notebooks that provide flexible, interactive post-processing, and visualization capabilities. These notebooks allow users to select specific scenarios, years, and countries for customized analysis beyond the automated CSV generation workflow. The notebooks are organized into two categories: those that replicate and visualize the main results (Section 1), and those that produce additional analyses not included in the automated workflow.

#### General Structure

All notebooks follow a consistent three-section structure:

1. **Helpers (Section 0)**: Import statements loading the corresponding `generate_*.py` script and supporting libraries. This section also initializes plotting parameters and constants.

2. **Network Retrieval (Section 1)**: Interactive code cells where users specify:

   - Years to analyze (e.g., `[2025, 2030, 2035, 2040]`)
   - Scenarios to compare (e.g., `["baseline", "energy-match-50", "hourly-match-50-99"]`)
   - Geographic scope (e.g., system-wide, specific countries)

   The networks are then loaded into a pandas DataFrame indexed by year and scenario, with both full networks and stripped GO market sub-networks.

3. **Figure Generation (Section 2+)**: One or more sections with code cells that:

   - Call the appropriate `derive_*` functions from the imported script
   - Generate plots directly in the notebook
   - Optionally save figures to disk (controlled by `save_fig` flag)
   - Display results interactively for exploration

This modular structure enables rapid iteration: users can modify scenario selections in Section 1 and re-run Section 2 without reloading data, or adjust plotting parameters and regenerate specific figures independently.

#### Notebooks for Main Results

The following notebooks correspond directly to the three main result types:

- **`plot_frontier.ipynb`**: Visualizes energy procurement frontiers using functions from `generate_frontier.py`. Generates line plots showing the trade-off between energy matching and hourly matching for different scenarios, years, and geographic scopes (EU-wide, individual countries, or country comparisons).

- **`plot_time_series.ipynb`**: Plots hourly generation and demand time series using functions from `generate_time_series.py`. Users can select specific time windows (e.g., a week in May) to examine detailed system operation and GO market dynamics.

- **`plot_time_comparison.ipynb`**: Produces comparative visualizations using all functions from `generate_time_comparison.py`. Generates bar charts, heatmaps, and other plots for the 14 result types (a-n), with options to analyze system-wide or country-specific impacts.

#### Notebooks for Additional Results

Beyond the main automated results, the project includes notebooks for supplementary analyses:

- **`plot_country_comparison.ipynb`**: Generates country-level comparative analysis across years and scenarios using `generate_country_comparison.py`. This notebook:

  - Produces horizontal bar charts comparing countries side-by-side for selected metrics (energy mix, capacity, emissions, etc.)
  - Creates geographic heatmap visualizations on European maps showing spatial patterns of results
  - Enables identification of countries with similar or divergent outcomes across scenarios
  - Uses consistent color schemes and formatting for multi-country comparisons
  
  Key result types include energy mix (a), GO market generation (b), capacity mix (c-d), storage capacity (e1-e2), system costs (g), GO market revenue (h), marginal prices (i), emissions (j), and curtailment (k). An additional map visualization (b_map) shows GO market activity spatially.

- **`plot_resource_utilization(cap_vs_max_cap).ipynb`**: Analyzes variable renewable energy source (VRES) resource utilization using `generate_time_comparison.py` functions. This notebook:

  - Compares optimal installed VRES capacity against maximum technical potential for each technology and location
  - Calculates utilization percentages to identify whether resources are fully exploited or constrained
  - Visualizes utilization shares across scenarios and years to understand how GO market requirements affect renewable deployment patterns
  - Helps assess the spatial efficiency of renewable resource allocation
  
  This analysis corresponds to result type (n) in the time comparison framework and provides insights into grid integration constraints and siting limitations for renewable energy.

#### Supporting Scripts

Other than the scripts used for the main results, the notebooks utilize other supporting Python scripts that are imported but not executed directly:

- **`notebooks_function.py`**: Contains shared utility functions, constants, and helper routines used across all notebooks:

  - Carrier name mapping and color schemes (e.g., `rename_map`, `category_colors`)
  - Network preparation functions (`prepare_network()`, `drop_year()`, `rename_year()`)
  - GO market network extraction (`strip_network_GoO()`)
  - Statistics aggregation across multiple networks (`get_stats_all()`, `get_stats_prices()`)
  - Data cleaning utilities (`clean_virtual_names()`, `sum_except_color()`)
  - Figure size configuration constants (`FIGSIZE_BAR`, `FIGSIZE_HEATMAP_MAP`, `FIGSIZE_ARROW`)

- **`generate_country_comparison.py`**: Provides functions for country-level analysis, including plotting routines for bar charts (`plot_bar()`), map visualizations (`plot_map()`), and derive functions for all result types adapted for country comparison context.

The combination of automated CSV generation (Section 1), interactive dashboard (Section 2), and flexible notebooks (Section 3) provides a comprehensive post-processing framework that serves both standardized reporting needs and exploratory analysis requirements.
