# Results Post-Processing

This document describes the post-processing workflow for analyzing and visualizing the results of the Google-GO project. The post-processing framework consists of three main components: (1) automated scripts that generate CSV files with key results, (2) an interactive dashboard for data exploration, and (3) Jupyter notebooks for customized analysis and visualization.

## 1. Main Results Generation

The main results are generated through three standalone Python scripts, each producing a dedicated CSV file containing processed data from the network optimization results. These scripts follow a consistent high-level structure: they iterate through all available network results (stored as `.nc` files in the `results/` directory), apply post-processing functions to extract relevant metrics, and aggregate the data into structured CSV outputs suitable for further analysis or visualization.

The three primary result types are:

- **Energy Procurement Frontier**: Quantifies the relationship between energy matching requirements and achievable hourly matching levels.
- **Time Series**: Extracts hourly generation and demand data for both the electricity system and the GO market.
- **Time Comparison**: Provides comprehensive comparative metrics across scenarios and years, including energy mix, capacity expansion, costs, emissions, and market indicators.

Each script is designed to be executed independently from the command line, with optional arguments to customize output paths and enable tutorial mode for testing purposes.

### 1.1 Energy Procurement Frontier (`generate_frontier.py`)

(TBD: change when we establish the final name fro the frontier)

This script generates the energy procurement frontier, which quantifies the trade-off between energy matching (percentage of annual renewable energy produced over the demand) and hourly matching (percentage of hours where renewable generation meets or exceeds demand) for different regions and scenarios.

#### Imported Modules and Dependencies

The script imports standard data science libraries (`pandas`, `numpy`), PyPSA for network handling, and other utility packages.

#### Core Functions

- **Score and Load Extraction**: `get_score_load(n, score="cfe")` extracts the clean/renewable energy score and electricity load from a network object. The score represents hourly clean energy availability (from `n.buses_t.cfe_p` or `n.buses_t.res_p`), while the load represents electricity demand. Both are grouped by country.

- **Frontier Calculation**: `get_hourly_energy_matrix(n, score, load)` computes the hourly matching percentage for energy matching levels ranging from 1% to 120%. For each energy matching level, it calculates:
  - The target load (total load scaled by energy matching level)
  - Unfilled energy (shortfall between target and available clean energy)
  - Remaining score (potential surplus clean energy)
  - Hourly matching percentage (fraction of target load met by clean energy)

  The function returns a pandas Series with hourly matching values indexed by energy matching percentage. Values become `NaN` when the constraint cannot be satisfied.

- **Main Processing Loop**: `main(output="results_frontier", score="cfe", tutorial=False)`: Orchestrates the frontier generation process:
  1. Scans the `../results/` directory recursively for all network files (`.nc`)
  2. For each network:
     - Loads the PyPSA network
     - Extracts score and load data
     - Calculates EU-wide frontier (aggregating all countries)
     - Calculates national-level frontiers for each country
  3. Organizes results in a MultiIndex DataFrame with dimensions: [scenario, year, country]
  4. Exports to CSV file

  The `tutorial` flag limits processing to the first 7 networks for testing purposes.

#### Script Execution

When run as a standalone script, it accepts command-line arguments:

```bash
python generate_frontier.py --output results_frontier --score cfe [--tutorial]
```

- `--output`: Name of output CSV file (default: `results_frontier`)
- `--score`: Score type to use, either `cfe` (carbon-free energy) or `res` (renewable energy source) (default: `cfe`)
- `--tutorial`: Enable tutorial mode for quick testing

The script suppresses PyPSA logging to WARNING level for cleaner output.

---

### 1.2 Time Series (`generate_time_series.py`)

This script extracts hourly time series data for both the electricity balance (grid-level generation and consumption) and the GO market (certificate generation and consumption), organizing the data for temporal analysis and visualization.

#### Imported Modules and Dependencies

The script imports standard libraries, PyPSA for network handling, and other utility packages.
In addition, this script imports helper functions from `notebooks_function.py`.

#### Core Functions

- **Electricity Balance Collection**: `collect_electricity_balance(n)` extracts the electricity energy balance from the network:
  1. Retrieves energy balance statistics grouped by country, bus carrier, and technology carrier
  2. Filters for AC and low voltage buses, excluding transmission and distribution infrastructure
  3. Aggregates generation/consumption by country and carrier
  4. Collects electricity load across all demand sectors
  5. Combines generation and load data with metadata labels ("area" for stacked area plot, "line" for demand overlay)
  6. Converts units to GW and adds result type labels
  
  Returns a DataFrame with MultiIndex: [country, carrier, type, Results, y_label]

- **GO Market Collection**: `collect_go_market(m)` extracts GO market time series from a stripped network containing only GO market components:
  1. Retrieves energy balance for GO market generators and consumers
  2. Cleans virtual carrier names (removes "virtual " prefix, to align with the electricity energy balance carrier naming)
  3. Separates generation (area plot) from demand (line plot)
  4. Combines and labels data appropriately
  5. Converts units to GW-GoO
  
  Returns a DataFrame with the same MultiIndex structure as electricity balance data.

- **Main Processing Loop**: `main(output="results_time_series", tutorial=False)` manages the time series extraction workflow:
  1. Scans the `../results/` directory recursively for all network files (`.nc`)
  2. For each network:
     - Loads the PyPSA network
     - Collects electricity balance time series
     - Strips and processes GO market network (if present)
     - Collects GO market time series
     - Labels data with scenario and year information
  3. Converts country codes to short names for readability
  4. Exports consolidated DataFrame to CSV

  The `tutorial` flag limits processing for testing purposes.

#### Script Execution

Command-line interface:

```bash
python generate_time_series.py --output results_time_series [--tutorial]
```

- `--output`: Name of output CSV file (default: `results_time_series`)
- `--tutorial`: Enable tutorial mode

---

### 1.3 Time Comparison (`generate_time_comparison.py`)

This script generates comprehensive comparative metrics across scenarios, years, and countries, producing 14 different result types (labeled *a* through *n*) that characterize the energy system transformation, market dynamics, and environmental impacts.

#### Imported Modules and Dependencies

The script imports standard libraries, PyPSA for network handling, and other utility packages.
In addition, this script imports helper functions from `notebooks_function.py`.

Also, constants are defined for unit conversions and plot formatting, and technology carrier lists define relevant subsets for specific analyses (e.g., `vres_carriers` for variable renewables, `clean_carriers` for carbon-neutral technologies).

#### Helper Functions

- **Data Processing Utilities**:

  - `_get_country_name(country)`: Converts country codes to short names; handles "system" keyword for EU-wide analysis
  - `_filter_by_country(df, country, country_column)`: Filters DataFrame by country; passes through unchanged for system-level analysis
  - `_prepare_csv_output(df, title, y_label, country)`: Structures data with MultiIndex for CSV export (Results, y_label, carrier) × (year, scenario, scope)
  - `_prepare_colors_output(colors, title)`: Adds MultiIndex to color mapping for consistent visualization
  - `_save_plot(ax, save_fig, fig_path, title)`: Handles figure saving with directory creation
  - `_calculate_ylim(df, margin)`: Computes appropriate y-axis limits with margin for both positive and negative stacked values
  - `_get_reference_network(df_networks)`: Retrieves a cached reference network for color scheme consistency

- **Plotting Functions**:
  - `plot_bar(df, colors, ylabel, title, figsize, vert_lines, ylim)` creates stacked bar charts with:
    - Configurable figure size based on number of years
    - Automatic handling of positive/negative stacks
    - Total value labels above/below bars
    - Optional vertical separator lines between scenarios
    - Custom legend positioning and formatting

  - `plot_bar_with_share(df, colors, df_share, ylabel, ylabel_share, title, figsize, vert_lines, ylim, ylim_share)` extends `plot_bar` with a secondary y-axis overlay showing percentage shares or ratios

  - `plot_heatmap(df, title, cmap_style, legend_title, figsize)` generates country-year heatmaps for spatial-temporal pattern visualization

  - `calculate_abatement_cost(df_cost, df_co2)` computes CO2 abatement cost as the cost difference divided by emissions difference relative to a baseline

  - `plot_abatement_cost_arrow(df, title, y_label, figsize, ylim)` creates arrow plot showing abatement cost progression with annotations

- **Technical Analysis Functions**: `get_cap_vres(df, carriers, groupby_list)` calculates VRES capacity utilization by comparing optimal capacity against maximum available potential

#### Core Result Derivation Functions

Each of the 14 main results is generated by a dedicated `derive_*` function following a consistent pattern:

1. Extract relevant data from network statistics (e.g., energy balance, capacity optimization, prices)
2. Filter and aggregate by country (if specified) or system-wide
3. Apply unit conversions and data transformations
4. Generate colors based on technology carriers
5. Optionally plot the results
6. Optionally save figures and/or CSV files
7. Return processed DataFrame and color mapping

- **Result Types**

  - (a) **`derive_energy_mix`**: Net electricity generation by technology carrier (TWh)
  - (b) **`derive_energy_mix_go`**: GO market generation by technology (TWh-GoO)
  - (c) **`derive_capacity_mix`**: Total installed capacity by technology (GW)
  - (d) **`derive_capacity_mix_new`**: New capacity additions (proxy for GO market impact) (GW)
  - (e1) **`derive_storage_energy_capacity`**: Energy storage capacity for GO market technologies (GWh)
  - (e2) **`derive_storage_power_capacity`**: Power storage capacity for GO market technologies (GW)
  - (f) **`derive_total_system_cost`**: Total system cost including all components (b€)
  - (g) **`derive_total_system_cost_new`**: Cost of new technologies (GO market proxy) (b€)
  - (h) **`derive_go_market_revenue`**: Revenue from GO certificate sales by technology (b€)
  - (i) **`derive_marginal_price`**: Marginal price of GO certificates for consumers (€/MWh)
  - (j) **`derive_co2_emissions`**: Total CO2 emissions from electricity generation (MtCO2)
  - (k) **`derive_cfe_curtailment`**: Clean energy curtailment (TWh)
  - (l) **`derive_cfe_utilization`**: Capacity factor of clean energy resources (-)
  - (m) **`derive_co2_abatement_cost`**: CO2 abatement cost relative to baseline (€/tCO2)
  - (n) **`derive_resource_utilization`**: VRES resource utilization (% of maximum potential). N.B.: this result is not saved in the CSV file, whereas is accounted for in the notebooks for additional results (See Section **3. Jupyter Notebooks** for more details)

- **Aggregation Function**: `derive_all_figures(df_networks, country, plot_fig, save_fig, fig_path, save_csv, figures, figsize_*)` orchestrates the generation of all requested figures:
  - Accepts a list of figure identifiers (e.g., `['a', 'b', 'c']`)
  - Calls corresponding `derive_*` functions
  - Aggregates results and colors into dictionaries
  - Returns consolidated DataFrames for CSV export

#### Network Management Functions

- `load_countries_from_config()` reads the list of modelled countries from `config.go.yaml`

- `save_df_csv(df_dict, path_csv, filename)` saves nested dictionaries of DataFrames to CSV:
  - Creates output directory if needed
  - Concatenates data across countries/results
  - Handles both results and color mappings

- `retrieve_networks(tutorial)` loads network results into a structured DataFrame:
  - Reads scenario and year lists from configuration
  - Loads PyPSA networks from result files
  - Prepares networks (adds carriers, colors, names)
  - Strips and stores GO market sub-networks
  - Returns DataFrame indexed by (year, scenario)

- **Main Processing Function**: `main(path_csv, results, colors, tutorial)` coordinates the entire time comparison workflow:
  1. Loads all network results via `retrieve_networks()`
  2. Retrieves country list from configuration
  3. Iterates through countries (or system level)
  4. Calls `derive_all_figures()` for each country
  5. Aggregates results and color mappings across countries
  6. Exports to CSV files

  In tutorial mode, only the first 2 countries are processed for testing.

#### Script Execution

Command-line interface:

```bash
python generate_time_comparison.py --path_csv figures/time_comparison/csv \
    --output_results results --output_colors colors [--tutorial]
```

- `--path_csv`: Output directory for CSV files (default: `figures/time_comparison/csv`)
- `--output_results`: Name for results CSV file without extension (default: `results`)
- `--output_colors`: Name for colors CSV file without extension (default: `colors`)
- `--tutorial`: Enable tutorial mode

---

## 2. Interactive Dashboard

*This section will contain information about the AI-generated dashboard built upon the three CSV files described in Section 1. Content to be added.*

---

## 3. Jupyter Notebooks

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
