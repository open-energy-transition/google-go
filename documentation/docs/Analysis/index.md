This section describes the post-processing workflow for analyzing and visualizing the results of the Google-GO project. The post-processing framework consists of three main components:

1.  **Main Results Generation**: Automated Python scripts (`generate_frontier.py`, `generate_time_series.py`, `generate_time_comparison.py`) that process network optimization results and output structured CSV files. These scripts quantify the energy procurement frontier, extract hourly time series data for electricity and GO markets, and provide comprehensive comparative metrics across scenarios and years, including energy mix, capacity expansion, costs, emissions, and market indicators.

2.  **Interactive Dashboard**: An AI-generated interactive dashboard built upon the CSV files from the main results generation. This dashboard provides a user-friendly interface for exploring the processed data and gaining insights through interactive visualizations.

3.  **Jupyter Notebooks**: A collection of Jupyter notebooks offering flexible and interactive post-processing and visualization capabilities. These notebooks are categorized into those that replicate and visualize the main results (e.g., `plot_frontier.ipynb`, `plot_time_series.ipynb`, `plot_time_comparison.ipynb`) and those that perform additional, customized analyses such as country-level comparisons (`plot_country_comparison.ipynb`) and resource utilization assessments (`plot_resource_utilization(cap_vs_max_cap).ipynb`). They allow users to select specific scenarios, years, and countries for in-depth analysis.

## Table Of Content

*   [Main Results](main_results_generation.md)
*   [Interactive Dashboard](interactive_dasboard.md)
*   [Jupyter Notebooks](jupyter_notebooks.md)