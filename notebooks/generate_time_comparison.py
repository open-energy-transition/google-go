from notebooks_function import *

from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import re
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import pypsa
from collections import Counter
import os
import country_converter
import yaml

# Constants for unit conversions
MWH_TO_TWH = 1e6
EUR_TO_BILLION_EUR = 1e9
MW_TO_GW = 1e3
TONS_TO_MEGATONS = 1e6

plot_formats = {
    "Energy mix": {"letter": "(a)", "y_label": "Net generation (TWh)"},
    "Energy mix - GO Market": {"letter": "(b)", "y_label": "Net generation (TWh-GoO)"},
    "Capacity mix": {"letter": "(c)", "y_label": "Capacity (GW)"},
    "Capacity mix - new technologies (proxy of GO Market)": {"letter": "(d)", "y_label": "Capacity (GW)"},
    "Storage in GO Market - Energy capacity": {"letter": "(e)", "y_label": "Energy capacity (GWh)"},
    "Storage in GO Market - Power capacity": {"letter": "(e)", "y_label": "Power capacity (GW)"},
    "Total system cost": {"letter": "(f)", "y_label": "Total system cost (b€)"},
    "Total system cost - new technologies (proxy of GO Market)": {"letter": "(g)", "y_label": "Total system cost (b€)"},
    "GO Market revenue by technology": {"letter": "(h)", "y_label": "Market size (b€)"},
    "Marginal price of GoO consumers": {"letter": "(i)", "y_label": "Marginal price (€/MWh)"},
    "CO2 emissions": {"letter": "(j)", "y_label": "CO2 emissions (MtCO2)"},
    "CFE curtailment": {"letter": "(k)", "y_label": "Curtailment (TWh)"},
    "CFE utilization": {"letter": "(l)", "y_label": "Utilization factor (-)"},
    "CO2 abatement cost": {"letter": "(m)", "y_label": "Abatement cost (€/tCO2)"},
}

vres_carriers = [
    "solar",
    "solar rooftop",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
]

clean_carriers = [
    "green_ocgt", "adv_firm_tech", "nuclear", "allam",
    "offwind", "offwind-ac", "offwind-dc", "offwind-float", "onwind", "solar", "solar-hsat", "solar rooftop",
    "ror", "hydro", "urban central solid biomass CHP", "geothermal",
    ]

### Helper functions
def _get_country_name(country: Optional[str]) -> str:
    """Convert country code to display name."""
    if country == "system":
        return "system"
    return country_converter.CountryConverter().convert(country, to="short_name")

def _filter_by_country(df: pd.DataFrame, country: Optional[str], country_column: str = "country1") -> pd.DataFrame:
    """Filter dataframe by country if not 'system'."""
    if country == "system":
        return df
    return df[df.index.get_level_values(country_column).isin([country])]

def _prepare_csv_output(df: pd.DataFrame, title: str, y_label: str, country: str) -> pd.DataFrame:
    """Prepare dataframe for CSV export with proper MultiIndex structure."""
    df_csv = df.copy()
    country_name = _get_country_name(country)
    df_csv.index = pd.MultiIndex.from_product(
        [[title], [y_label], df_csv.index], 
        names=['Results', 'y_label', 'carrier']
    )
    df_csv.columns = pd.MultiIndex.from_product(
        [df_csv.columns.get_level_values("scenario").unique().tolist(), 
         df_csv.columns.get_level_values("year").unique().tolist(), 
         [country_name]], 
        names=['scenario', 'year', 'scope']
    )
    return df_csv

def _prepare_colors_output(colors: pd.Series, title: str) -> pd.Series:
    """Prepare colors series with proper MultiIndex."""
    colors_out = colors.copy()
    colors_out.index = pd.MultiIndex.from_product(
        [[title], colors_out.index], 
        names=['Results', 'carrier']
    )
    return colors_out

def _save_plot(ax, save_fig: bool, fig_path: Optional[str], title: str) -> None:
    """Save plot to file if requested."""
    if save_fig and fig_path:
        os.makedirs(fig_path, exist_ok=True)
        ax.figure.savefig(f"{fig_path}/{title}.png", dpi=300, bbox_inches="tight")

# Simple cache for reference network (avoids DataFrame hashing issues)
_reference_network_cache = {}

def _get_reference_network(df_networks: pd.DataFrame) -> pypsa.Network:
    """Get a reference network, preferring ones with 'match' in scenario name.
    Cached to avoid repeated lookups.
    """
    # Use id() as cache key since DataFrames aren't hashable
    cache_key = id(df_networks)
    
    if cache_key not in _reference_network_cache:
        match_nets = df_networks[df_networks.index.get_level_values("scenario").str.contains("match", case=False)]
        _reference_network_cache[cache_key] = (
            match_nets["network"].iloc[0] if len(match_nets) > 0 else df_networks["network"].iloc[0]
        )
    
    return _reference_network_cache[cache_key]

### Functions to plot figures
def plot_bar(df, colors, ylabel=None, title=None, figsize=(12, 6), vert_lines=True, ylim=False):
    # Create figure and axis with adjustable size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean 'virtual ' prefix from index names
    df = clean_virtual_names(df.copy())
    
    # Remove column names from MultiIndex if present
    df_plot = df.copy()
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = df_plot.columns.set_names([None, None])

    # Create stacked bar plot on the provided axis
    df_plot.T.plot(kind="bar", stacked=True, legend=True, color=colors, ax=ax)

    # Reverse legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc='upper left', bbox_to_anchor=(1, 1),
              title='Carrier')

    # Add total values above each bar
    bar_totals = df.T.sum(axis=1)  # totals per bar
    bar_height = df.clip(lower=0).T.sum(axis=1)
    
    if ylim:
        ax.set_ylim(ylim)
    
    for i, total in enumerate(bar_totals):
        color = "black"
        
        if ylim:
            if bar_height.iloc[i] > ylim[1]:
                bar_height.iloc[i] = ylim[1]
                color = "red"
        
        ax.text(
            i,                              # x position
            bar_height.iloc[i],             # y position
            "{:.0f}".format(total),         # label
            ha='center', va='bottom',
            color=color,
            fontsize=9
        )

    # Check if we have MultiIndex columns (year, scenario structure)
    if isinstance(df.columns, pd.MultiIndex):
        # Get years and scenarios
        years = df.columns.get_level_values('year').unique()
        scenarios = df.columns.get_level_values('scenario').unique()
        
        # Set x-axis labels to show only scenarios
        ax.set_xticklabels([sc for _, sc in df.columns], rotation=45, ha='right')
        
        # Add year labels as a secondary x-axis
        # Calculate position for each year group
        year_positions = []
        year_labels = [] 
        for year in years:
            year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
            if year_cols:
                year_positions.append(np.mean(year_cols))
                year_labels.append(str(year))
        
        # Add vertical lines between years
        if vert_lines and len(years) > 1:
            for year in years[:-1]:
                year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
                if year_cols:
                    ax.axvline(max(year_cols) + 0.5, color='gray', linestyle='--', linewidth=1)
        
        # Add secondary x-axis for years
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(year_positions)
        ax2.set_xticklabels(year_labels)
        ax2.tick_params(axis='x', which='both', length=0)  # Hide tick marks
        ax2.spines['top'].set_visible(False)
    else:
        # Fallback for non-MultiIndex columns (old behavior)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        
        if vert_lines:
            group_bars = max(Counter([int(c[-4:]) for c in df.columns]).values())
            num_bars = df.T.shape[0]
            for i in range(0, num_bars, group_bars):
                ax.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)
    
    # Labels
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(axis='y')

    return ax


def plot_bar_with_share(df, colors, df_share, ylabel=None, ylabel_share=None, title=None, figsize=(12, 6), vert_lines=True, ylim=False, ylim_share=False):
    # Create figure and axis with adjustable size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean 'virtual ' prefix from index names
    df = clean_virtual_names(df.copy())
    
    # Remove column names from MultiIndex if present
    df_plot = df.copy()
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = df_plot.columns.set_names([None, None])

    # Create stacked bar plot on the provided axis
    df_plot.T.plot(kind="bar", stacked=True, legend=False, color=colors, ax=ax)

    # Create secondary y-axis for share line plot
    ax_share = ax.twinx()
    
    if ylim_share:
        ax_share.set_ylim(ylim_share)
    
    # Check if we have MultiIndex columns (year, scenario structure)
    if isinstance(df.columns, pd.MultiIndex):
        # Get years and scenarios
        years = df.columns.get_level_values('year').unique()
        scenarios = df.columns.get_level_values('scenario').unique()
        
        # Set x-axis labels to show only scenarios
        ax.set_xticklabels([sc for _, sc in df.columns], rotation=45, ha='right')
        
        # Prepare share data grouped by year
        share_by_year = {}
        for year in years:
            year_mask = df_share.columns.get_level_values('year') == year
            year_cols = df_share.columns[year_mask]
            if len(year_cols) > 0:
                share_by_year[year] = df_share.loc["total", year_cols]
        
        # Plot line for each year
        for idx, (year, year_data) in enumerate(share_by_year.items()):
            # Get x positions for this year
            x_positions = [i for i, (y, s) in enumerate(df.columns) if y == year]
            y_values = year_data.values * 100  # Convert to percentage
            
            # Plot line with markers
            ax_share.plot(x_positions, y_values, 
                         marker='o', linewidth=2, markersize=8,
                         label=f'{year}', color='black')
            
            # Add value labels on points
            for x, y in zip(x_positions, y_values):
                ax_share.text(x - 0.1, y + 0.1, f'{y:.1f}%', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color='black')
        
        # Add year labels as a secondary x-axis
        year_positions = []
        year_labels = []
        for year in years:
            year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
            if year_cols:
                year_positions.append(np.mean(year_cols))
                year_labels.append(str(year))
        
        # Add vertical lines between years
        if vert_lines and len(years) > 1:
            for year in years[:-1]:
                year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
                if year_cols:
                    ax.axvline(max(year_cols) + 0.5, color='gray', linestyle='--', linewidth=1)
        
        # Add secondary x-axis for years
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(year_positions)
        ax2.set_xticklabels(year_labels)
        ax2.tick_params(axis='x', which='both', length=0)  # Hide tick marks
        ax2.spines['top'].set_visible(False)
        
        # Add legend for share lines
        #ax_share.legend(loc='upper right', bbox_to_anchor=(1, 0.9), title='Share by Year')
        
    else:
        # Fallback for non-MultiIndex columns (old behavior)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        
        # Plot share as a simple line
        x_positions = range(len(df.columns))
        y_values = df_share.loc["total"].values * 100  # Convert to percentage
        
        ax_share.plot(x_positions, y_values, 
                     marker='o', linewidth=2, markersize=8,
                     color='red', label='Share')
        
        # Add value labels on points
        for x, y in zip(x_positions, y_values):
            ax_share.text(x - 0.1, y + 0.5, f'{y:.1f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='black')
        
        if vert_lines:
            group_bars = max(Counter([int(c[-4:]) for c in df.columns]).values())
            num_bars = df.T.shape[0]
            for i in range(0, num_bars, group_bars):
                ax.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)
        
        #ax_share.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    
    # Create handles for bars
    handles_bars = [plt.Rectangle((0,0),1,1, fc=colors[i]) for i in range(len(df.index))]
    labels_bars = df.index.tolist()
    
    # Add share line to legend
    line_handle = Line2D([0], [0], color='black', linewidth=2, marker='o', 
                         markersize=8, label='total (right axis)')
    
    # Combine: bars (reversed) + share line
    handles_combined = handles_bars[::-1] + [line_handle]
    labels_combined = labels_bars[::-1] + ['total (right axis)']
    
    ax.legend(handles_combined, labels_combined,
              loc='upper left', bbox_to_anchor=(1.05, 1),)
    
    # Labels
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylabel_share:
        ax_share.set_ylabel(ylabel_share)
    else:
        ax_share.set_ylabel('Share (%)')
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)

    ax.grid(axis='y')
    ax_share.set_ylim(bottom=0)  # Start share axis from 0

    return ax, ax_share

def plot_heatmap(df, title, cmap_style, legend_title, figsize=(16,8)):
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(df, 
                annot=True,           # Show values in cells
                fmt='.2f',            # Format with 2 decimals
                cmap=cmap_style,        # Red-Yellow-Green colormap
                vmin=0, 
                vmax=1,
                cbar_kws={'label': legend_title},
                linewidths=0.5,       # Add grid lines
                linecolor='gray',
                ax=ax)

    # Get scenario names for x-axis labels (extract from MultiIndex)

    scenario_labels = [sc for _, sc in df.columns]
    ax.set_xticklabels(scenario_labels, rotation=45, ha='right')

    # Add vertical white lines between years
    years = df.columns.get_level_values('year').unique()
    if len(years) > 1:
        for year in years[:-1]:
            year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
            if year_cols:
                ax.axvline(max(year_cols) + 1, color='black' if cmap_style == 'RdYlGn' else 'white', linewidth=2)


    # Add secondary x-axis for years at the top
    year_positions = []
    year_labels = []
    for year in years:
        year_cols = [i for i, (y, s) in enumerate(df.columns) if y == year]
        if year_cols:
            year_positions.append(np.mean(year_cols)+0.5)
            year_labels.append(str(year))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(year_positions)
    ax2.set_xticklabels(year_labels)
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.spines['top'].set_visible(False)

    # Improve aesthetics
    ax.set_xlabel("")
    ax.set_ylabel('Carrier', fontweight='bold')
    ax.set_title(title)

    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return ax

def calculate_abatement_cost(df_cost, df_co2):
    # Get unique years
    years = df_cost.columns.get_level_values('year').unique()

    # Initialize DataFrame for MAC results
    ac_data = []

    for year in years:
        # Get baseline values for this year
        baseline_cost = df_cost.loc["total", (year, "baseline")]
        baseline_emissions = df_co2.loc["total", (year, "baseline")]
        
        # Get all scenarios for this year
        year_scenarios = [sc for y, sc in df_cost.columns if y == year and sc != "baseline"]
        
        for scenario in year_scenarios:
            scenario_cost = df_cost.loc["total", (year, scenario)]
            scenario_emissions = df_co2.loc["total", (year, scenario)]
            
            # Calculate MAC (in EUR/tCO2, since costs are in billion EUR and emissions in MtCO2)
            delta_cost = abs(scenario_cost - baseline_cost)  # billion EUR
            delta_emissions = abs(scenario_emissions - baseline_emissions)  # MtCO2
            
            if delta_emissions > 0:
                ac = (delta_cost * 1e9) / (delta_emissions * 1e6)  # EUR/tCO2
            else:
                ac = 0
            
            ac_data.append({
                'year': year,
                'scenario': scenario,
                'AC': ac,
                'delta_cost': delta_cost,
                'delta_emissions': delta_emissions
            })

    # Create DataFrame
    df_ac = pd.DataFrame(ac_data)
    # Pivot to have scenarios as columns and years as rows for easier visualization
    df_ac_pivot = df_ac.pivot(index='scenario', columns='year', values='AC')
    
    return df_ac_pivot

def plot_abatement_cost_arrow(df, title, y_label):
    
    fig, ax = plt.subplots(figsize=(4, 6))
    
    x_positions = range(len(df))
    colors = plt.cm.tab10.colors  # color palette for different years
    
    # Plot dots and draw arrows dynamically
    for col_idx, col_name in enumerate(df.columns):
        y_values = df[col_name]
        ax.scatter(x_positions, y_values, color=colors[col_idx % len(colors)], label=col_name)
        
        # Draw arrows from previous column to current column
        if col_idx > 0:
            prev_values = df.iloc[:, col_idx - 1]
            for i, (y_prev, y_curr) in enumerate(zip(prev_values, y_values)):
                ax.annotate("", xy=(i, y_curr), xytext=(i, y_prev),
                            arrowprops=dict(arrowstyle="->", color='gray', lw=1))
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df.index, rotation=90)
    
    # Grid and legend
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(title="Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim([0,250])
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Year")
    
    plt.tight_layout()
    #plt.show()

    return ax

### Functions to derive figures (a-m)
def derive_energy_mix(df_networks: pd.DataFrame, country: Optional[str] = None, 
                     plot_fig: bool = False, save_fig: bool = False, 
                     fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive energy mix statistics from networks.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Energy mix"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "energy_balance",
                               groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","country1"])
    df["color"] = colors
    df = df[
        df.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & ~df.index.get_level_values("carrier").isin(["AC", "DC", "electricity", "low voltage",
                                                       "electricity distribution grid", "BEV charger",
                                                       "home battery charger", "home battery discharger"])
    ]
    df = _filter_by_country(df, country)
    df = df.groupby("carrier").apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby("carrier").apply(sum_except_color)
    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MWH_TO_TWH

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_energy_mix_go(df_networks: pd.DataFrame, country: Optional[str] = None, 
                        plot_fig: bool = False, save_fig: bool = False, 
                        fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive GO Market energy mix statistics.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Energy mix - GO Market"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df_GoO = df_networks[~df_networks.index.get_level_values("scenario").str.contains("baseline")]
    df, colors = get_stats_all(df_GoO["GoO"],
                               "energy_balance",
                               groupby=["carrier", "bus_carrier"] if country == "system" else ["carrier","country1"])
    df["color"] = colors
    df = df[df.index.get_level_values("carrier") != "GoO"]
    df = _filter_by_country(df, country)
    df = df.groupby("carrier").apply(sum_except_color)
    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MWH_TO_TWH

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_capacity_mix(df_networks: pd.DataFrame, country: Optional[str] = None, 
                       plot_fig: bool = False, save_fig: bool = False, 
                       fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive capacity mix statistics from networks.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Capacity mix"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "optimal_capacity",
                               groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","country1"])
    df["color"] = colors

    df = df[
        ~df.index.get_level_values("bus_carrier").isin(["GoO","co2","co2 stored"])
        & ~df.index.get_level_values("carrier").isin(["AC","DC","electricity distribution grid","low voltage"] + list(grouping_storage.keys()))
        & ~df.index.get_level_values("component").isin(["Store","StorageUnit"])
    ]
    df = _filter_by_country(df, country)
    df = df.loc[~df.index.isin([("Generator","coal","coal"),("Generator","gas","gas")])]
    df = df.rename(index=grouping_storage).groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MW_TO_GW

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_capacity_mix_new(df_networks: pd.DataFrame, country: Optional[str] = None, 
                           plot_fig: bool = False, save_fig: bool = False, 
                           fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive capacity mix for new technologies (proxy of GO Market).
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Capacity mix - new technologies (proxy of GO Market)"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "optimal_capacity",
                               groupby=["carrier","bus_carrier","build_year"] if country == "system" else ["carrier","bus_carrier","build_year","country1"])

    for c in df.columns:
        scenario_year = int(c[0])
        df.loc[df.index.get_level_values("build_year") != scenario_year,c] = 0

    df["color"] = colors

    df = df[
        ~df.index.get_level_values("bus_carrier").isin(["GoO","co2","co2 stored"])
        & ~df.index.get_level_values("carrier").isin(["AC","DC","electricity distribution grid","low voltage"] + list(grouping_storage.keys()))
        & ~df.index.get_level_values("component").isin(["Store","StorageUnit"])
    ]
    df = _filter_by_country(df, country)
    df = df.loc[~df.index.isin([("Generator","coal","coal"),("Generator","gas","gas")])]
    df = df.rename(index=grouping_storage).groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MW_TO_GW

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_storage_energy_capacity(df_networks: pd.DataFrame, country: Optional[str] = None, 
                                   plot_fig: bool = False, save_fig: bool = False, 
                                   fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive storage energy capacity in GO Market.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Storage in GO Market - Energy capacity"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "optimal_capacity",
                               groupby=["carrier"] if country == "system" else ["carrier","country1"])
    df["color"] = colors

    df = df[
        df.index.get_level_values("component").isin(["Store"])
        & df.index.get_level_values("carrier").isin(list(grouping_storage.keys()))
        & ~df.index.get_level_values("carrier").isin(['EV battery','home battery'])
    ]
    df = _filter_by_country(df, country)
    df = df.groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MW_TO_GW

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_storage_power_capacity(df_networks: pd.DataFrame, country: Optional[str] = None, 
                                  plot_fig: bool = False, save_fig: bool = False, 
                                  fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive storage power capacity in GO Market.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Storage in GO Market - Power capacity"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "optimal_capacity",
                               groupby=["carrier"] if country == "system" else ["carrier","country1"])
    df["color"] = colors

    df = df[
        df.index.get_level_values("component").isin(["Link"])
        & df.index.get_level_values("carrier").isin(list(grouping_storage.keys()))
        & ~df.index.get_level_values("carrier").isin(['BEV charger','li-ion discharger','iron-air discharger','home battery charger','home battery discharger'])
    ]
    df = _filter_by_country(df, country)
    df = df.groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MW_TO_GW

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_total_system_cost(df_networks: pd.DataFrame, country: Optional[str] = None, 
                            plot_fig: bool = False, save_fig: bool = False, 
                            fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive total system cost.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    if country != "system":
        return None, None
    
    results = "Total system cost"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, _ = get_stats_all(df_networks["network"], "system_cost")
    
    df = df.rename(index=rename_map).groupby("carrier").sum()
    df = df[df.index != "GO penalty"]
    df = df / EUR_TO_BILLION_EUR

    colors = pd.Series([category_colors[i] for i in df.index], index=df.index)

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_total_system_cost_new(df_networks: pd.DataFrame, country: Optional[str] = None, 
                                plot_fig: bool = False, save_fig: bool = False, 
                                fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive total system cost for new technologies (proxy of GO Market).
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Total system cost - new technologies (proxy of GO Market)"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, _ = get_stats_all(df_networks["network"],
                         "system_cost",
                         groupby=["carrier","build_year"] if country == "system" else ["carrier","build_year","country1"])
    
    if country != "system":
        df = _filter_by_country(df, country)

    for c in df.columns:
        scenario_year = int(c[0])
        df.loc[df.index.get_level_values("build_year") != scenario_year,c] = 0

    df = df.rename(index=rename_map).groupby("carrier").sum()
    df = df[df.index != "electricity grid"]
    df = df / EUR_TO_BILLION_EUR

    colors = pd.Series([category_colors[i] for i in df.index], index=df.index)

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_go_market_revenue(df_networks: pd.DataFrame, country: Optional[str] = None, 
                            plot_fig: bool = False, save_fig: bool = False, 
                            fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive GO Market revenue by technology.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "GO Market revenue by technology"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df_GoO = df_networks[~df_networks.index.get_level_values("scenario").str.contains("baseline")]
    df, colors = get_stats_all(df_GoO["GoO"],
                               "revenue",
                               groupby=["carrier"] if country == "system" else ["carrier","country1"])
    df["color"] = colors
    df = df[df.index.get_level_values("carrier") != "GoO"]
    df = _filter_by_country(df, country)
    df = df.groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / EUR_TO_BILLION_EUR

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_marginal_price(df_networks: pd.DataFrame, country: Optional[str] = None, 
                         plot_fig: bool = False, save_fig: bool = False, 
                         fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive marginal price of GoO consumers.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "Marginal price of GoO consumers"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, _ = get_stats_prices(df_networks["network"])
    
    keywords = ["GO Demand", "low voltage"]
    pattern = "|".join(keywords)
    
    # Use cached reference network
    n = _get_reference_network(df_networks)
    
    df = df[
        df.index.str.extract(f"({pattern})", expand=False).isin(keywords)
        & (df.index.map(n.buses.country).isin([country]) if country != "system" else True)
    ]
    
    if country == "system":
        # System-level: weighted average
        df_w, _ = get_stats_all(df_networks["network"], "energy_balance", groupby=["carrier","bus"], comps=["Load"])
        df_w = df_w[df_w.index.get_level_values("bus").str.extract(f"({pattern})", expand=False).isin(keywords)]
        df_w = df_w.groupby("bus").sum()

        denominator = (
            df_w.groupby(df_w.index.map(n.buses.carrier))
                .transform("sum")
                .reindex(df.index)
        )

        df_weighted = df.mul(df_w).div(denominator)
        df = df_weighted.groupby(df_weighted.index.map(n.buses.carrier)).sum()
        # Filter to available carriers in the data
        available_carriers = [c for c in ["low voltage", "GoO"] if c in df.index]
        if available_carriers:
            df = df.loc[available_carriers]
    else:
        # Country-level: group by carrier
        df.index = df.index.map(n.buses.carrier)
        df = df.groupby(df.index).sum()
    
    colors = pd.Series(df.index.map(n.carriers.color), index=df.index)

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_co2_emissions(df_networks: pd.DataFrame, country: Optional[str] = None, 
                        plot_fig: bool = False, save_fig: bool = False, 
                        fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive CO2 emissions.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "CO2 emissions"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "energy_balance",
                               groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","name"])
    df["color"] = colors
    
    df = df[
        df.index.get_level_values("bus_carrier").isin(["co2"])
        & df.index.get_level_values("component").isin(["Link"])
        & (df.index.get_level_values("name").str.contains(country) if country != "system" else True)
    ]
    df = df.groupby("carrier").apply(sum_except_color)
    
    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / TONS_TO_MEGATONS

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_bar(df, colors, ylabel=y_label, title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df, title_fig, y_label, country), _prepare_colors_output(colors, title_fig)
    return None, None

def derive_cfe_curtailment(df_networks: pd.DataFrame, country: Optional[str] = None, 
                          plot_fig: bool = False, save_fig: bool = False, 
                          fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive CFE curtailment statistics.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "CFE curtailment"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    df, colors = get_stats_all(df_networks["network"],
                               "curtailment",
                               groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","name"])
    df["color"] = colors

    df = df[
        df.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & df.index.get_level_values("carrier").isin(vres_carriers)
        & (df.index.get_level_values("name").str.contains(country) if country != "system" else True)
    ]

    df = df.groupby("carrier").apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby("carrier").apply(sum_except_color)

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MWH_TO_TWH

    # derive total curtailment and curtailment shares
    df_curt = df.copy()
    df_curt.loc["total"] = df_curt.sum(axis=0)

    df_gen, _ = get_stats_all(df_networks["network"],
                              "energy_balance",
                              groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","country1"])

    df_gen = df_gen[
        df_gen.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & df_gen.index.get_level_values("carrier").isin(df_curt.index)
    ]
    df_gen = _filter_by_country(df_gen, country)
    df_gen = df_gen.groupby("carrier").sum()
    df_gen = df_gen / MWH_TO_TWH
    df_gen.loc["total"] = df_gen.sum(axis=0)

    df_curt_share = df_curt / (df_gen + df_curt)

    country_name = _get_country_name(country)
    if plot_fig:
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax, ax_share = plot_bar_with_share(df, colors, df_curt_share,
                                            ylabel=y_label,
                                            ylabel_share="Curtailment share (%)",
                                            title=plot_title)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        df_csv_curt = _prepare_csv_output(df_curt, title_fig, y_label, country)
        df_csv_share = _prepare_csv_output(df_curt_share, title_fig, "Curtailment share (-)", country)
        df_csv = pd.concat([df_csv_curt, df_csv_share], axis=0)
        return df_csv, _prepare_colors_output(colors, title_fig)
    return None, None

def derive_cfe_utilization(df_networks: pd.DataFrame, country: Optional[str] = None, 
                          plot_fig: bool = False, save_fig: bool = False, 
                          fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive CFE utilization statistics.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    results = "CFE utilization"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    # Net yearly generation
    df_gen, _ = get_stats_all(df_networks["network"],
                              "energy_balance",
                              groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","country1"])

    df_gen = df_gen[
        df_gen.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & df_gen.index.get_level_values("carrier").isin(clean_carriers)
    ]
    df_gen = _filter_by_country(df_gen, country).groupby("carrier").sum()

    # Nominal yearly generation
    df_cap, _ = get_stats_all(df_networks["network"],
                              "optimal_capacity",
                              groupby=["carrier","bus_carrier"] if country == "system" else ["carrier","bus_carrier","country1"])

    df_cap = df_cap[
        ~df_cap.index.get_level_values("bus_carrier").isin(["GoO","co2","co2 stored"])
        & df_cap.index.get_level_values("carrier").isin(clean_carriers)
    ]
    df_cap = _filter_by_country(df_cap, country).groupby("carrier").sum()

    # Calculate utilization rate
    df_use = round(df_gen / (df_cap * 8760), 2)

    country_name = _get_country_name(country)
    if plot_fig:
        cmap_style = 'viridis'
        plot_title = title_fig if country == "system" else f"{title_fig} - {country_name}"
        ax = plot_heatmap(df_use, plot_title, cmap_style, legend_title=y_label)
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        return _prepare_csv_output(df_use, title_fig, y_label, country), None
    return None, None

def derive_co2_abatement_cost(df_networks: pd.DataFrame, country: Optional[str] = None, 
                              plot_fig: bool = False, save_fig: bool = False, 
                              fig_path: Optional[str] = None, save_csv: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Derive CO2 abatement cost.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plot
        save_fig: Whether to save plot to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        
    Returns:
        Tuple of (dataframe with results, series with colors) if save_csv=True, else (None, None)
    """
    if country != "system":
        return None, None

    results = "CO2 abatement cost"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title_fig = f"{letter} {results}"

    # Derive total system cost
    df, _ = get_stats_all(df_networks["network"], "system_cost")
    
    if country != "system":
        df = _filter_by_country(df, country)
    
    df = df.rename(index=rename_map).groupby("carrier").sum()
    df = df[df.index != "GO penalty"]
    df = df / EUR_TO_BILLION_EUR
    df_cost = df.copy()
    df_cost.loc["total"] = df_cost.sum(axis=0)

    # Derive CO2 emissions
    df, _ = get_stats_all(df_networks["network"],
                         "energy_balance",
                         groupby=["carrier","bus_carrier"])
    
    df = df[
        df.index.get_level_values("bus_carrier").isin(["co2"])
        & df.index.get_level_values("component").isin(["Link"])
        & (df.index.get_level_values("name").str.contains(country) if country != "system" else True)
    ]
    df = df.groupby("carrier").sum()
    df = df / TONS_TO_MEGATONS
    df_co2 = df.copy()
    df_co2.loc["total"] = df_co2.sum(axis=0)

    df_ac = calculate_abatement_cost(df_cost, df_co2)
    ax = plot_abatement_cost_arrow(df_ac, title=title_fig, y_label=y_label)

    country_name = _get_country_name(country)
    if plot_fig:
        _save_plot(ax, save_fig, fig_path, title_fig)

    if save_csv:
        df_csv = df_ac.copy().unstack("scenario").to_frame().T
        df_csv["carrier"] = "abatement cost"
        df_csv = df_csv.set_index("carrier")
        return _prepare_csv_output(df_csv, title_fig, y_label, country), None
    return None, None

def derive_all_figures(df_networks, country=None, plot_fig=False, save_fig=False, fig_path=None, save_csv=False, figures=None):
    """Derive all or selected figures.
    
    Args:
        df_networks: DataFrame containing network data
        country: Country code or 'system' for system-wide analysis
        plot_fig: Whether to generate plots
        save_fig: Whether to save plots to file
        fig_path: Path for saving figures
        save_csv: Whether to return CSV-formatted data
        figures: List of figure letters to generate (e.g., ['a', 'c', 'f']).
                If None, generates all figures (a-m)
    
    Returns:
        Tuple of (results DataFrame, colors DataFrame) if save_csv=True, else (None, None)
    """
    # If no specific figures requested, generate all
    if figures is None:
        figures = ['a', 'b', 'c', 'd', 'e1', 'e2', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    
    # Mapping of figure letters to derive functions
    figure_functions = {
        'a': derive_energy_mix,
        'b': derive_energy_mix_go,
        'c': derive_capacity_mix,
        'd': derive_capacity_mix_new,
        'e1': derive_storage_energy_capacity,
        'e2': derive_storage_power_capacity,
        'f': derive_total_system_cost,
        'g': derive_total_system_cost_new,
        'h': derive_go_market_revenue,
        'i': derive_marginal_price,
        'j': derive_co2_emissions,
        'k': derive_cfe_curtailment,
        'l': derive_cfe_utilization,
        'm': derive_co2_abatement_cost,
    }
    
    # Generate requested figures
    results_dict = {}
    colors_dict = {}
    
    for fig_letter in figures:
        if fig_letter in figure_functions:
            results, colors = figure_functions[fig_letter](df_networks, country, plot_fig, save_fig, fig_path, save_csv)
            results_dict[fig_letter] = results
            colors_dict[fig_letter] = colors
        else:
            print(f"Warning: Unknown figure '{fig_letter}' requested. Available: {list(figure_functions.keys())}")
    
    # Legacy variable names for backward compatibility
    results_csv_a, colors_csv_a = results_dict.get('a'), colors_dict.get('a')
    results_csv_b, colors_csv_b = results_dict.get('b'), colors_dict.get('b')
    results_csv_c, colors_csv_c = results_dict.get('c'), colors_dict.get('c')
    results_csv_d, colors_csv_d = results_dict.get('d'), colors_dict.get('d')
    results_csv_e1, colors_csv_e1 = results_dict.get('e1'), colors_dict.get('e1')
    results_csv_e2, colors_csv_e2 = results_dict.get('e2'), colors_dict.get('e2')
    results_csv_f, colors_csv_f = results_dict.get('f'), colors_dict.get('f')
    results_csv_g, colors_csv_g = results_dict.get('g'), colors_dict.get('g')
    results_csv_h, colors_csv_h = results_dict.get('h'), colors_dict.get('h')
    results_csv_i, colors_csv_i = results_dict.get('i'), colors_dict.get('i')
    results_csv_j, colors_csv_j = results_dict.get('j'), colors_dict.get('j')
    results_csv_k, colors_csv_k = results_dict.get('k'), colors_dict.get('k')
    results_csv_l, _ = results_dict.get('l'), colors_dict.get('l')
    results_csv_m, _ = results_dict.get('m'), colors_dict.get('m')
    
    if save_csv:
        # Collect only non-None results
        results_to_concat = [r for r in [
            results_csv_a, results_csv_b, results_csv_c, results_csv_d,
            results_csv_e1, results_csv_e2, results_csv_f, results_csv_g,
            results_csv_h, results_csv_i, results_csv_j, results_csv_k,
            results_csv_l, results_csv_m
        ] if r is not None]
        
        colors_to_concat = [c for c in [
            colors_csv_a, colors_csv_b, colors_csv_c, colors_csv_d,
            colors_csv_e1, colors_csv_e2, colors_csv_f, colors_csv_g,
            colors_csv_h, colors_csv_i, colors_csv_j, colors_csv_k
        ] if c is not None]
        
        if results_to_concat:
            results_csv_all = pd.concat(results_to_concat, axis=0)
            results_csv_all = results_csv_all[~results_csv_all.index.duplicated()]
        else:
            results_csv_all = None
        
        if colors_to_concat:
            colors_csv_all = pd.concat(colors_to_concat, axis=0)
            colors_csv_all = colors_csv_all[~colors_csv_all.index.duplicated()]
        else:
            colors_csv_all = None
        
        return results_csv_all, colors_csv_all
    else:
        return None, None

def save_df_csv(df_csv_all, path_csv, results):
    df_csv = pd.concat(df_csv_all.values(), axis=1)
    df_csv = df_csv.loc[:, ~df_csv.columns.duplicated()]

    os.makedirs(path_csv, exist_ok=True)
    csv_path = os.path.join(path_csv, f"{results}.csv")
    df_csv.to_csv(csv_path)
    print(f"{results} saved to {csv_path}")

def load_countries_from_config(config_path: str = "../config/config.go.yaml") -> List[str]:
    """Load countries list from configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        List of country codes with 'system' prepended
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        countries = config.get('countries', [])
        return ['system'] + countries
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using default countries")
        return ['system', 'DE']
    except Exception as e:
        print(f"Warning: Error loading config ({e}), using default countries")
        return ['system', 'DE']


def retrieve_networks(tutorial: bool = False) -> pd.DataFrame:
    """Retrieve and organize all networks from results directory.
    
    Automatically discovers all scenarios and years available. In tutorial mode,
    limits selection to 3 scenarios including baseline.
    
    Args:
        tutorial: If True, load maximum 3 scenarios (including baseline if present)
    
    Returns:
        DataFrame with MultiIndex (year, scenario) containing networks.
        Columns: ["network", "GoO"]
    """
    tqdm_kwargs = dict(
        ascii=False,
        unit=" Networks",
        desc="Processing networks ",
    )

    base_path = Path("../results/")
    files = list(base_path.rglob("*.nc"))
    
    # Filter files to match any .nc file with a 4-digit year and organize by scenario/year
    network_files = []
    selected_scenarios = set()
    
    # First pass: add baseline scenario if tutorial
    if tutorial:
        for fn in files:
            match = re.search(r'(\d{4})\.nc$', fn.name)
            if match and "baseline" in fn.parts[fn.parts.index("results") + 1].lower():
                year = int(match.group(1))
                scenario = fn.parts[fn.parts.index("results") + 1]
                selected_scenarios.add(scenario)
                network_files.append((year, scenario, fn))
    
    # Second pass: add remaining files
    for fn in files:
        match = re.search(r'(\d{4})\.nc$', fn.name)
        if match:
            year = int(match.group(1))
            scenario = fn.parts[fn.parts.index("results") + 1]
            
            if tutorial:
                if scenario not in selected_scenarios and len(selected_scenarios) < 3:
                    selected_scenarios.add(scenario)
                if scenario in selected_scenarios and (year, scenario, fn) not in network_files:
                    network_files.append((year, scenario, fn))
            else:
                network_files.append((year, scenario, fn))
    
    # Extract unique years and scenarios
    years = sorted(set(year for year, _, _ in network_files))
    scenarios = sorted(set(sc for _, sc, _ in network_files))
    
    # Build MultiIndex
    index = pd.MultiIndex.from_product(
        [years, scenarios],
        names=["year", "scenario"]
    )
    
    # Create empty DataFrame
    df_networks = pd.DataFrame(index=index, columns=["network", "GoO"])
    
    # Load networks
    for year, scenario, fn in tqdm(network_files, **tqdm_kwargs):
        try:
            n = pypsa.Network(fn)
            n = prepare_network(n)
            n.name = f"{scenario}-{year}"
            df_networks.loc[(year, scenario), "network"] = n
            
            m = strip_network_GoO(n)
            m.name = "GoO-" + m.name
            df_networks.loc[(year, scenario), "GoO"] = m
        except Exception as e:
            print(f"Error loading {scenario}-{year}: {e}")
            continue
    
    # Remove rows where network loading failed
    df_networks = df_networks.dropna()
    
    print(f"\nLoaded {len(df_networks)} networks:")
    print(f"  Years: {years}")
    print(f"  Scenarios: {scenarios}")
    
    return df_networks
    
def main(path_csv: Optional[str] = None, results: Optional[str] = None, 
         colors: Optional[str] = None, tutorial: bool = False) -> None:
    """Main function to generate time comparison analysis.
    
    Args:
        path_csv: Path to output directory for CSV files
        results: Name of the results CSV file (without extension)
        colors: Name of the colors CSV file (without extension)
        tutorial: If True, run in tutorial mode with limited scenarios
    """
    df_networks = retrieve_networks(tutorial=tutorial)
    
    # Load countries from config file
    countries = load_countries_from_config()
    print(f"Analyzing {len(countries)} countries: {countries[:5]}..." if len(countries) > 5 else f"Analyzing countries: {countries}")

    results_dict_all = {}
    colors_dict_all = {}
    iteration = 0
    if tutorial:
        countries = countries[:2]  # Limit to first 2 countries in tutorial mode
    for country in countries:
        iteration += 1
        print(f"\nProcessing country {iteration}/{len(countries)}: {_get_country_name(country)} ({country})")
        results_dict_all[country], colors_dict_all[country] = derive_all_figures(
            df_networks, country, plot_fig=False, save_fig=False, fig_path=None, save_csv=True
        )
        if iteration == 1:
            save_df_csv(colors_dict_all, path_csv, colors) # Save colors only for the system-level
          
    save_df_csv(results_dict_all, path_csv, results)

if __name__ == "__main__":
    
    import logging
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Generate procurement energy frontier for all scenarios and all year.")

    parser.add_argument(
        "--path_csv",
        type=str,
        default="figures/time_comparison/csv",
        help="Path to the output CSV files"
    )

    parser.add_argument(
        "--output_results",
        type=str,
        default="results",
        help="Name of the output CSV file with results (without extension)"
    )

    parser.add_argument(
        "--output_colors",
        type=str,
        default="colors",
        help="Name of the output CSV file with colors (without extension)"
    )

    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="If set, run only a few iterations for testing"
    )

    args = parser.parse_args()


    main(path_csv=args.path_csv, results=args.output_results, colors=args.output_colors, tutorial=args.tutorial)