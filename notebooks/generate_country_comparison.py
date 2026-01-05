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
import cartopy.crs as ccrs
from pypsa.plot.maps.static import add_legend_circles, add_legend_patches

# Constants for unit conversions
MWH_TO_TWH = 1e6
EUR_TO_BILLION_EUR = 1e9
MW_TO_GW = 1e3
TONS_TO_MEGATONS = 1e6

# Constants for figure sizes
FIGSIZE_BAR_COUNTRIES = (12, 6)  # Default size for country comparison bar plots

# Plot formats for each figure type
plot_formats = {
    "Energy mix": {"letter": "(a)", "y_label": "Net generation (TWh)"},
    "Energy mix - GO Market": {"letter": "(b)", "y_label": "Net generation (TWh-GoO)"},
    "Capacity mix": {"letter": "(c)", "y_label": "Capacity (GW)"},
    "Capacity mix - new technologies (proxy of GO Market)": {"letter": "(d)", "y_label": "Capacity (GW)"},
    "Storage in GO Market - Energy capacity": {"letter": "(e1)", "y_label": "Energy capacity (GWh)"},
    "Storage in GO Market - Power capacity": {"letter": "(e2)", "y_label": "Power capacity (GW)"},
    "Total system cost - new technologies (proxy of GO Market)": {"letter": "(g)", "y_label": "Total system cost (b€)"},
    "GO Market revenue by technology": {"letter": "(h)", "y_label": "Market size (b€)"},
    "Marginal price of GoO consumers": {"letter": "(i)", "y_label": "Marginal price (€/MWh)"},
    "CO2 emissions": {"letter": "(j)", "y_label": "CO2 emissions (MtCO2)"},
    "CFE curtailment": {"letter": "(k)", "y_label": "Curtailment (TWh)"},
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

### Functions to plot figures
def plot_bar(df, colors, ylabel=None, title=None, figsize=(12, 6), vert_lines=True, ylim=False):
    # Create figure and axis with adjustable size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean 'virtual ' prefix from index names
    if df.index.name is not "year":
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
              title='Carrier' if df.index.name is not "year" else 'Year')

    # Add total values above each bar
    bar_totals = df.T.sum(axis=1)  # totals per bar
    # Calculate bar height considering both positive and negative values
    bar_height_pos = df.clip(lower=0).T.sum(axis=1)
    bar_height_neg = df.clip(upper=0).T.sum(axis=1)
    
    if ylim:
        ax.set_ylim(ylim)
    
    for i, total in enumerate(bar_totals):
        color = "black"
        
        # Position text above positive part or below negative part
        if total >= 0:
            text_position = bar_height_pos.iloc[i]
            valign = 'bottom'
        else:
            text_position = bar_height_neg.iloc[i]
            valign = 'top'
        
        if ylim and ylim[1] is not 100:
            if bar_height_pos.iloc[i] > ylim[1]:
                text_position = ylim[1]
                color = "red"
        
        ax.text(
            i,                              # x position
            text_position,                  # y position
            "{:.0f}".format(total),         # label
            ha='center', va=valign,
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
    ax.set_xlabel("")

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
        if df.index.name == "year":
            ax.set_title(title, pad=20)

    ax.grid(axis='y')

    return ax

def add_legends(df, ax, colors, n):
    
    # --- Carrier patches legend ---
    legend_colors = colors.values
    legend_labels = colors.index
    
    legend_kw = {
        "loc": "upper left",
        "bbox_to_anchor": (1.0, 1.0),
        "handletextpad": 0.2,
        "title": r"Carrier",
        "alignment": "center",
        "ncol": 1
    }
    
    add_legend_patches(
        ax,
        legend_colors,
        legend_labels,
        legend_kw=legend_kw
    )

    df_bus = df.groupby("bus").sum()
    df_bus = df_bus[df_bus > 0]
    
    for bus, size in df_bus.items():
        # Get coordinates
        lon = n.buses.at[bus, "x"]
        lat = n.buses.at[bus, "y"]
        
        # Transform coordinates into the plot projection
        x, y = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())

        ax.text(
            x, y,
            f"{size:.1f}",
            ha="center", va="center",
            fontsize=8,
            color="black",
            zorder=5
        )


def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def plot_map(df, df_n, colors, unit=None, title=None, bus_size_factor=1e1,save=False, path=None):
    
    colors = colors.groupby("carrier").first()
    proj = load_projection({})

    for i in df_n.index:
        #print(f"-------------------------------------Analyzing {i}")
        n = df_n.loc[i]

        df_scenario = df[i]
        
        bus_size = df_scenario / bus_size_factor

        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})
        n.plot.map(ax=ax, bus_size=bus_size, bus_colors = colors, line_widths=0, link_widths=0, boundaries = [-11, 30, 34, 71])

        add_legends(df_scenario, ax, colors, n)

        if title is not None:
            fig_title = f"{title} {i}"
            ax.set_title(fig_title)

        if unit is not None:
            fig.text(
                0.32, 0.5,
                f"{unit}",
                rotation=90,
                va='center',
                ha='center',
            )
        if save:
            ax.figure.savefig(f"{path}/{fig_title}.png", dpi=300, bbox_inches="tight")

### Helper functions
def _save_plot(fig, save_fig: bool, fig_path: Optional[str], title: str) -> None:
    """Save plot to file if requested."""
    if save_fig and fig_path:
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(f"{fig_path}/{title}.png", dpi=300, bbox_inches="tight")

def _calculate_ylim(df: pd.DataFrame, margin: float = 0.15) -> Tuple[float, float]:
    """Calculate appropriate y-axis limits with margin.
    
    Args:
        df: DataFrame with data values
        margin: Percentage margin to add above max value (default 15%)
        
    Returns:
        Tuple of (ymin, ymax)
    """
    # For stacked bars with positive and negative values, we need to calculate
    # the max of positive stacks and min of negative stacks separately
    if len(df.shape) == 2:
        # Sum positive and negative values separately for each column
        positive_stack = df.clip(lower=0).sum(axis=0)
        negative_stack = df.clip(upper=0).sum(axis=0)
        
        max_val = positive_stack.max()
        min_val = negative_stack.min()
    else:
        max_val = df.max()
        min_val = df.min()
    
    # Add margin
    if max_val > 0:
        ymax = max_val * (1 + margin)
    else:
        ymax = max_val * (1 - margin) if max_val < 0 else 0
    
    if min_val < 0:
        ymin = min_val * (1 + margin)
    else:
        ymin = 0
    
    return (ymin, ymax)

### Functions to derive figures
def derive_energy_mix(df_networks: pd.DataFrame, 
                     plot_fig: bool = False, save_fig: bool = False, 
                     fig_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive energy mix statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Energy mix"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "energy_balance", groupby=["country1","carrier","bus_carrier"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        df.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & ~df.index.get_level_values("carrier").isin(["AC", "DC", "electricity", "low voltage",
                                                       "electricity distribution grid", "BEV charger",
                                                       "home battery charger", "home battery discharger"])
    ]
    
    # Groupby with sum_except_color to preserve colors
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MWH_TO_TWH
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_energy_mix_go(df_networks: pd.DataFrame, 
                        plot_fig: bool = False, save_fig: bool = False, 
                        fig_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive GO Market energy mix statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Energy mix - GO Market"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df_GoO = df_networks[df_networks.index.get_level_values("scenario") != "baseline"]
    df, colors = get_stats_all(df_GoO["GoO"], "energy_balance", groupby=["country1","carrier","bus_carrier"])
    df["color"] = colors
    
    # Filter dataframe - for GoO network exclude GoO carrier
    df = df[df.index.get_level_values("carrier") != "GoO"]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MWH_TO_TWH
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_capacity_mix(df_networks: pd.DataFrame, 
                       plot_fig: bool = False, save_fig: bool = False, 
                       fig_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive capacity mix statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Capacity mix"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "optimal_capacity", groupby=["country1","carrier","bus_carrier"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        ~df.index.get_level_values("bus_carrier").isin(["GoO","co2","co2 stored"])
        & ~df.index.get_level_values("carrier").isin(["AC","DC","electricity distribution grid","low voltage"] + list(grouping_storage.keys()))
        & ~df.index.get_level_values("component").isin(["Store","StorageUnit"])
    ]
    # Remove specific coal and gas generators
    df = df.loc[~df.index.isin([("Generator","coal","coal"),("Generator","gas","gas")])] 
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MW_TO_GW
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_capacity_mix_new(df_networks: pd.DataFrame, 
                           plot_fig: bool = False, save_fig: bool = False, 
                           fig_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive capacity mix for new technologies - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Capacity mix - new technologies (proxy of GO Market)"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "optimal_capacity", groupby=["country1","carrier","bus_carrier","build_year"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        ~df.index.get_level_values("bus_carrier").isin(["GoO","co2","co2 stored"])
        & ~df.index.get_level_values("carrier").isin(["AC","DC","electricity distribution grid","low voltage"] + list(grouping_storage.keys()))
        & ~df.index.get_level_values("component").isin(["Store","StorageUnit"])
        & (df.index.get_level_values("build_year") > 2020)
    ]
    # Remove specific coal and gas generators
    df = df.loc[~df.index.isin([("Generator","coal","coal"),("Generator","gas","gas")])] 
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MW_TO_GW
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_storage_energy_capacity(df_networks: pd.DataFrame, 
                                   plot_fig: bool = False, save_fig: bool = False, 
                                   fig_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive storage energy capacity statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Storage in GO Market - Energy capacity"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "optimal_capacity", groupby=["country1","carrier"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        df.index.get_level_values("component").isin(["Store"])
        & df.index.get_level_values("carrier").isin(list(grouping_storage.keys()))
        & ~df.index.get_level_values("carrier").isin(['EV battery','home battery'])
    ]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MW_TO_GW
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_storage_power_capacity(df_networks: pd.DataFrame, 
                                  plot_fig: bool = False, save_fig: bool = False, 
                                  fig_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive storage power capacity statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Storage in GO Market - Power capacity"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "optimal_capacity", groupby=["country1","carrier"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        df.index.get_level_values("component").isin(["Link"])
        & df.index.get_level_values("carrier").isin(list(grouping_storage.keys()))
        & ~df.index.get_level_values("carrier").isin(['BEV charger','li-ion discharger','iron-air discharger',
                                                        'home battery charger','home battery discharger'])
    ]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MW_TO_GW
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_total_system_cost_new(df_networks: pd.DataFrame, 
                                plot_fig: bool = False, save_fig: bool = False, 
                                fig_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive total system cost for new technologies - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Total system cost - new technologies (proxy of GO Market)"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "system_cost", groupby=["country1","carrier","build_year"])
    df["color"] = colors
    
    # Filter by build_year > 2020
    df = df[df.index.get_level_values("build_year") > 2020]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / EUR_TO_BILLION_EUR
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_go_market_revenue(df_networks: pd.DataFrame, 
                            plot_fig: bool = False, save_fig: bool = False, 
                            fig_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive GO Market revenue by technology - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "GO Market revenue by technology"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df_GoO = df_networks[df_networks.index.get_level_values("scenario") != "baseline"]
    df, colors = get_stats_all(df_GoO["GoO"], "revenue", groupby=["country1","carrier"])
    df["color"] = colors
    
    # Filter dataframe
    df = df[df.index.get_level_values("carrier") != "GoO"]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / EUR_TO_BILLION_EUR
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_marginal_price(df_networks: pd.DataFrame, 
                         plot_fig: bool = False, save_fig: bool = False, 
                         fig_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive marginal price of GoO consumers - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "Marginal price of GoO consumers"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    # Get prices from all networks
    df, _ = get_stats_prices(df_networks["network"])
    
    keywords = ["GO Demand", "low voltage"]
    pattern = "|".join(keywords)
    
    # Get a reference network for mapping - prefer "match" scenarios
    match_scenarios = [idx for idx in df_networks.index if "match" in idx[1].lower()]
    if match_scenarios:
        year, sc = match_scenarios[0]
    else:
        year, sc = df_networks.index[0]
    n_ref = df_networks.loc[(year, sc), "network"]
    
    # Filter by keywords
    df = df[df.index.str.extract(f"({pattern})", expand=False).isin(keywords)]
    
    # Map bus names to country and carrier, creating MultiIndex
    df["country1"] = df.index.map(n_ref.buses.country)
    df["carrier"] = df.index.map(n_ref.buses.carrier)
    
    # Set MultiIndex
    df = df.set_index(["country1", "carrier"], append=True)
    df = df.droplevel(0)  # Remove bus names
    
    # Add color column manually from carrier colors
    df["color"] = df.index.get_level_values("carrier").map(n_ref.carriers.color)
    
    # Groupby with sum_except_color
    df = df.groupby(["country1", "carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        
        # Map colors from colors_by_carrier
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_co2_emissions(df_networks: pd.DataFrame, 
                        plot_fig: bool = False, save_fig: bool = False, 
                        fig_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive CO2 emissions statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "CO2 emissions"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    # Use "name" groupby (like time_comparison for Links)
    df, colors = get_stats_all(df_networks["network"], "energy_balance", groupby=["carrier","bus_carrier","name"])
    df["color"] = colors
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        df.index.get_level_values("bus_carrier").isin(["co2"])
        & df.index.get_level_values("component").isin(["Link"])
    ]
    
    # Get a reference network to map bus names to countries
    year, sc = df_networks.index[0]
    n_ref = df_networks.loc[(year, sc), "network"]
    
    # Map name to country via bus1 (Links have bus1 which we can map to country)
    df["country1"] = df.index.get_level_values("name").map(n_ref.links.bus1).map(n_ref.buses.country)
    df = df.set_index("country1", append=True)
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1")
        df_country = df_country / TONS_TO_MEGATONS
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_cfe_curtailment(df_networks: pd.DataFrame, 
                          plot_fig: bool = False, save_fig: bool = False, 
                          fig_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """Derive CFE curtailment statistics - Bar plot by country.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figsize: Figure size as (width, height) tuple
    """
    results = "CFE curtailment"
    letter = plot_formats[results]['letter']
    y_label = plot_formats[results]['y_label']
    title = f"{letter} {results}"
    
    df, colors = get_stats_all(df_networks["network"], "curtailment", groupby=["country1","carrier","bus_carrier"])
    df["color"] = colors
    
    # Define vres_carriers list
    vres_carriers = [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]
    
    # Filter dataframe - same filters as time_comparison
    df = df[
        df.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
        & df.index.get_level_values("carrier").isin(vres_carriers)
    ]
    
    # Groupby with sum_except_color
    df = df.groupby(["country1","carrier"]).apply(sum_except_color)
    df = df.rename(index=grouping_storage).groupby(["country1","carrier"]).apply(sum_except_color)
    
    # Filter out empty country names
    df = df[df.index.get_level_values("country1") != ""]
    
    # Extract colors after groupby
    colors_by_carrier = df["color"].groupby(level="carrier").first()
    df = df.drop("color", axis=1)
    
    for i in df.columns:
        print(f"-------------------------------------Analyzing {i}")
        title_fig = f"{title} {i}"
        df_country = df[i]
        df_country = df_country.unstack("country1").groupby("carrier").sum()
        df_country = df_country / MWH_TO_TWH
        
        colors_plot = df_country.index.map(colors_by_carrier)
        
        ylim = _calculate_ylim(df_country)
        ax = plot_bar(
            df_country, 
            colors_plot, 
            ylabel=y_label, 
            title=title_fig,
            figsize=figsize,
            ylim=ylim,
            vert_lines=False
        )
        
        if plot_fig:
            plt.show()
        else:
            plt.close()
            
        if save_fig:
            _save_plot(ax.figure, save_fig, fig_path, title_fig)


def derive_energy_mix_go_map(df_networks: pd.DataFrame, 
                             plot_fig: bool = False, save_fig: bool = False, 
                             fig_path: Optional[str] = None) -> None:
    """Derive GO Market energy mix - Map visualization.
    
    Args:
        df_networks: DataFrame containing network data indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
    """
    results = "Energy mix - GO Market"
    letter = plot_formats[results]['letter']
    title = f"{letter} {results} - Map"
    unit = plot_formats[results]['y_label']
    
    df_GoO = df_networks[df_networks.index.get_level_values("scenario") != "baseline"]
    df, colors = get_stats_all(df_GoO["GoO"], "energy_balance", groupby=["bus","carrier"])
    df["color"] = colors
    df = df[df.index.get_level_values("carrier") != "GoO"].groupby(["bus","carrier"]).apply(sum_except_color)

    # Clean 'virtual ' prefix from index names
    df = clean_virtual_names(df.copy())

    colors = df["color"]
    df = df.drop("color", axis=1)
    df = df / MWH_TO_TWH

    plot_map(df, df_GoO["GoO"], colors, unit=unit, title=title, save=save_fig, path=fig_path)
    
    if plot_fig:
        plt.show()
    else:
        plt.close()


def derive_all_figures(df_networks: pd.DataFrame,
                       plot_fig: bool = False, save_fig: bool = False,
                       fig_path: Optional[str] = None, figures: Optional[List[str]] = None,
                       figsize_bar: Tuple[int, int] = (12, 6)) -> None:
    """
    Generate all country comparison figures.
    
    Args:
        df_networks: DataFrame containing networks indexed by (year, scenario)
        plot_fig: Whether to display plots
        save_fig: Whether to save plots to files
        fig_path: Path to save figures
        figures: List of figure letters to generate (e.g., ['a', 'b', 'c'])
        figsize_bar: Figure size for bar plots
    """
    
    figures_map = {
        'a': derive_energy_mix,
        'b': derive_energy_mix_go,
        'c': derive_capacity_mix,
        'd': derive_capacity_mix_new,
        'e1': derive_storage_energy_capacity,
        'e2': derive_storage_power_capacity,
        'g': derive_total_system_cost_new,
        'h': derive_go_market_revenue,
        'i': derive_marginal_price,
        'j': derive_co2_emissions,
        'k': derive_cfe_curtailment,
        'b_map': derive_energy_mix_go_map,
    }
    
    # If no specific figures requested, generate all
    if figures is None:
        figures = list(figures_map.keys())
    
    for fig_letter in figures:
        if fig_letter not in figures_map:
            print(f"Warning: Figure '{fig_letter}' not recognized, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Generating figure {fig_letter}")
        print(f"{'='*60}")
        
        if fig_letter == 'b_map':
            figures_map[fig_letter](df_networks, plot_fig=plot_fig, save_fig=save_fig, fig_path=fig_path)
        else:
            figures_map[fig_letter](df_networks, plot_fig=plot_fig, save_fig=save_fig, 
                                   fig_path=fig_path, figsize=figsize_bar)
