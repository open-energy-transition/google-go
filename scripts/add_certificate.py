# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This script is part of a PyPSA-Eur workflow that adds Guarantee of Certificates components.
"""

import io
import logging
import os
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import requests

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)


def get_geo_center(shapes, output_index=None, x_offset=0, y_offset=0):
    gdf = gpd.read_file(shapes)

    # Rename 'name' column to 'country' if it exists
    if "name" in gdf.columns:
        gdf = gdf.rename(columns={"name": "country"})
    else:
        gdf["country"] = "EU"

    # Suppress the warning about geographic CRS
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
        gdf["center"] = gdf.centroid

    # Extract lon/lat
    gdf["x"] = gdf["center"].x + x_offset  # Longitude
    gdf["y"] = gdf["center"].y + y_offset  # Latitude

    # Handle output index
    if isinstance(output_index, str):
        gdf["new_index"] = [output_index + " " + c for c in gdf["country"]]
        gdf = gdf.set_index("new_index")
    else:
        gdf = gdf.set_index("country", drop=False)

    return gdf


def add_virtual_carriers(n, carriers):
    virtual_carrier_dict = {c: f"virtual {c}" for c in carriers}
    new_carriers = [
        orig
        for orig, virt in virtual_carrier_dict.items()
        if virt not in n.carriers.index
    ]

    if not new_carriers:
        logger.info(
            "No new virtual carriers added. All are already present in the network."
        )
        return

    existing = n.carriers.loc[n.carriers.index.intersection(new_carriers)].copy()
    missing = set(new_carriers) - set(existing.index)
    if missing:
        logger.warning(
            f"The following carriers were requested but not found in the network and will be skipped:\n - {'\n - '.join(missing)}"
        )

    if existing.empty:
        logger.warning("No valid carriers found to create virtual versions.")
        return

    virtual_carriers = existing.copy()
    virtual_carriers.index = "virtual " + virtual_carriers.index
    virtual_carriers["nice_name"] = "Virtual " + virtual_carriers["nice_name"]

    n.add(
        "Carrier",
        virtual_carriers.index,
        co2_emissions=0,
        color=virtual_carriers.color,
        nice_name=virtual_carriers.nice_name,
    )

    logger.info(
        "Virtual carriers defined:\n - " + "\n - ".join(virtual_carriers.index.tolist())
    )


def get_virtual_ppl_dataframe(n, certificate, planning_horizons):
    carriers = certificate["plant_carriers"]
    status = certificate["plant_status"]
    plant_group = certificate["plant_grouping"].copy()
    bus_group = certificate["bus_grouping"]
    max_lifetime_as_new = certificate["max_plant_lifetime_as_new"]
    new_threshold = int(planning_horizons) - max_lifetime_as_new

    # Filtering out links with generator carriers (coal, oil, nuclear)
    shared_carriers = set(n.generators.carrier) & set(n.links.carrier)

    # Collect all (country, carrier) pairs from generator-like components
    df = pd.DataFrame()

    for comp in ["generators", "links"]:
        df_comp = getattr(n, comp)
        df_comp = (
            df_comp.rename(columns={"bus1": "bus"})
            if "bus1" in df_comp.columns
            else df_comp
        )
        df_comp = df_comp[["bus", "carrier", "build_year"]].copy()
        df_comp["component"] = comp
        df_comp["country"] = df_comp["bus"].map(n.buses.country.to_dict())
        df_comp["status"] = np.where(
            df_comp["build_year"] >= new_threshold, "new", "existing"
        )
        included_carriers = (
            set(carriers) - shared_carriers if comp == "generators" else carriers
        )

        df_comp = df_comp[
            (df_comp["carrier"].isin(included_carriers))
            & ~(df_comp["country"].isin(["", np.nan]))
            & (df_comp["status"].isin(status))
        ]

        df = pd.concat([df, df_comp])

    df["ppl"] = df.index
    plant_group.extend(["component", "country"])
    df_vpp = pd.DataFrame(df.groupby(plant_group)["ppl"].apply(list))
    df_vpp["virtual_ppl"] = (
        "virtual "
        + df_vpp.index.get_level_values("country")
        + (
            " " + df_vpp.index.get_level_values("carrier")
            if "carrier" in plant_group
            else " " + df_vpp.index.get_level_values("component")
        )
        + (
            " " + df_vpp.index.get_level_values("status")
            if "status" in plant_group
            else ""
        )
    )

    df_vpp["virtual_carrier"] = (
        "virtual " + df_vpp.index.get_level_values("carrier")
        if "carrier" in plant_group
        else "GoO"
    )

    df_vpp["virtual_bus"] = (
        "GO Supply "
        + df_vpp.index.get_level_values("country")
        + (
            " " + df_vpp.index.get_level_values("carrier")
            if "carrier" in bus_group
            else ""
        )
        + (
            " " + df_vpp.index.get_level_values("status")
            if "status" in bus_group
            else ""
        )
    )

    df_vpp = df_vpp.reset_index().set_index("virtual_ppl")

    return df_vpp


def add_virtual_ppl(n, certificate, planning_horizons):
    df_vpp = get_virtual_ppl_dataframe(n, certificate, planning_horizons)

    df_bus = df_vpp.groupby(["virtual_bus", "country"]).size().reset_index("country")
    print(snakemake.input.country_shapes)
    df_node = get_geo_center(
        snakemake.input.country_shapes,
        x_offset=certificate["map"]["supply"][0],
        y_offset=certificate["map"]["supply"][1],
    )
    df_bus["x"] = df_bus["country"].map(df_node.x)
    df_bus["y"] = df_bus["country"].map(df_node.y)

    n.add(
        "Bus",
        df_bus.index,
        carrier="GoO",
        country=df_bus["country"],
        location=df_bus["country"],
        x=df_bus["x"],
        y=df_bus["y"],
    )

    n.add(
        "Generator",
        df_vpp.index,
        bus=df_vpp["virtual_bus"],
        carrier=df_vpp["virtual_carrier"],
        p_nom_extendable=True,
    )

    logger.info(
        "Virtual power plant buses added:\n - " + "\n - ".join(df_bus.index.tolist())
    )


def extract_AIB_statistics(output):
    """
    Downloads, processes, and exports electricity cancellation statistics from the AIB's
    July 2025 dataset.

    Parameters
    ----------
    output : str
        The file path where the resulting CSV file will be saved.

    Returns
    -------
    None
        The function saves the output to disk and does not return a value.
    """

    # Download the ZIP file
    url = "https://www.aib-net.org/sites/default/files/assets/facts/market%20information/statistics/activity%20statistics/AIB%20Statistics%20(July%202025).zip"
    target_file = "Transaction View - Data/Cancellations per Domain of Consumption by Reported Date-2025-08-26.xlsx"
    response = requests.get(url)
    response.raise_for_status()  # Ensure it downloaded successfully

    # Open the excel spreadsheet
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(target_file) as f:
            df = pd.read_excel(f)

    # Create a 'date' column as the last day of each year-month
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_number"], day=1))

    # Aggregate volume by date and domain
    df = df.groupby(["date", "domain_name"])["sum(volume)"].sum().unstack()

    # Combine Belgium regional domains into one
    belgium_regions = [
        "BEB - Belgium (Brussels)",
        "BEF - Belgium (Flanders)",
        "BEW - Belgium (Wallonia)",
    ]
    if all(col in df.columns for col in belgium_regions):
        df["BE - Belgium"] = df[belgium_regions].sum(axis=1)
        df.drop(columns=belgium_regions, inplace=True)

    df.columns = [col[:2] for col in df.columns]

    full_index = pd.to_datetime(
        [
            f"{year}-{month}-01"
            for year in df.index.year.unique()
            for month in range(1, 13)
        ]
    )

    df = df.reindex(full_index)

    # Group by year
    yearly_sums = df.groupby(df.index.year).sum()

    # Build the set of missing years per column
    missing_years = df.apply(
        lambda col: (
            set(col.index.year[col.isna()])  # Years with NaNs
            | set(yearly_sums.index[yearly_sums[col.name] == 0])  # Years with zero sum
        )
    )

    for col in df.columns:
        df[col] = df[col].mask(df.index.year.isin(missing_years.get(col, set())))

    # Final DataFrame
    df.to_csv(output)


def get_load_demand(n, profile="", set_logger=True):
    """
    Extracts and optionally processes the electricity load demand from a PyPSA network object.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network object containing time-series data for loads and buses.

    profile : str, optional (default = "")
        Determines how the load demand should be modified:
        - "total": Returns the raw total load demand per country (no modification).
        - "total_daily_avg": Averages the load per day and forward-fills to match the original resolution.
        - "baseload": Sets all load values to the overall mean (flat profile).

    set_logger : bool, optional (default = True)
        Whether to log which profile is selected using the global logger.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with timestamps as the index and countries as columns, containing
        the processed electricity load demand.
    """
    elec_load = n.loads[n.loads.carrier == "electricity"].index
    load = n.loads_t.p_set[elec_load].copy()
    load.columns = load.columns.map(n.loads.bus).map(n.buses.country)
    load = load.T.groupby(level=0).sum().T

    if profile == "total_daily_avg":
        load = load.resample("d").mean().reindex(n.snapshots, method="ffill")
    elif profile == "total":
        pass  # No modification
    elif profile == "baseload":
        load[:] = load.mean()
    else:
        valid_profiles = ["total_daily_avg", "baseload", "total"]
        raise ValueError(
            f"Invalid profile: '{profile}'. Must be one of {valid_profiles}."
        )

    if set_logger:
        logger.info(f"{profile} profile for go demand is selected")

    return load


def get_go_background_demand(n, aib_filepath, profile=""):
    """
    Generates a background Guarantee of Origin (GO) electricity demand profile
    for each country in a PyPSA network based on historical AIB statistics.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network object containing time-series snapshots and bus information.

    profile : str, optional (default = "")
        Defines the method for shaping the background GO demand profile:
        - "historical": Uses the historical monthly distribution of GO cancellations from AIB data.
        - all profiles from get_load_demand()

    Returns
    -------
    pandas.DataFrame
        A DataFrame with snapshots as index and countries as columns, representing
        the background GO demand per country, scaled to match historical GO data.
    """

    # Download or extract data if file is missing
    if not os.path.isfile(aib_filepath):
        logger.info(
            f"{aib_filepath} not found. Extracting and Saving AIB statistics..."
        )
        extract_AIB_statistics(aib_filepath)

    # Load the GO data
    go_data = pd.read_csv(aib_filepath, index_col=0, parse_dates=True)
    valid_countries = sorted(list(set(go_data.columns) & set(n.buses.country)))
    go_data = go_data[valid_countries]

    yearly_totals = go_data.groupby(go_data.index.year).sum().replace(0, np.nan)
    load = pd.DataFrame(index=n.snapshots, columns=go_data.columns)

    if profile == "historical":
        df_norm = go_data.apply(
            lambda row: row / yearly_totals.loc[row.name.year], axis=1
        )

        monthly_mean = df_norm.groupby(df_norm.index.month).mean()
        monthly_profile = monthly_mean.div(monthly_mean.sum())
        snapshot_counts = n.snapshots.to_series().groupby(n.snapshots.month).size()

        # Note: This method means that SEG time aggregations cannot be done
        for month in n.snapshots.month.unique():
            mask = load.index.month == month
            load.loc[mask] = monthly_profile.loc[month].values / snapshot_counts[month]
    elif profile in ["total_daily_avg", "total", "baseload"]:
        load = get_load_demand(n, profile=profile, set_logger=False)
        load = load[valid_countries]
        load = load / load.sum()
    else:
        valid_profiles = ["historical", "total_daily_avg", "total", "baseload"]
        raise ValueError(
            f"Invalid profile: '{profile}'. Must be one of {valid_profiles}."
        )

    logger.info(f"{profile} profile for background go demand is selected")

    scaling = n.snapshot_weightings.objective.sum() / len(
        n.snapshot_weightings.objective
    )
    demand = getattr(yearly_totals, "max")() * 1 / scaling
    load *= demand

    return load


def add_demand(n, load, cert_demand, cert_map, name):
    energy_matching = cert_demand["energy_matching"] / 100
    hourly_matching = cert_demand["hourly_matching"] / 100

    df_demand = get_geo_center(
        snakemake.input.country_shapes,
        "GO Demand",
        x_offset=cert_map[0],
        y_offset=cert_map[1],
    )

    valid_countries = sorted(list(set(df_demand.country) & set(load.columns)))
    df_demand = df_demand[df_demand.country.isin(valid_countries)]

    if name:
        df_demand.index = df_demand.index.astype(str) + " " + name

    n.add(
        "Bus",
        df_demand.index,
        carrier="GoO",
        country=df_demand["country"],
        location=df_demand["country"],
        x=df_demand["x"],
        y=df_demand["y"],
    )

    df_load = load * energy_matching
    df_load.columns = [f"GO Demand {c}" for c in df_load.columns]
    df_load = df_load.astype("float64")

    if name:
        df_load.columns = df_load.columns.astype(str) + " " + name
    n.add(
        "Load",
        df_load.columns,
        p_set=df_load,
        carrier="GoO",
        bus=df_load.columns,
    )
    logger.info("GO Demand added:\n - " + "\n - ".join(df_load.columns.tolist()))

    # 247-go is defined as hourly_matching = 100%, where vol-match is defined as hourly_matching = 0%
    if hourly_matching != 1:
        df_buffer = pd.DataFrame(n.snapshot_weightings.objective @ df_load)
        df_buffer["e_nom"] = df_buffer["objective"] * (1 - hourly_matching)
        df_buffer["bus"] = df_buffer.index
        df_buffer.rename(index=lambda i: i.replace("Demand", "Buffer"), inplace=True)

        n.add(
            "Store",
            df_buffer.index,
            bus=df_buffer["bus"],
            carrier="GoO",
            e_nom=df_buffer["e_nom"],
            e_min_pu=-1,
            e_cyclic=True,
        )

        logger.info(
            f"GO Buffer added proportion to {(1 - hourly_matching) * 100}% of GO demand"
        )


def add_go_market(n, cert_demand, cert_map, name):
    supply_filter = cert_demand["supply_filter"]
    scope = cert_demand["scope"]
    market_name = cert_demand["market_name"]

    regex_supply = "(?=.*GO Supply)"
    regex_demand = f"(?=.*GO Demand)(?=.*{name})"
    filtered = {k: v for k, v in supply_filter.items() if isinstance(v, list) and v}

    if filtered:
        for value in filtered.values():
            pattern = f"(?=.*\\b(?:{'|'.join(value)})\\b)"
            regex_supply += pattern

    combined_regex = f"(?:{regex_supply})|(?:{regex_demand})"

    # Apply filter
    df_link = n.buses[n.buses.index.str.contains(combined_regex)][
        ["carrier", "country"]
    ].copy()

    # Assign bus0 and bus1 based on the type
    if scope == "national":
        shape = snakemake.input.country_shapes
        df_link["market_bus"] = ["GO Market " + c for c in df_link.country]
        country_list = df_link.country.unique()
    else:
        shape = snakemake.input.europe_shape
        df_link["market_bus"] = "GO Market EU"
        country_list = ["EU"]

    map_select = (
        cert_map["market " + market_name] if market_name else cert_map["market"]
    )

    df_market = get_geo_center(
        shape, "GO Market", x_offset=map_select[0], y_offset=map_select[1]
    )

    valid_countries = sorted(list(set(df_market.country) & set(country_list)))
    df_market = df_market[df_market.country.isin(valid_countries)]

    if market_name:
        df_market.index = df_market.index.astype(str) + " " + market_name
        df_link["market_bus"] = df_link["market_bus"].astype(str) + " " + market_name

    n.add(
        "Bus",
        df_market.index,
        carrier="GoO",
        country=df_market["country"],
        location=df_market["country"],
        x=df_market["x"],
        y=df_market["y"],
    )

    logger.info("GO Market added:\n - " + "\n - ".join(df_market.index.tolist()))

    supply_mask = df_link.index.str.contains("GO Supply")
    demand_mask = df_link.index.str.contains("GO Demand")
    df_link["bus0"] = df_link.index.where(supply_mask, df_link["market_bus"])
    df_link["bus1"] = df_link.index.where(demand_mask, df_link["market_bus"])

    # For df_link with the index containing "GO Supply", if the same name already occurs in n.links.index but it compose of different buses
    df_link.index = [
        f"{idx} to {market}" if m else idx
        for idx, market, m in zip(df_link.index, df_link["market_bus"], supply_mask)
    ]

    # Drop already existed links
    already_exist_links = list(set(n.links.index) & set(df_link.index))
    df_link = df_link.drop(already_exist_links)

    n.add(
        "Link",
        df_link.index,
        bus0=df_link["bus0"],
        bus1=df_link["bus1"],
        carrier="GoO",
        p_nom_extendable=True,
        reversed=False,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_certificate",
            run="",
            opts="",
            clusters="50",
            configfiles="config/config.go.yaml",
            ll="",
            sector_opts="",
            planning_horizons="2030",
        )
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)

    certificate = snakemake.params.certificate
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    # Add Certificates
    n.add(
        "Carrier",
        "GoO",
        co2_emissions=0,
        color="#98c41c",
        nice_name="Guarantee of Origin",
    )
    if "carrier" in certificate["plant_grouping"]:
        carriers = certificate["plant_carriers"]
        add_virtual_carriers(n, carriers)

    # Add national supply bus
    add_virtual_ppl(n, certificate, planning_horizons)

    if certificate["background_demand"]["enable"]:
        load_background = get_go_background_demand(
            n,
            certificate["aib_filepath"],
            profile=certificate["background_demand"]["go_profile"],
        )
        add_demand(
            n,
            load_background,
            certificate["background_demand"],
            certificate["map"]["background_demand"],
            "Background",
        )
        add_go_market(
            n, certificate["background_demand"], certificate["map"], "Background"
        )
    if certificate["new_demand"]["enable"]:
        load_new = get_load_demand(
            n,
            profile=certificate["new_demand"]["go_profile"],
        )
        add_demand(
            n,
            load_new,
            certificate["new_demand"],
            certificate["map"]["new_demand"],
            "New",
        )
        add_go_market(n, certificate["new_demand"], certificate["map"], "New")

    n.export_to_netcdf(snakemake.output[0])
