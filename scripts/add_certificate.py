# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This script is part of a PyPSA-Eur workflow that adds Guarantee of Certificates components.
"""

import logging

import numpy as np
import geopandas as gpd
import pandas as pd
import pypsa
import warnings

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)


def get_largest_country_node(n, output_index=None):
    df = pd.DataFrame(n.snapshot_weightings.objective @ n.loads_t.p_set)
    df["country"] = df.index.map(n.buses.country)

    df = df.dropna(subset=["country"])
    max_indices = df.groupby("country")["objective"].idxmax()
    df = df.loc[max_indices]

    df["x"] = df.index.map(n.buses.x)
    df["y"] = df.index.map(n.buses.y)

    if isinstance(output_index, str):
        df["new_index"] = [output_index + " " + c for c in df.country]
        df = df.set_index("new_index")

    return df


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
        + (" " + df_vpp.index.get_level_values("carrier") if "carrier" in plant_group else " " + df_vpp.index.get_level_values("component"))
        + (" " + df_vpp.index.get_level_values("status") if "status" in plant_group else "")
    )

    
    df_vpp["virtual_carrier"] = (
        "virtual " + df_vpp.index.get_level_values("carrier") if "carrier" in plant_group else "GoO"
    )

    df_vpp["virtual_bus"] = (
        "GO Supply "
        + df_vpp.index.get_level_values("country")
        + (" " + df_vpp.index.get_level_values("carrier") if "carrier" in bus_group else "")
        + (" " + df_vpp.index.get_level_values("status") if "status" in bus_group else "")
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
        y_offset=certificate["map"]["supply"][1]
    )
    #df_node = get_largest_country_node(n).set_index("country")
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


def add_virtual_demand(n, certificate):
    energy_matching = certificate["energy_matching"] / 100
    hourly_matching = certificate["hourly_matching"] / 100
    df_demand = get_geo_center(
        snakemake.input.country_shapes,
        "GO Demand",
        x_offset=certificate["map"]["demand"][0], 
        y_offset=certificate["map"]["demand"][1]
    )
    # df_demand = get_largest_country_node(n, "GO Demand")

    n.add(
        "Bus",
        df_demand.index,
        carrier="GoO",
        country=df_demand["country"],
        location=df_demand["country"],
        x=df_demand["x"],
        y=df_demand["y"],
    )

    # This is the simplified version of GO profile
    df_load = n.loads_t.p_set.copy()
    go_list = n.buses.loc[df_demand.index, "location"]
    go_list = pd.Series(go_list.index, index=go_list.values)
    df_load.columns = df_load.columns.map(n.loads.bus).map(n.buses.country).map(go_list)
    df_load = df_load.T.groupby(df_load.columns).sum().T

    df_load *= energy_matching

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


def add_go_market(n, certificate):
    scope = certificate["scope"]
    # Filter all GO Supply and GO Demand buses
    df_link = n.buses[n.buses.index.str.contains("GO Supply|GO Demand")][
        ["carrier", "country"]
    ].copy()

    # Assign bus0 and bus1 based on the type
    if scope == "national":
        shape = snakemake.input.country_shapes
        df_link["market_bus"] = ["GO Market " + c for c in df_link.country]
    else:
        shape = snakemake.input.europe_shape
        df_link["market_bus"] = "GO Market EU"
    
    df_market = get_geo_center(
        shape,
        "GO Market",
        x_offset=certificate["map"]["market"][0], 
        y_offset=certificate["map"]["market"][1]
    )

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

    df_link["bus0"] = df_link.index.where(
        df_link.index.str.contains("GO Supply"), df_link["market_bus"]
    )
    df_link["bus1"] = df_link.index.where(
        df_link.index.str.contains("GO Demand"), df_link["market_bus"]
    )

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
            clusters="39",
            configfiles="config/config.go.yaml",
            ll="v1.0",
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

    # Add national demand bus
    add_virtual_demand(n, certificate)

    # Add GO Market as an intermediary
    add_go_market(n, certificate)

    n.export_to_netcdf(snakemake.output[0])
