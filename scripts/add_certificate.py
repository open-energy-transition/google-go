# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This script is part of a PyPSA-Eur workflow that adds Guarantee of Certificates components.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)


def get_virtual_ppl_dataframe(n, certificate, planning_horizons):
    carriers = certificate["plant_carriers"]
    status = certificate["plant_status"]
    grouping = certificate["plant_grouping"].copy()
    max_lifetime_as_new = certificate["max_plant_lifetime_as_new"]
    new_threshold = int(planning_horizons) - max_lifetime_as_new

    grouping.append("component")

    if certificate["scope"] == "national" and "country" not in grouping:
        grouping.append("country")

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
    df = pd.DataFrame(df.groupby(grouping)["ppl"].apply(list))
    df["virtual_ppl"] = (
        "virtual"
        + (" " + df.index.get_level_values("country") if "country" in grouping else "")
        + (" " + df.index.get_level_values("carrier") if "carrier" in grouping else "")
        + (" " + df.index.get_level_values("status") if "status" in grouping else "")
    )
    df = df.reset_index().set_index("virtual_ppl")

    return df


def create_national_go_market(n):
    df = pd.DataFrame(n.snapshot_weightings.objective @ n.loads_t.p_set)
    df["country"] = df.index.map(n.buses.country)

    df = df.dropna(subset=["country"])
    max_indices = df.groupby("country")["objective"].idxmax()
    df = df.loc[max_indices]

    df["x"] = df.index.map(n.buses.x)
    df["y"] = df.index.map(n.buses.y)
    df["bus"] = "GO Market " + df["country"]
    df["demand"] = "GO Demand " + df["country"]

    df_bus = df.set_index("bus")
    df_demand = df.set_index("demand")

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
        "Store",
        df_demand.index,
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="GoO",
        bus=df_demand["bus"],
    )

    logger.info(f"New Buses added:\n - {'\n - '.join(set(df_bus.index))}")
    logger.info(f"New Stores added:\n - {'\n - '.join(set(df_demand.index))}")


# def add_virtual_carriers(n, carriers):
#     virtual_carrier_dict = {c: f"virtual {c}" for c in carriers}
#     new_carriers = [
#         orig
#         for orig, virt in virtual_carrier_dict.items()
#         if virt not in n.carriers.index
#     ]

#     if not new_carriers:
#         logger.info(
#             "No new virtual carriers added. All are already present in the network."
#         )
#         return

#     existing = n.carriers.loc[n.carriers.index.intersection(new_carriers)].copy()
#     missing = set(new_carriers) - set(existing.index)
#     if missing:
#         logger.warning(
#             f"The following carriers were requested but not found in the network and will be skipped:\n - {'\n - '.join(missing)}"
#         )

#     if existing.empty:
#         logger.warning("No valid carriers found to create virtual versions.")
#         return

#     virtual_carriers = existing.copy()
#     virtual_carriers.index = "virtual " + virtual_carriers.index
#     virtual_carriers["nice_name"] = "Virtual " + virtual_carriers["nice_name"]

#     n.add(
#         "Carrier",
#         virtual_carriers.index,
#         co2_emissions=0,
#         color=virtual_carriers.color,
#         nice_name=virtual_carriers.nice_name,
#     )

#     logger.info(
#         "Virtual carriers defined:\n - " + "\n - ".join(virtual_carriers.index.tolist())
#     )


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

    vpp = get_virtual_ppl_dataframe(n, certificate, planning_horizons)

    # Add Certificates
    n.add(
        "Carrier",
        "GoO",
        co2_emissions=0,
        color="#98c41c",
        nice_name="Guarantee of Origin",
    )

    # Add GO Market(s) and Demand
    if certificate["scope"] == "national":
        vpp["virtual_bus"] = ["GO Market " + country for country in vpp.country]
        create_national_go_market(n)
    else:
        vpp["virtual_bus"] = ["GO Market"]

        n.add(
            "Bus",
            "GO Market",
            carrier="GoO",
            country="EU",
            location="EU",
            x=n.buses.loc["EU", "x"],
            y=n.buses.loc["EU", "y"],
        )

        n.add(
            "Store",
            "GO Demand",
            e_nom_extendable=True,
            e_min_pu=-1,
            carrier="GoO",
            bus="GO Market",
        )

    # Add Virtual Power Plants
    n.add(
        "Generator",
        vpp.index,
        bus=vpp["virtual_bus"],
        carrier="GoO",
        p_nom_extendable=True,
    )

    n.export_to_netcdf(snakemake.output[0])
