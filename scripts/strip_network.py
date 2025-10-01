# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This script strips the network to produce electricity only models
"""

import logging

import pypsa

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)

carrier_to_keep = [
    # Electricity Sources (Renewable & Non-renewable)
    "AC",
    "DC",
    "solar",
    "solar-hsat",
    "solar rooftop",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "hydro",
    "ror",
    "PHS",
    "nuclear",
    "lignite",
    "coal",
    "oil",
    "oil primary",
    "gas",
    "uranium",
    # Battery-related
    "battery",
    "EV battery",
    "home battery",
    "battery charger",
    "battery discharger",
    "home battery charger",
    "home battery discharger",
    # Electricity Use Cases
    "electricity",
    "industry electricity",
    "agriculture electricity",
    "agriculture machinery electric",
    # Transport
    "land transport EV",
    "BEV charger",
    # Hydrogen-related
    "H2",
    "H2 Store",
    "H2 Fuel Cell",
    "H2 Electrolysis",
    "H2 pipeline",
    "H2 turbine",
    # CO₂ / Carbon-related
    "co2",
    "co2 stored",
    "co2 sequestered",
    "CO2 pipeline",
    "DAC",
    # Fossil Fuel Technologies
    "CCGT",
    "OCGT",
    # "SMR CC",
    # Infrastructure / Grid
    "electricity distribution grid",
    "gas pipeline",
    "low voltage",
    # Miscellaneous / Unknown
    None,
]


def extend_carrier_list(config_elec):
    
    config_carriers = [
        carrier
        for key in ["conventional_carriers", "renewable_carriers", "novel_carriers"]
        for carrier in config_elec[key]
    ]
    
    storage_carriers = [
        suffix
        for key in ["StorageUnit", "Store", "Link"]
        for carrier in config_elec["extendable_carriers"][key]
        for suffix in [carrier, carrier + " charger", carrier + " discharger"]
    ]
    return config_carriers + storage_carriers


def strip_network(n, carriers):  # , country=None):
    """
    Strip network to core electricity related components

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to modify.
    carriers : list
        The carrier to be kept

    Returns
    -------
    pypsa.Network
    """

    m = n.copy()

    # if country:
    #     countries_to_keep = country + [""]
    # else:
    #     countries_to_keep = m.buses.country.unique()

    nodes_to_keep = m.buses[
        m.buses.carrier.isin(
            carriers
        )  # & m.buses.country.isin(countries_to_keep)
    ].index
    m.remove("Bus", n.buses.index.symmetric_difference(nodes_to_keep))

    for c in m.iterate_components(
        ["Generator", "Link", "Line", "Store", "StorageUnit", "Load"]
    ):
        if c.name in ["Link", "Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(
                nodes_to_keep
            )
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean & c.df.carrier.isin(carrier_to_keep)]
        to_drop = c.df.index.symmetric_difference(to_keep)
        m.remove(c.name, to_drop)

    m.links.loc[m.links.carrier.isin(["H2 Electrolysis", "H2 Fuel Cell"]), "bus2"] = ""

    logger.info("Strip network to core electricity related components")

    return m


def merge_load(n):
    """
    Consolidates and simplifies electricity demand loads in a PyPSA network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to modify.

    Returns
    -------
    None
        pypsa.Network is then modified.
    """

    aux_elec_demand = [
        "industry electricity",
        "agriculture electricity",
        "agriculture machinery electric",
    ]

    # Simplify load
    elec_index = n.loads[n.loads.carrier.isin(aux_elec_demand)].index
    load = n.loads.loc[elec_index].copy()
    p_set = load.groupby("bus")["p_set"].sum() * 2

    n.remove("Load", elec_index)

    p_set = p_set.rename({v: k for k, v in n.loads.bus.items()})
    n.loads_t.p_set[p_set.index] += p_set

    logger.info("Merge electricity demand loads to one electricity loads per bus")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "strip_network",
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

    config_elec = snakemake.params.electricity
    options = snakemake.params.strip_network
    
    carrier = carrier_to_keep + extend_carrier_list(config_elec)

    n = strip_network(n, carrier)

    if options["merge_load"]:
        merge_load(n)

    if options["snapshots_start"]:
        new_snapshots = n.snapshots[(n.snapshots >= options["snapshots_start"])]
        n.set_snapshots(new_snapshots)

        logger.info(f"strip snapshots to start at {options['snapshots_start']}")

    if options["snapshots_end"]:
        new_snapshots = n.snapshots[(n.snapshots < options["snapshots_end"])]
        n.set_snapshots(new_snapshots)

        logger.info(f"strip snapshots to end at {options['snapshots_end']}")

    n.export_to_netcdf(snakemake.output[0])
