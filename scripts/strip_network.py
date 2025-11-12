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
    overwrite_config_by_year,
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

    for c in n.components:
        if c.name not in ["Generator", "Link", "Line", "Store", "StorageUnit", "Load"]:
            continue

        if c.name in ["Link", "Line"]:
            location_boolean = c.static.bus0.isin(nodes_to_keep) & c.static.bus1.isin(
                nodes_to_keep
            )
        else:
            location_boolean = c.static.bus.isin(nodes_to_keep)
        to_keep = c.static.index[location_boolean & c.static.carrier.isin(carrier)]
        to_drop = c.static.index.symmetric_difference(to_keep)
        m.remove(c.name, to_drop)

    m.links.loc[m.links.carrier.isin(["H2 Electrolysis", "H2 Fuel Cell"]), "bus2"] = ""

    logger.info("Strip network to core electricity related components")

    return m


def merge_load(n, config_elec):
    """
    Consolidates and simplifies electricity demand loads in a PyPSA network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to modify.
    
    config_elec : dict
        The electricity configuration dictionary.

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
    p_set = load.groupby("bus")["p_set"].sum()

    n.remove("Load", elec_index)

    p_set = p_set.rename({v: k for k, v in n.loads.bus.items()})
    n.loads_t.p_set[p_set.index] += p_set

    # Add exogenous electricity demand for heating
    if config_elec["heating_factors"].get("enable", False):
        logger.info("Adding exogenous electricity demand for heating")
        heat_share = config_elec["heating_factors"]["share"]
        heat_elc_load = n.loads_t.p_set[p_set.index] * heat_share / (1-heat_share)
        n.loads_t.p_set[p_set.index] += heat_elc_load

    if config_elec["hydrogen_factors"].get("enable", False):
        logger.info("Adding exogenous electricity demand for hydrogen production")
        hydrogen_share = config_elec["hydrogen_factors"]["share"]
        hydrogen_elc_load = n.loads_t.p_set[p_set.index] * hydrogen_share / (1-hydrogen_share)
        n.loads_t.p_set[p_set.index] += hydrogen_elc_load

    logger.info("Merge electricity demand loads to one electricity loads per bus")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "strip_network",
            run="baseline",
            opts="",
            clusters="39",
            configfiles="config/config.go.yaml",
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
        overwrite_config_by_year(snakemake.config, snakemake.params, snakemake.wildcards.get("planning_horizons", None))
        merge_load(n, config_elec)

    if options["snapshots_start"]:
        new_snapshots = n.snapshots[(n.snapshots >= options["snapshots_start"])]
        n.set_snapshots(new_snapshots)

        logger.info(f"strip snapshots to start at {options['snapshots_start']}")

    if options["snapshots_end"]:
        new_snapshots = n.snapshots[(n.snapshots < options["snapshots_end"])]
        n.set_snapshots(new_snapshots)

        logger.info(f"strip snapshots to end at {options['snapshots_end']}")

    n.export_to_netcdf(snakemake.output[0])
