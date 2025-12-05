from notebooks_function import *

import pandas as pd
import pypsa
import country_converter as coco
import argparse
import re

from pathlib import Path
from tqdm import tqdm

cc = coco.CountryConverter()

elec_demand = [
    "electricity",
    "land transport EV",
    "industry electricity",
    "agriculture electricity",
    "agriculture machinery electric",
]

def collect_electricity_balance(n):
    
    df_energy_balance = n.statistics.energy_balance(
        groupby=["country","bus_carrier","carrier"],
        nice_names=False,
        aggregate_time=False
    )
    df_energy_balance = df_energy_balance[
        df_energy_balance.index.get_level_values("bus_carrier").isin(
            ["AC", "low voltage"
        ])
        & ~df_energy_balance.index.get_level_values("carrier").isin([
            "AC", "DC", "electricity", "low voltage", 
            "electricity distribution grid", "BEV charger",
            "home battery charger", "home battery discharger"
        ])
    ]
    df_energy_balance = df_energy_balance.groupby(["country","carrier"]).sum()
    df_energy_balance["type"] = "area"
    df_energy_balance = df_energy_balance.set_index("type", append=True)
    
    df_load = n.statistics.energy_balance(
        groupby=["country"], 
        components=["Load"], 
        carrier=elec_demand,
        nice_names=False,
        aggregate_time=False
    )
    df_load = df_load.groupby("country").sum()
    df_load["carrier"] = "electricity"
    df_load["type"] = "line"
    df_load = df_load.set_index(["carrier","type"], append=True)

    # combine them together
    df = pd.concat([df_energy_balance, df_load])
    df *= 1e-3
    df["Results"] = "Electricity Balance"
    df["y_label"] = "Generation (GW)"
    df = df.set_index(["Results","y_label"], append=True)

    return df


def collect_go_market(m):
    
    df_go_market = m.statistics.energy_balance(
        groupby=["country", "carrier"], 
        nice_names=False,
        aggregate_time=False
    ).groupby(["country","carrier"]).sum()
    
    df_go_market = df_go_market[
        ~df_go_market.index.get_level_values("carrier").isin(["GoO"])
    ]
    df_go_market = clean_virtual_names(df_go_market)
    df_go_market["type"] = "area"
    df_go_market = df_go_market.set_index("type", append=True)
    
    df_go_load = - m.statistics.energy_balance(
        groupby=["country", "carrier"], 
        components=["Load"],
        nice_names=False,
        aggregate_time=False
    ).groupby(["country","carrier"]).sum()
    df_go_load["type"] = "line"
    df_go_load = df_go_load.set_index("type", append=True)

    # combine them together
    df = pd.concat([df_go_market, df_go_load])
    df *= 1e-3
    df["Results"] = "GO Market"
    df["y_label"] = "Generation (GW-GoO)"
    df = df.set_index(["Results","y_label"], append=True)

    return df


def main(output="results_time_series", tutorial=False):

    tqdm_kwargs = dict(
        ascii=False,
        unit=" Networks",
        desc="Processing networks ",
    )
    
    data_all = pd.DataFrame()
    base_path = Path(f"../results/")
    files = list(base_path.rglob("*.nc"))
    
    for fn in tqdm(files, **tqdm_kwargs):
        
        n = pypsa.Network(fn)
        df = collect_electricity_balance(n)

        m = strip_network_GoO(n)
        if not m.generators.empty:
            df = pd.concat([df,collect_go_market(m)])
    
        # Set the multicolumns
        parts = fn.parts
        scenario = parts[parts.index("results") + 1]
        year = re.findall(r'\d{4}', parts[-1])[0]
    
        df["scenario"] = scenario
        df["year"] = year
        df = df.set_index(["scenario","year"], append=True)
        df = df.reorder_levels(["scenario", "year", "Results", "y_label", "country", "type", "carrier"])

        data_all = pd.concat([data_all,df])

        if tutorial:
            if 'iterate' not in locals():
                iterate = 0
                
            iterate += 1
            if iterate > 6:
                break

    # Convert to short name
    countries = data_all.index.get_level_values("country").unique()
    country_mapping = {c: cc.convert(c, to="short_name") for c in countries}
    data_all = data_all.rename(index=country_mapping, level="country")

    data_all

    # Save to an csv
    print(f"Saving results in {output}.csv")
    data_all.to_csv(output + ".csv")

if __name__ == "__main__":
    
    import logging
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Generate procurement energy frontier for all scenarios and all year.")

    parser.add_argument(
        "--output",
        type=str,
        default="results_time_series",
        help="Name of the output CSV file (without extension)"
    )

    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="If set, run only a few iterations for testing"
    )

    args = parser.parse_args()

    main(output=args.output, tutorial=args.tutorial)