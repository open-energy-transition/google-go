import pandas as pd
import numpy as np
import country_converter as coco
import argparse
import re
import pypsa

from pathlib import Path
from tqdm import tqdm

cc = coco.CountryConverter()

def get_score_load(n, score="cfe"):

    score = n.buses_t[f"{score}_p"].copy()
    score.columns = score.columns.map(n.buses.country)
    score = score.T.groupby(score.columns).sum().T
    
    elec_list = list(set(n.loads.carrier.unique()) - {"GoO"})
    load = n.statistics.energy_balance(
        nice_names=False, 
        comps=["Load"], 
        groupby=["country"], 
        carrier=elec_list, 
        groupby_time=False
    )
    load = - load.groupby("country").sum().T

    return score, load

def get_hourly_energy_matrix(n, score, load):
    weighting = n.snapshot_weightings.objective
    hourly_matching = pd.Series(index=range(1, 121), dtype=float)
    
    for energy_matching in hourly_matching.index:
        load_set = load * energy_matching / 100
    
        unfilled_energy = weighting @ (load_set - score).clip(lower=0)
        remaining_score = weighting @ (score - load_set).clip(lower=0)
        total_load = weighting @ load_set
    
        if remaining_score >= unfilled_energy:
            hourly_matching[energy_matching] = 100 * (1 - unfilled_energy / total_load)
        else:
            hourly_matching[energy_matching] = np.nan

    return hourly_matching

def main(output="results_frontier", score="cfe", tutorial=False):

    tqdm_kwargs = dict(
        ascii=False,
        unit=" Networks",
        desc="Processing networks ",
    )
    
    data_all = pd.DataFrame()
    base_path = Path(f"../results/")
    files = list(base_path.rglob("*.nc"))
    
    for fn in tqdm(files, **tqdm_kwargs):
    
        data = pd.DataFrame()
        
        n = pypsa.Network(fn)
    
        res, load = get_score_load(n, score=score)
    
        # First, create an EU wide results
        global_res = res.sum(axis=1)
        global_load = load.sum(axis=1)
    
        data["EU"] = get_hourly_energy_matrix(n, global_res, global_load)

        # Then, create for all countries
        for country in res.columns:
            national_res = res[country]
            national_load = load[country]
    
            country_name = cc.convert(names=country, to='name_short')
            data[f"{country_name}"] = get_hourly_energy_matrix(n, national_res, national_load)

        # Set the multicolumns
        parts = fn.parts
        scenario = parts[parts.index("results") + 1]
        year = re.findall(r'\d{4}', parts[-1])[0]
    
        data.columns = pd.MultiIndex.from_product(
            [[scenario], [year], data.columns],
            names=["scenario", "year", "country"]
        )
    
        data_all = pd.concat([data_all, data], axis=1)
    
        if tutorial:
            if 'iterate' not in locals():
                iterate = 0
                
            iterate += 1
            if iterate > 6:
                break

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
        default="results_frontier",
        help="Name of the output CSV file (without extension)"
    )

    parser.add_argument(
        "--score",
        type=str,
        default="cfe",
        help="Score type to use (e.g., cfe, res)"
    )

    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="If set, run only a few iterations for testing"
    )

    args = parser.parse_args()

    main(output=args.output, score=args.score, tutorial=args.tutorial)
