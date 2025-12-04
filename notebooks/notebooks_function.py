import pandas as pd
import numpy as np
import pypsa
import warnings

rename_map = {
    # clean
    "nuclear": "clean",
    "uranium": "clean",
    "adv_firm_tech": "clean",

    # co2
    "co2": "co2",
    "co2 sequestered": "co2",
    "co2 stored": "co2",

    # electricity
    "electricity": "electricity",

    # electricity grid
    "AC": "electricity grid",
    "DC": "electricity grid",
    "electricity distribution grid": "electricity grid",
    "low voltage": "electricity grid",

    # EV
    "BEV charger": "EV",
    "EV battery": "EV",
    "land transport EV": "EV",

    # fossil
    "CCGT": "fossil",
    "OCGT": "fossil",
    "coal": "fossil",
    "gas": "fossil",
    "lignite": "fossil",
    "oil": "fossil",
    "oil primary": "fossil",

    # GoO
    "GoO": "GoO",
    "virtual adv_firm_tech": "GoO",
    "virtual green_ocgt": "GoO",
    "virtual nuclear": "GoO",
    "virtual ror": "GoO",
    "virtual offwind-ac": "GoO",
    "virtual offwind-dc": "GoO",
    "virtual offwind-float": "GoO",
    "virtual onwind": "GoO",
    "virtual solar": "GoO",
    "virtual solar rooftop": "GoO",
    "virtual solar-hsat": "GoO",
    "virtual H2 Store": "GoO",
    "virtual iron-air": "GoO",
    "virtual li-ion": "GoO",
    
    "GO penalty":"GO penalty",

    # renewables
    "green_ocgt": "renewables",
    "hydro": "renewables",
    "offwind-ac": "renewables",
    "offwind-dc": "renewables",
    "offwind-float": "renewables",
    "onwind": "renewables",
    "ror": "renewables",
    "solar": "renewables",
    "solar rooftop": "renewables",
    "solar-hsat": "renewables",

    # storage
    "H2": "storage",
    "H2 Electrolysis": "storage",
    "H2 Fuel Cell": "storage",
    "H2 Store": "storage",
    "H2 tank": "storage",
    "H2 turbine": "storage",
    "PHS": "storage",
    "home battery": "storage",
    "home battery charger": "storage",
    "home battery discharger": "storage",
    "iron-air": "storage",
    "iron-air charger": "storage",
    "iron-air discharger": "storage",
    "li-ion": "storage",
    "li-ion charger": "storage",
    "li-ion discharger": "storage",
}

category_colors = {
    "EV": "#baf238",              # matches BEV / EV battery theme (bright lime-green)
    "GoO": "#46caf0",             # close to ammonia/NH3 + certification-style "blue"
    "clean": "#ff8c00",           # matches nuclear/uranium (clean baseload)
    "co2": "#f29dae",             # matches co2 family (soft pink)
    "electricity grid": "#97ad8c",# matches distribution grid / low voltage
    "fossil": "#545454",          # coal baseline grey (neutral fossil indicator)
    "renewables": "#235ebc",      # wind blue (iconic renewable color)
    "storage": "#ace37f",          # battery green (unifying storage theme)
    "GO penalty": "#dd2e23",
}

grouping_storage = {
    "H2": "H2",
    "H2 Electrolysis": "H2",
    "H2 Fuel Cell": "H2",
    "H2 Store": "H2",
    "H2 tank": "H2",
    "H2 turbine": "H2",
    "home battery": "home battery",
    "home battery charger": "home battery",
    "home battery discharger": "home battery",
    "iron-air": "iron-air",
    "iron-air charger": "iron-air",
    "iron-air discharger": "iron-air",
    "li-ion": "li-ion",
    "li-ion charger": "li-ion",
    "li-ion discharger": "li-ion",
    "BEV charger": "EV",
    "EV battery": "EV",
    "land transport EV": "EV",
}

def prepare_network(n):

    # Fill missing colors
    append_color = {
        '':'#aaaaaa', 
        'none':'#aaaaaa',
        'H2 tank':'#bf13a0', 
        'H2 tank Electrolysis':'#bf13a0',
        'H2 tank Fuel Cell':'#bf13a0',
        'virtual H2 Store':'#bf13a0', 
        'adv_firm_tech':'#d19D00', 
        'green_ocgt':'#2fb537', 
        'virtual adv_firm_tech':'#d19D00',
        'virtual green_ocgt':'#2fb537', 
        'iron-air':"grey",
        'iron-air charger':"grey",
        'iron-air discharger':"grey",
        'virtual iron-air':"grey",
        'GO penalty':"#dd2e23",
    }

    for key, value in append_color.items():
        n.carriers.loc[key,"color"] = value

    return n

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

def rename_year(n, year):

    # Change snapshotdate to year
    for c in n.components:
        for table in c.dynamic:
            c.dynamic[table].index = c.dynamic[table].index.map(lambda x: x.replace(year=year))
    
    n.snapshots = n.snapshots.map(lambda x: x.replace(year=year))

    return n
    

def strip_network_GoO(n):
    m = n.copy()
    
    nodes_to_keep = m.buses[m.buses.carrier == "GoO"].index
    m.remove("Bus", m.buses.index.symmetric_difference(nodes_to_keep))
    
    carrier_to_keep = list(m.carriers.filter(like="virtual", axis=0).index)
    carrier_to_keep.append("GoO")
    
    for c in n.components:
        if c.name not in ["Generator", "Link", "Line", "Store", "StorageUnit", "Load"]:
            continue
        
        if c.name in ["Link", "Line"]:
            location_boolean = c.static.bus0.isin(nodes_to_keep) & c.static.bus1.isin(
                nodes_to_keep
            )
        else:
            location_boolean = c.static.bus.isin(nodes_to_keep)
        to_keep = c.static.index[location_boolean & c.static.carrier.isin(carrier_to_keep)]
        to_drop = c.static.index.symmetric_difference(to_keep)
        m.remove(c.name, to_drop)

    return m


def get_stats_all(df_all, stats, **kwarg):
    cols = {}
    color_list = {}

    for year, sc in df_all.index:
        n = df_all.loc[(year, sc)]
        # Use tuple (year, scenario) as column key instead of network name
        cols[(year, sc)] = getattr(n.statistics, stats)(nice_names=False, **kwarg)
        color_list.update(n.carriers["color"].to_dict())

    if "aggregate_time" in kwarg:
        return cols, None

    df = pd.DataFrame(cols).fillna(0)
    # Create MultiIndex columns with year and scenario
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['year', 'scenario'])
    colors = [color_list[i] for i in df.index.get_level_values("carrier")]
    
    return df, colors   


def get_stats_prices(df_all, **kwarg):
    cols = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        for year, sc in df_all.index:
            n = df_all.loc[(year, sc)]
            cols[(year, sc)] = getattr(n.statistics, "prices")(**kwarg)

    df = pd.DataFrame(cols).fillna(0)
    # Create MultiIndex columns with year and scenario
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['year', 'scenario'])

    if "bus_carrier" in df.index.names:
        colors = [n.carriers.loc[i,"color"] for i in df.index.get_level_values("bus_carrier")]
    else:
        colors = None
    
    return df, colors


def sum_except_color(group):
    sum_values = group.drop(columns='color').sum()
    color = group['color'].iloc[0]
    sum_values['color'] = color
    
    return sum_values


def clean_virtual_names(df):
    """Remove 'virtual ' prefix from index names for cleaner legend labels."""
    if isinstance(df.index, pd.MultiIndex):
        # For MultiIndex, clean the 'carrier' level if it exists
        if 'carrier' in df.index.names:
            level_idx = df.index.names.index('carrier')
            new_index = df.index.set_levels(
                df.index.levels[level_idx].str.replace('virtual ', '', regex=False),
                level=level_idx
            )
            df.index = new_index
    else:
        # For simple index, just replace
        df.index = df.index.str.replace('virtual ', '', regex=False)
    return df


def group_by_build_year(n, c):
    """Group components by their build year."""
    if "build_year" in n.c[c].static.columns:
        return n.c[c].static['build_year']
    else:
        return None 


def group_by_country_focus(n, c, port=""):
    """Group components by specific country and find alternative if missing."""
    from pypsa.statistics import groupers
    
    bus = f"bus{port}"
    component_buses = n.c[c].static[bus]
    buses_country = n.c.buses.static.country
    
    country = groupers._map_with_multiindex(component_buses, buses_country)
    
    if "bus1" in n.c[c].static.columns:
        missing = country.isna() | (country == "")

        component_bus1 = n.c[c].static["bus1"]
        country[missing] = groupers._map_with_multiindex(component_bus1, buses_country)[missing]

    return country.rename("country1")

pypsa.statistics.groupers.add_grouper("build_year", group_by_build_year)
pypsa.statistics.groupers.add_grouper("country1", group_by_country_focus)