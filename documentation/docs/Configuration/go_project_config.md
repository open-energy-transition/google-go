# Google-GO Project Configuration: `config.go.yaml`

This section details the configuration settings used in the Google-GO project, specifically focusing on the `config/config.go.yaml` file and its differences from the default PyPSA-Eur configuration found in `config/config.default.yaml`. These modifications enable the specialized modeling required for Guarantee of Origin (GO) certificates and renewable energy targets within the project's framework.

## General Project Context

The Google-GO project models a network with **39 nodes** (countries/regions) and employs a **myopic foresight** approach. The planning horizons are set to **2025, 2030, 2035, and 2040**, with a 5-year timestep between each horizon.

## Configuration Differences and New Additions

Below, each section of `config.go.yaml` is examined, highlighting differences from `config.default.yaml` and explaining new configurations.

### `run`

The `run` section specifies the naming conventions and scenario management for the project.

| Parameter          | `config.go.yaml` | Description                                                 |
| :----------------- | :--------------------------- | :---------------------------------------------------------- |
| name             | baseline                   | Sets the name for the run, used in output file naming.      |
| scenarios        |                              |                                                             |
| - file           | config/scenarios.go.yaml   | Specifies the YAML file defining the scenarios to be run.   |

### `foresight`

The `foresight` setting dictates the planning methodology used in the optimization.

| Parameter   | `config.go.yaml` | Description                                           |
| :---------- | :----------------| :---------------------------------------------------- |
| foresight | myopic         | Configures the model to use a myopic planning approach. |

### `scenario`

This section defines the core parameters for scenario generation, including spatial and temporal resolution.

| Parameter         | `config.go.yaml`            | Description                                                 |
| :---------------- | :-------------------------- | :---------------------------------------------------------- |
| clusters          | 39                          | Specifies the number of clustered nodes in the network.     |
| planning_horizons | [2025, 2030, 2035, 2040]    | Defines the years for which the model will be optimized.    |

### `enable`

The `enable` section controls the activation of various features and rules within the workflow.

| Parameter     | Unit | Values          | `config.go.yaml`             | Description                                          |
| :----------   | :--- | :-------------- | :--------------------------- | :--------------------------------------------------- |
| certificate   | -    | {true, false}   | true                       | If `true`, the Guarantee of Origin (GOs) market is modeled. |

### `co2_budget`

The `co2_budget` parameter in `config.go.yaml` is explicitly set to `false`, disabling the CO2 budget constraint in favor of CO2 pricing mechanisms. In `config.default.yaml`, this section contains a dictionary of year-specific CO2 budget values.

### `electricity`

This section contains extensive modifications related to electricity generation, storage, and demand.

| Parameter                       | Unit       | Values                          | `config.go.yaml`                                    | Description       |
| :------------------------------ | :--------- | :---| :-------------------------- | :------------------------------------------------- |
| novel_carriers | - | - | [green_ocgt, adv_firm_tech]   | A project-specific feature enabling novel technologies. Refer to `scripts/prepare_sector_networks.py` function `add_novel_technologies()` for details on its implementation.                                                               |
|extendable_carriers|||
| - StorageUnit | - | -  | [li-ion, iron-air]            | Specifies a list of storage technologies that can be extended, which is currently a project-specific feature not upstreamed to PyPSA-Eur. |
| - Store     | -   | -  | [li-ion, iron-air, H2 tank]   | Specifies a list of store technologies that can be extended, including project-specific options. |
| make_Store_StorageUnit | boolean    | {true, false}    | true  | If `true`, activates constraints in `scripts/solve_network.py` (`add_storage_inverter_fix` and `add_storage_duration_fix`) to make PyPSA stores and links behave as storage units, allowing them to not participate in the market. |
| max_hours  |||
| - li-ion   | hours | - | 6   | Defines the maximum hours of storage for lithium-ion batteries. |
| - iron-air | hours | - | 100 | Defines the maximum hours of storage for iron-air batteries.    |
| heating_demand |||
| - enable       | boolean    | {true, false} | true | If `true`, models exogenous electricity demand for heating. |
| - share        | percentage | -             | 0.16 | Specifies the share of total heating demand met by electricity. |
| hydrogen_demand |||
| - enable        | boolean    | {true, false} | false | If `true`, models exogenous electricity demand for hydrogen production. |
| - share         | percentage | -             | 0.    | Specifies the share of total hydrogen demand met by electricity. |
| ci_load |||
| - load_path | path  | -  | data/CI-load.csv  | Path to the Eurostat Commercial and Industrial (CI) load input data, used to determine Guarantee of Origin (GO) demand. |
| - load_year | year  | -  | 2023              | The reference year for the CI load data. |
| - ci_overwrite |||
| - - CH | percentage | -  | 0.58              | Manually overwritten CI share for Switzerland, as Eurostat data is unavailable. (Source: IEA) |
| - - GB | percentage | -   | 0.60             | Manually overwritten CI share for Great Britain, as Eurostat data is unavailable. (Source: IEA) |

### `clustering`

The `clustering` section defines parameters for network clustering, particularly for temporal resolution in the sector model.

| Parameter                  | Unit     | `config.go.yaml` | Description                                                        |
| :------------------------  | :------  | :---------       | :----------------------------------------------------------------- |
| temporal.resolution_sector | hours    | 3H             | Sets the temporal resolution for the sector model to 3 hours.      |

### `adjustments`

This section applies specific adjustments to component attributes within the network.

In `config.go.yaml`, `adjustments.sector.absolute` includes marginal cost additions for `DC` and `electricity distribution grid` links, and `PHS`, `li-ion`, `iron-air`, and `home battery` storage units/stores. These adjustments are implemented to mitigate unintended storage cycles. In `config.default.yaml`, `adjustments.sector.absolute` is set to `false`.

### `sector`

The `sector` section controls the inclusion of various sector-coupling components.

| Parameter     | `config.go.yaml` | Description                                                                 |
| :------------ | :--------------------------- | :-------------------------------------------------------------------------- |
| co2_network | false                      | Disables the modeling of the CO2 network.                                   |
| H2_network  | false                      | Disables the modeling of the hydrogen network.                              |
| gas_network | false                      | Disables the modeling of the gas network. All are disabled to focus on the electricity network. |

### `costs`

The `costs` section defines economic parameters for different technologies.

| Parameter                         | Unit           | Values | `config.go.yaml`                  | Description                                                                     |
| :------------------- | :----------- | :----- | :---  | :------------------------------------------------------------------------------ |
| marginal_cost        |||
| - green_ocgt         | €/MWh          | -      | 223.8 | Marginal cost for green open-cycle gas turbines, not available in PyPSA Technology Data. |
| - cost.adv_firm_tech | €/MWh          | -      | 13.6 | Marginal cost for advanced firm technologies, not available in PyPSA Technology Data. |
| overwrites           |||
| - capital_cost       |||
| - - green_ocgt       | €/MW           | -      | 72790 | Capital cost overwrite for green open-cycle gas turbines.                       |
| - - adv_firm_tech    | €/MW           | -      | 585153    | Capital cost overwrite for advanced firm technologies.                          |
| - lifetime           |||
| - - green_ocgt       | years          | -      | 25   | Lifetime overwrite for green open-cycle gas turbines.                           |
| - - adv_firm_tech    | years          | -      | 40  | Lifetime overwrite for advanced firm technologies.                              |
| - emission_prices    |||
| - - enable           | boolean        | -      | true  | Enables the use of emission prices in the model.                                |
| - - co2              | €/t_CO2        | -      | 0.    | Sets the price of CO2 emissions.                                                |

### `strip_network`

This entire section is a new configuration in `config.go.yaml`, designed to interact with `scripts/strip_network.py` for simplifying the network.

| Parameter       | Unit    | Values          | `config.go.yaml` | Description                                                                                                                                                                    |
| :------------ | :------ | :------------ | :---- | :--------------------------------------------------------- |
| enable        | boolean | {true, false} | true  | If `true`, activates the network stripping functionality.  |
| merge_load    | boolean | {true, false} | true  | If `true`, consolidates and simplifies electricity demand loads, merging auxiliary demands into the main electricity load per bus. |
| snapshots_start | date  | [false]       | false | Specifies a start date to filter the network's snapshots. Set to `false` to use the default snapshots.start from the snapshots section. |
| snapshots_end | date    | [false]       | false | Specifies an end date to filter the network's snapshots. Set to `false` to use the default snapshots.end from the snapshots section.  |

### `res_target`

This entire section is a new configuration in `config.go.yaml`, used for defining renewable energy targets.

| Parameter                     | Unit       | Values                                  | `config.go.yaml`  | Description                                                                                                                                                                                                                                                                                             |
| :---------------------------- | :--------- | :-------------------------------------- | :---------------------------- | :--------- |
| `system_share_target`         | boolean    | `{true, false}`                         | `false`                       | If `true`, sets a system-wide (e.g., EU+) renewable energy share target.                                                                                                                                                                                                              |
| `country_share_target`        | boolean    | `{true, false}`                         | `false`                       | If `true`, sets country-specific renewable energy share targets.                                                                                                                                                                                                                      |
| `tyndp_data.enable`           | boolean    | `{true, false}`                         | `true`                        | If `true`, uses data from the TYNDP (Ten-Year Network Development Plan) for renewable energy share targets. If `false`, `res_shares_overwrite` is used.                                                                                                                                 |
| `tyndp_data.tyndp_scenario`   | string     | `{"National Trends", "Distributed Energy"}` | `National Trends`             | Specifies the TYNDP scenario to use for renewable energy targets (e.g., "National Trends", "Distributed Energy").                                                                                                                                                                     |
| `res_shares_overwrite.XK`     | percentage | -                                       | `0.43`                        | Manually overwrites the renewable energy share target for Kosovo (XK) for 2030, as this data is not provided in TYNDP. (Source: https://ember-energy.org/data/global-renewable-power-sector-targets-2030/)                                                                           |
| `res_shares_overwrite`        | dict       | -                                       | (`XK: 0.43` and others)       | This dictionary allows manual overwrites for renewable energy shares. To overwrite a system-wide share, "EU+" should be used as the country code. This configuration directly impacts the `add_rps_constraints(n, planning_horizons)` function in `scripts/solve_network.py`. |

### `grid_policy`

This entire section is a new configuration in `config.go.yaml`, used to define categories of energy carriers.

| Parameter          | Unit | Values | `config.go.yaml` | Description   |
| :----------------- | :--- | :----- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| renewable_carriers | list | -      | [green_ocgt, adv_firm_tech, offwind, offwind-ac, offwind-dc, offwind-float, onwind, ror, hydro, urban central solid biomass CHP, geothermal, solar, solar-hsat, solar rooftop] | Defines a list of carriers considered as renewable energy sources. Used by `add_rps_constraints(n, planning_horizons)` in `scripts/solve_network.py`.                       |
| clean_carriers   | list | -      | [green_ocgt, adv_firm_tech,offwind, offwind-ac, offwind-dc, offwind-float, onwind, ror, hydro, urban central solid biomass CHP, geothermal, solar, solar-hsat, solar rooftop, nuclear, allam] | Defines a list of carriers considered as clean energy sources. Used by `add_rps_constraints(n, planning_horizons)` in `scripts/solve_network.py`.                             |
| emitters         | list | -      | [CCGT, OCGT, coal, lignite, oil]                                                                                                                                               | Defines a list of carriers considered as emitting sources. Used by `add_rps_constraints(n, planning_horizons)` in `scripts/solve_network.py`.                               |

### `certificate`

This entire section is a new and critical configuration in `config.go.yaml`, directly related to `scripts/add_certificate.py` and the certificate constraints in `scripts/solve_network.py` (`add_virtual_ppl_matching`, `add_buffer_matching`, and `add_virtual_storage_matching`).

| Parameter                       | Unit       | Values                                                                                                                            | `config.go.yaml`  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :------------------------------ | :--------- | :-------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| plant_carriers | list       | -   | [green_ocgt, adv_firm_tech, offwind, ..., nuclear] | Defines which technology carriers are allowed to participate in the Guarantee of Origin (GO) market. |
| plant_status   | list       | [new, existing] | [new]  | Specifies the age status of power plants eligible for the GO market. Options are `new`, `existing`, or both. |
| max_plant_lifetime_as_new   | years      | | 2         | Determines the maximum age (in years) for a power plant to be considered "new" within each planning horizon. |
| plant_grouping              | list       | [] or [carrier, status] | [carrier, status] | Defines how virtual carriers are aggregated. This allows for disaggregating generators by attributes like carrier and status, providing detailed information on which technologies contribute to the GO market. An empty list [] would group all virtual carriers together, losing specific carrier/status information. |
| bus_grouping                  | list       | [] or [carrier, status]  | []  | Defines how GO supply buses are aggregated. An empty list [] groups all GO supply into one per country. Other options (e.g., [carrier, status]) are possible to further disaggregate supply. |
| aib_filepath                  | path       | str |  data/AIB_monthly_data.csv   | The file path to the Association of Issuing Bodies (AIB) data, containing historical GO demand statistics.
| storage_carriers | list       | -  |[li-ion, iron-air, H2 Store] | Specifies which storage technologies are eligible to participate in the GO market. |
| GO_penalty       | €/MWh      | -  | 1e5                         | A safeguard mechanism: a penalty incurred if GO demand cannot be fulfilled by supply, preventing infeasibility in the model.     |
| map              | list       | -    | (various coordinates)   | This sub-section is solely for visualizing the GO market on PyPSA map plots, defining x and y offsets for supply, demand, and market components. |

#### Backgroun demand

| Parameter   | Unit       | Values    | `config.go.yaml`  | Description  |
| :---------- | :--------- | :-------- | :---------------- | :----------- |
| enable      | boolean    | {true, false}                                                                                                                   | false                       | If `true`, enables the modeling of background GO demand.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| scope       | string     | global                                                                                                                          | global                      | Defines the geographical scope of the background GO market. Set to global (EU-wide) because national GO demand can exceed national electricity demand. In theory, this is an EU-wide market allowing certificate trading between participating countries. Not all countries in the model participate in AIB; only those listed in `supply_filter.country` do. This market is currently not used in our scenarios but can be activated.                                                                                                                      |
| supply_filter.country | list       | -                                                                                                                                 | ['AT', 'BE', ..., 'SK'] | A list of countries whose supply is filtered for the background GO market.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| supply_filter.status | boolean/string | [false, 'new', 'existing']                                                                                      | false                       | A filter for power plant status (e.g., `new`, `existing`). `false` means no filter is applied by status.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| supply_filter.carrier | boolean/string | [false, 'solar', 'onwind', ...]                                                                                 | false                       | A filter for power plant carriers. `false` means no filter is applied by carrier.                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| market_name | string     | -                                                                                                                                 | AIB                         | A suffix for the name of the market.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| go_profile  | string     | {"historical", "total_daily_avg", "total", "baseload"}                                                                  | historical                  | Defines the load profile of the GO demand (e.g., `historical`, `total_daily_avg`, `total`, `baseload`).                                                                                                                                                                                                                                                                                                                                                                                                                      |
| participant | string     | {"load", "ci"}                                                                                                              | load                        | Specifies the share of energy matching reference. `load` refers to the entire grid; `ci` refers to commercial and industrial sectors only.                                                                                                                                                                                                                                                                                                                                                                                           |
| energy_matching | percent    | -                                                                                                                                 | 100                         | The percentage of historical GO demand that is required to be matched with certificates.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| hourly_matching | percent    | -                                                                                                                                 | 0                           | The percentage of GO demand that must match hourly (vs. buffered). For the AIB market, no hourly matching is required.           |

#### New Demand

| Parameter                       | Unit       | Values                                                                                                                            | `config.go.yaml`  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :------------------------------ | :--------- | :-------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| enable             | boolean    | {true, false}                                                                                                                   | false                       | If `true`, enables the modeling of new GO demand. Multiple demand types can coexist in one market.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| scope              | string     | {"national", "global"}                                                                                                      | national                    | Defines the geographical scope of the new GO market (`national` or `global`). This market is designed to be nationally bounded with hourly matching, with all countries participating in their respective national GO markets. The main scenarios and sensitivities of the project revolve around the configurations listed here.                                                                                                                                                                                                               |
| supply_filter.country | boolean/string | [false, 'AT', 'BE', ...]                                                                                                        | false                       | A filter for countries whose supply is considered for the new GO market. `false` means no filter. This requires `bus_grouping` to be disaggregated.                                                                                                                                                                                                                                                                                                                                                                                         |
| supply_filter.status | boolean/string | [false, 'new', 'existing']                                                                                                      | false                       | A filter for power plant status (e.g., `new`, `existing`). `false` means no filter is applied by status. This requires `bus_grouping` to be disaggregated.                                                                                                                                                                                                                                                                                                                                                                                         |
| supply_filter.carrier | boolean/string | [false, 'solar', 'onwind', ...]                                                                                                 | `false`                       | A filter for power plant carriers. `false` means no filter is applied by carrier. This requires `bus_grouping` to be disaggregated.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| market_name        | string     | -                                                                                                                                 | ""                        | A suffix for the name of the market.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| go_profile         | string     | {"total_daily_avg", "total", "baseload"}                                                                                  | `total`                       | Defines the load profile of the GO demand (`total_daily_avg`, `total`, `baseload`). This determines whether the GO demand follows electricity demand fluctuations, is averaged daily, or remains constant.                                                                                                                                                                                                                                                                                                                                   |
| participant        | string     | {"load", "ci"}                                                                                                              | ci                          | Specifies the share of energy matching reference (`load` for entire grid, `ci` for commercial and industrial sectors only).                                                                                                                                                                                                                                                                                                                                                                                                                      |
| energy_matching    | percent    | -                                                                                                                                 | 50                          | The percentage of energy matching required for the new GO demand, relative to the participant's load.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| hourly_matching    | percent    | -                                                                                                                                 | 90                          | The percentage of GO demand that must match hourly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |


### `overwrite_years`

This entire section is a new project-specific configuration in `config.go.yaml`. Its current purpose is to define year-specific cost overwrites, specifically for reducing the marginal price of `green_ocgt` in future planning horizons.

| Parameter                      | Unit  | Values | `config.go.yaml` | Description                                                                    |
| :----------------------------- | :---- | :----- | :--------------------------- | :----------------------------------------------------------------------------- |
| `2025.costs.marginal_cost.green_ocgt` | €/MWh | -      | `444.7`                      | Marginal cost for green OCGT in 2025.                                          |
| `2030.costs.marginal_cost.green_ocgt` | €/MWh | -      | `371.1`                      | Marginal cost for green OCGT in 2030.                                          |
| `2035.costs.marginal_cost.green_ocgt` | €/MWh | -      | `297.5`                      | Marginal cost for green OCGT in 2035.                                          |
| `2040.costs.marginal_cost.green_ocgt` | €/MWh | -      | `223.8`                      | Marginal cost for green OCGT in 2040.                                          |
