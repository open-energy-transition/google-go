This document outlines specific functions within the `solve_network.py` script that are particularly relevant to the Google-GO project. These functions introduce custom constraints and functionalities to the PyPSA network, enabling specialized modeling for Guarantee of Origin (GoO) certificates and renewable energy targets.

### Background Constraints

*   **`add_rps_constraints(n, planning_horizons)`**: This function is utilized when national renewable energy targets are activated. It adds Renewable Portfolio Standard (RPS) constraints to the network, based on TYNDP (Ten-Year Network Development Plan) results or manual inputs. These constraints enforce a minimum share of renewable energy generation at a national or system-wide level for each planning horizon.

### GO Constraints

The following functions (`add_virtual_ppl_matching`, `add_buffer_matching`, and `add_virtual_storage_matching`) are activated only when the `certificate` option is set to `true` in the `enable` configuration.

*   **`add_virtual_ppl_matching(n)`**: This function is crucial for binding the generation of electricity in the background PyPSA model to the Guarantee of Origin (GO) layer. It ensures that the output of virtual power plants (VPPs) in the GO layer accurately reflects the aggregated power generation from their corresponding real power plant units (generators and links) in the background model.

*   **`add_buffer_matching(n)`**: This function establishes constraints related to the GoO buffer, which manages hourly matching limits. It ensures that the sum of buffer dischargers is equivalent to the sum of buffer chargers, and that the buffer discharge does not exceed the defined hourly matching limits, facilitating the accounting of GoO certificates over time.

*   **`add_virtual_storage_matching(n)`**: Similar to `add_virtual_ppl_matching`, this function binds the storage operations in the background PyPSA model to the GO layer. It ensures that the virtual storage units in the GO layer accurately reflect the charging and discharging activities of their corresponding real storage units in the background model.

### Additional Storage Constraints

*   **`add_storage_inverter_fix(n)`**: This function introduces a constraint to ensure that for certain storage technologies (excluding H2 and H2 tank), the charger capacity is equivalent to the discharger capacity, adjusted by efficiency. 

*   **`add_storage_duration_fix(n)`**: This function adds a constraint to ensure that the Power-to-Energy (P/E) ratio for batteries is based on their `max_hours` configuration.

These constraints provides the possibility of having batteries that do not participate in the market, making stores and storage units essentially equivalent for these battery types. Both `add_storage_inverter_fix(n)` and `add_storage_duration_fix(n)` impact storages that are defined as PyPSA stores and links to make them behave as storage_units. Storages defined as stores and links are binded with the GO Layer, whereas storage_units are not.