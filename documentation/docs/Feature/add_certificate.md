This script, `add_certificate.py`, is a specialized script developed for the Google-GO project within the PyPSA-Eur workflow. It is designed to integrate Guarantee of Origin (GoO) certificates into energy system models. It essentially creates "virtual" representations of power plants and storage units, and sets up a market for these certificates within the PyPSA network.

Here's a breakdown of its main functions, categorized by their roles:

*   **Functions for Filtering and Binding with GO Market**:
    *   **`get_virtual_ppl_dataframe`**: Constructs a DataFrame that defines virtual power plants (VPPs) by grouping real power plant units (generators and links) based on certificate criteria. This information is used both here and for constraints in `solve_network.py`.
    *   **`get_virtual_storage_dataframe`**: Identifies real storage units and their associated charger/discharger links, grouping them into virtual storage systems for certificate tracking. This information is used both here and for constraints in `solve_network.py`.
    *   **`get_go_background_demand`**: Generates a background Guarantee of Origin (GO) electricity demand profile for each country based on historical AIB statistics.

*   **Functions for Creating the GO Layer**:
    *   **`add_virtual_carriers`**: Adds "virtual" carriers to the PyPSA network, which are duplicates of existing carriers used for tracking certificate flows.
    *   **`add_virtual_ppl`**: Adds virtual power plants to the PyPSA network as aggregate virtual generators and creates corresponding virtual buses.
    *   **`add_demand`**: Adds GoO demand to the PyPSA network by creating new buses and loads representing the certificate demand.
    *   **`add_go_market`**: Establishes a market bus for Guarantees of Origin, connecting GO supply and demand buses.
    *   **`add_virtual_storage`**: Adds virtual storage units to the PyPSA network as virtual generators, connected to national "GO Market" buses.

*   **Supporting Functions**:
    *   **`get_geo_center`**: Computes the geographic center (centroid) of countries or regions from a shapefile, used for placing virtual buses on a map.
    *   **`extract_AIB_statistics`**: Downloads, processes, and exports electricity cancellation statistics from the AIB's dataset.
    *   **`retrieve_ci_load`**: Retrieves industrial and commercial electricity load data.
    *   **`get_load_demand`**: Extracts and processes electricity load demand from the PyPSA network.

The script's overall workflow involves:

1.  **Virtual Carrier Addition**: Adds "GoO" as a carrier and creates virtual carriers based on plant groupings.
2.  **Virtual Power Plant Creation**: Groups real generators and links into virtual power plants and adds them to the network.
3.  **Demand Integration**:
    *   If enabled, calculates and adds "background" GoO demand using AIB statistics and creates a corresponding market.
    *   If enabled, calculates and adds "new" GoO demand based on load profiles and creates a corresponding market.
4.  **Virtual Storage Integration**: If national scope demands are enabled and storage carriers are defined, groups real storage units into virtual storage and adds them to the network.

Essentially, this script extends a standard PyPSA energy model to include the complexities of a Guarantee of Origin certificate system, allowing for the simulation and analysis of certificate trading and its impact on the energy system.
