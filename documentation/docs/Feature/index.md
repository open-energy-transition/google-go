The following features are implemented through specialized Python scripts (`strip_network.py`, `add_certificate.py`, and parts of `solve_network.py`) developed specifically for the Google-GO project within the PyPSA-Eur workflow. These scripts extend the standard PyPSA functionalities to address the unique modeling requirements of Guarantee of Origin (GO) certificates and simplified network analysis.

### Strip Network

The `strip_network.py` script is designed to simplify complex energy system networks. Its primary purpose is to transform a comprehensive model into an electricity-only representation by removing non-electricity-related components.

### Add Certificate

The `add_certificate.py` script integrates the GO certificate system into the energy system models. It achieves this by creating "virtual" representations of power plants and storage units, and establishing a market for these certificates within the PyPSA network.

### Solve Network Constraints

The `solve_network.py` script, particularly its custom functions, introduces specialized constraints and functionalities essential for the Google-GO project. These include:

*   **Background Constraints**: Functions like `add_rps_constraints` enforce Renewable Portfolio Standard (RPS) targets at national or system-wide levels, based on TYNDP data or manual inputs.
*   **GO Constraints**: Functions such as `add_virtual_ppl_matching`, `add_buffer_matching`, and `add_virtual_storage_matching` are critical for binding the physical electricity generation and storage operations to the virtual GOs layer. These ensure that the GO market accurately reflects real-world energy flows and manages hourly matching limits.
*   **Additional Storage Constraints**: Functions like `add_storage_inverter_fix` and `add_storage_duration_fix` ensure specific behaviors for storage technologies, making certain types of batteries behave consistently as storage units for market participation.

## Table Of Content

*   [Strip Network](strip_network.md)
*   [Add Certificate](add_certificate.md)
*   [Solve Network Constraints](solve_network_constraints.md)