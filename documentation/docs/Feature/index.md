This chapter explains the features implemented for the Google GO project. These scripts extend the standard PyPSA functionalities to address the unique modeling requirements of Guarantee of Origin (GO) certificates and network simplification.

### Strip Network

As detailed in the [Strip Network](strip_network.md), this script transforms complex models into electricity-only networks by removing non-electric components.

### Add Certificate

The structure of the "virtual" layer are described in [Add Certificate](add_certificate.md). This script creates "virtual" power plants and storage units to establish the GO certificate market within the network.

### Solve Network Constraints

[Solve Network Constraints](solve_network_constraints.md) are functions within `solve_network.py` script that implements the specialized logic required to link physical energy flows with the virtual GO layer.

## Table Of Content

*   [Strip Network](strip_network.md)
*   [Add Certificate](add_certificate.md)
*   [Solve Network Constraints](solve_network_constraints.md)