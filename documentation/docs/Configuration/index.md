The Google-GO project utilizes a hierarchical configuration structure to manage its modeling assumptions and scenarios, building upon the foundational PyPSA-Eur framework.

1.  **Base Configuration (`config.default.yaml`)**: This file provides the default, upstream PyPSA-Eur settings, establishing the foundational parameters for the energy system model.

2.  **Project Configuration (`config.go.yaml`)**: As detailed in [Project Config](go_project_config.md), this file overrides selected defaults from `config.default.yaml` with Google-GO project-specific settings. These modifications enable specialized modeling for Guarantee of Origin (GOs) certificates, renewable energy targets, and other project-specific features. It defines core parameters such as network nodes, foresight approach, planning horizons, and the activation of various features like the GOs market and novel technologies.

3.  **Scenario Configuration (`scenarios.go.yaml`)**: Described in [Scenarios Config](go_project_scenarios.md), this file defines a collection of specific scenarios. Each scenario in `scenarios.go.yaml` further overrides parameters from both `config.default.yaml` and `config.go.yaml` to explore different modeling assumptions, such as varying levels of Commercial & Industrial (C&I) participation in GO procurement, hourly matching requirements, and policy sensitivities. This allows for systematic exploration of different market designs and policy interactions.

This hierarchical structure ensures that the project can leverage the robust PyPSA-Eur defaults while precisely tailoring configurations for its unique research questions and scenario analyses.

## Table Of Content

*   [Project Config](go_project_config.md)
*   [Scenarios Config](go_project_scenarios.md)