# Google GO project
This repository is a soft-fork of [OET/PyPSA-Eur](https://github.com/open-energy-transition/pypsa-eur) and contains the entire project **Google GO** carried out by [Open Energy Transition (OET)](https://openenergytransition.org/) and [Google](https://about.google/), including code and visualization. The philosophy behind this repository is that no intermediary results are included, but all results are computed from raw data and code.

This repository is maintained using [OET's soft-fork strategy](https://open-energy-transition.github.io/handbook/docs/Engineering/SoftForkStrategy/). OET's primary aim is to contribute as much as possible to the open source (OS) upstream repositories. For long-term changes that cannot be directly merged upstream, the strategy organizes and maintains OET forks, ensuring they remain up-to-date and compatible with upstream, while also supporting future contributions back to the OS repositories.

OET, an international non-profit organization specializing in open energy modeling software development and support, broght its expertise to this project. The organization has a proven track record in promoting transparent, data-driven decision-making in energy policy and planning, with its software products (including PyPSA-Eur and PyPSA-Earth) used in more than 50 research and industry-related projects.

For further readings of PyPSA and PyPSA-Eur, check out:

* [PyPSA](https://docs.pypsa.org/latest/)
* [PyPSA-Eur](https://pypsa-eur.readthedocs.io/en/latest/)
* [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/)

### Introduction
(TBD: add a figure, maybe the one shown to Harry with all the European countries participation to AIB)

Guarantees of Origin (GOs) are certificates used in Europe proving that energy has been generated from renewable sources. This thus allows final customers, such as commercial and industry (C&I) ones, to buy and claim renewable energy, separating the "green" attribute from the physical electrons. Annual GOs matches renewable supply and consumption at annual level. They are the standard GOs and are issued at national level. Then, the [Association of Issuing Bodies (AIB)](https://www.aib-net.org/) coordinates the national issuing bodies, ensuring consistency across countries and facilitating cross-border trading. Instead, hourly GOs, also referred to as hourly Granular Certificates (GCs), are newer and more granular type, enabling near real-time matching between renewable supply and consumption.

This project aims to assess how annual and hourly GOs procurement can operate as an investment signal for the energy transition. This is done by comparing the system-level impacts in terms of: capacity expansion, asset-level dispatch, total system costs and emissions, GO pricing and market value. Also, the impacts are evaluated at country-level and by running several sensitivies. The latter aim to explore different: C&I demand participation, hourly matching requirements, and interaction with other market-based policies (e.g., CO2 price or renewable portfolio standards).

### Model Scope
To balance spatial and temporal resolution with computational efficiency, the model scope is defined as follows:

* **Spatial Scope**:
    * Geography: 34 countries.
    * Resolution: 39 nodes (ensuring that each country is represented by at least one node).
* **Temporal Scope**:
    * Planning horizons: from 2025 to 2040, with 5-years interval.
    * Resolution: 3-hours interval.
* **Sectoral Scope**:
The model is limited to the electricity sector, as it is the primary focus for clean procurement strategies. However, electrification of heating and hydrogen demand can be exogenously accounted for by means of dedicated configuration settings.
* **Technology Scope**:
The technologies eligible for the procurement involve all renewable energy sources, as well as short and long-duration energy storage options and clean firm technologies.