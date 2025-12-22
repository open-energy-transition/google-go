# Google GO project
This repository is a soft-fork of [OET/PyPSA-Eur](https://github.com/open-energy-transition/pypsa-eur) and contains the entire project **Google GO** carried out by [Open Energy Transition (OET)](https://openenergytransition.org/) and [Google](https://about.google/), including code and visualization. The philosophy behind this repository is that no intermediary results are included, but all results are computed from raw data and code.

This repository is maintained using [OET's soft-fork strategy](https://open-energy-transition.github.io/handbook/docs/Engineering/SoftForkStrategy/). OET's primary aim is to contribute as much as possible to the open source (OS) upstream repositories. For long-term changes that cannot be directly merged upstream, the strategy organizes and maintains OET forks, ensuring they remain up-to-date and compatible with upstream, while also supporting future contributions back to the OS repositories.

OET, an international non-profit organization specializing in open energy modeling software development and support, broght its expertise to this project. The organization has a proven track record in promoting transparent, data-driven decision-making in energy policy and planning, with its software products (including PyPSA-Eur and PyPSA-Earth) used in more than 50 research and industry-related projects.

For further readings of PyPSA and PyPSA-Eur, check out:

* [PyPSA](https://docs.pypsa.org/latest/)
* [PyPSA-Eur](https://pypsa-eur.readthedocs.io/en/latest/)
* [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/)

## Introduction Overview

### Background

The Google-GO project investigates how annual and hourly Guarantees of Origin (GOs) procurement can act as an investment signal for the energy transition. Further details are on [Background](background.md).

### Installation

For instructions on setting up your environment, cloning the repository, and installing necessary dependencies and solvers, please refer to the [Installation](installation.md) guide. This section provides detailed steps for both preferred (`pixi`) and legacy (`conda`) Python dependency management methods, as well as guidance on installing external solvers crucial for the PyPSA-Eur network optimization.

### Running Scenarios

The [Run Scenarios](run-scenarios.md) section explains how to utilize the `run_scenarios.py` script to manage and execute different project scenarios. This command-line utility simplifies the process of selecting base configurations, specific scenarios, and execution profiles (e.g., local CPU usage or computer clusters), enabling dynamic updates to configuration files and automated `snakemake` command execution.

### Assumptions

The Google-GO project introduces key modeling assumptions that differentiate it from the standard PyPSA-Eur model to align with the project's scope. Further details are on [Assumptions](assumptions.md).

## Table Of Content

* [Background](background.md)
* [Installation](installation.md)
* [Run Scenarios](run-scenarios.md)
* [Assumptions](assumptions.md)
