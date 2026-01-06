# Google GO project
This repository is a soft-fork of [OET/PyPSA-Eur](https://github.com/open-energy-transition/pypsa-eur) and contains the entire project **Google GO** carried out by [Open Energy Transition (OET)](https://openenergytransition.org/) and [Google](https://about.google/), including code and visualization. The philosophy behind this repository is that no intermediary results are included, but all results are computed from raw data and code.

This repository is maintained using [OET's soft-fork strategy](https://open-energy-transition.github.io/handbook/docs/Engineering/SoftForkStrategy/). OET's primary aim is to contribute as much as possible to the open source (OS) upstream repositories. For long-term changes that cannot be directly merged upstream, the strategy organizes and maintains OET forks, ensuring they remain up-to-date and compatible with upstream, while also supporting future contributions back to the OS repositories.

OET, an international non-profit organization specializing in open energy modeling software development and support, broght its expertise to this project. The organization has a proven track record in promoting transparent, data-driven decision-making in energy policy and planning, with its software products (including PyPSA-Eur and PyPSA-Earth) used in more than 50 research and industry-related projects.

For further readings of PyPSA and PyPSA-Eur, check out:

* [PyPSA](https://docs.pypsa.org/latest/)
* [PyPSA-Eur](https://pypsa-eur.readthedocs.io/en/latest/)
* [PyPSA-Earth](https://pypsa-earth.readthedocs.io/en/latest/)

## Overview

The following chapters are available in this documentation:

### Introduction

The introduction chapter provides context for the Google-GO project and explains how to reproduce its results.

* [Background](Introduction/background.md)
* [Installation](Introduction/installation.md)
* [Run Scenarios](Introduction/run-scenarios.md)
* [Assumptions](Introduction/assumptions.md)

### Configuration

The Google-GO project utilizes a hierarchical configuration structure to manage its modeling assumptions and scenarios, building upon the foundational PyPSA-Eur framework.

*   [Project Config](Configuration/go_project_config.md)
*   [Scenarios Config](Configuration/go_project_scenarios.md)

### Feature

This chapter explains the features implemented for the Google GO project.

*   [Strip Network](Feature/strip_network.md)
*   [Add Certificate](Feature/add_certificate.md)
*   [Solve Network Constraints](Feature/solve_network_constraints.md)

### Analysis

This chapter describes the post-processing workflow for analyzing and visualizing the results of the Google-GO project. 

*   [Main Results](Analysis/main_results_generation.md)
*   [Interactive Dashboard](Analysis/interactive_dashboard.md)
*   [Interactive Dashboard (Adv)](Analysis/interactive_dashboard_advanced.md)
*   [Jupyter Notebooks](Analysis/jupyter_notebooks.md)