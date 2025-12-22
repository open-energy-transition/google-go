This section is an excerpt from the PyPSA-Eur documentation regarding installation.

# Installation

The subsequently described installation steps are demonstrated as shell commands, where the path before the `$` sign denotes the
directory in which the commands following the `$` should be entered.

## Clone the Repository

First of all, clone the [PyPSA-Eur repository](https://github.com/PyPSA/pypsa-eur) using the version control system `git` in the command line.

```bash
$ git clone https://github.com/PyPSA/pypsa-eur.git
```

## Install Python Dependencies

### Preferred method: `pixi`

PyPSA-Eur relies on a set of other Python packages to function.
We manage these using [pixi](https://pixi.sh/latest/).
Once pixi is installed, you can activate the project environment for your operating system and have access to all the PyPSA-Eur dependencies from the command line:

```bash
$ pixi shell
```

> **Tip:** You can also set up automatic shell activation in several popular editors (e.g. in [VSCode](https://pixi.sh/dev/integration/editor/vscode/) or [Zed](https://pixi.sh/dev/integration/editor/zed/)).
> Refer to the `pixi` documentation for the most up-to-date options.

> **Note:** We don't currently support linux operating systems using ARM processors since certain packages, such as `PySCIPOpt`, require being built from source.

### Legacy method: `conda`

If you cannot access `pixi` on your machine, you can also install using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) (or `mamba`/`micromamba`).
To do so, we highly recommend you install from one of our platform-specific environment files:

* For Intel/AMD processors:

  - Linux: `envs/default_linux-64.pin.txt`
  - macOS: `envs/default_osx-64.pin.txt`
  - Windows: `envs/default_win-64.pin.txt`

* For ARM processors:

  - macOS (Apple Silicon): `envs/default_osx-arm64.pin.txt`
  - Linux (ARM): Currently not supported via lock files; requires building certain packages, such as `PySCIPOpt`, from source

```bash
$ conda update conda

$ conda create -n pypsa-eur -f envs/default_linux-64.pin.txt # select the appropriate file for your platform

$ conda activate pypsa-eur
```

These platform-specific files have locked dependencies, to ensure reproducibility.
If you are having difficulties with the above files, you can also install directly from the un-locked environment YAML file (not recommended):

```bash
$ conda update conda

$ conda env create -f envs/environment.yaml

$ conda activate pypsa-eur
```

## Install a Solver

PyPSA passes the PyPSA-Eur network model to an external solver for performing the optimisation.
PyPSA is known to work with the free software

- [HiGHS](https://highs.dev/)
- [Cbc](https://projects.coin-or.org/Cbc#DownloadandInstall)
- [GLPK](https://www.gnu.org/software/glpk/) ([WinGLKP](http://winglpk.sourceforge.net/))
- [SCIP](https://scipopt.github.io/PySCIPOpt/docs/html/index.html)

and the non-free, commercial software (for some of which free academic licenses are available)

- [Gurobi](https://www.gurobi.com/documentation/quickstart.html)
- [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
- [FICO Xpress Solver](https://www.fico.com/de/products/fico-xpress-solver)

For installation instructions of these solvers for your operating system, follow the links above.
Commercial solvers such as Gurobi and CPLEX currently significantly outperform open-source solvers for large-scale problems, and
it might be the case that you can only retrieve solutions by using a commercial solver.
Nevertheless, you can still use open-source solvers for smaller problems.

See also: [Instructions how to install a solver in the documentation of PyPSA](https://pypsa.readthedocs.io/en/latest/installation.html#getting-a-solver-for-linear-optimisation)

> **Note:** The rules `:mod:`cluster_network` solves a mixed-integer quadratic optimisation problem for clustering.
> The open-source solvers HiGHS, Cbc and GlPK cannot handle this. A fallback to SCIP is implemented in this case, which is included in the standard environment specifications.
> For an open-source solver setup install for example HiGHS **and** SCIP in your `conda` environment on OSX/Linux.
> To install the default solver Gurobi, run

```bash
$ conda activate pypsa-eur
$ conda install -c gurobi gurobi"=12.0.1"
```

Additionally, you need to setup your [Gurobi license](https://www.gurobi.com/solutions/licensing/).