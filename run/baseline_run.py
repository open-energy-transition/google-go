# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os
import shutil

import yaml

# ---------------------- Utility Functions ----------------------


def deep_update(original, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            deep_update(original[key], value)
        else:
            original[key] = value


def select_scenario(scenarios, name="scenarios"):
    print(f"Available {name}:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    if name == "scenarios":
        print("(Scenarios cannot include 'baseline' in their name)")
    elif name == "baseline":
        print("(Baselines must include 'baseline' in their name)")
    print("(You can also select the list number)")

    while True:
        answer = input("Select a scenario (or press Enter to cancel): ").strip()
        if not answer:
            return None
        if answer in scenarios:
            print(f"'{answer}' is selected.")
            return answer
        elif answer.isdigit():
            scenario_selected = scenarios[int(answer) - 1]
            print(f"'{scenario_selected}' is selected.")
            return scenario_selected
        else:
            print(f"'{answer}' not found in the list. Please try again.")


def select_profile():
    while True:
        answer = input("Is this in a computer cluster [y/n]?: ").strip().lower()
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return "--profile slurm"
        elif answer in ["n", "no"]:
            cpu = (
                input("How many CPUs do you want to use (all or a number)? ")
                .strip()
                .lower()
            )
            if cpu == "all":
                return "-call"
            elif cpu.isdigit():
                return f"-c{cpu}"
        print(f"'{answer}' is not a valid option. Please try again.")


def select_multiruns():
    while True:
        answer = (
            input("is there more runs that you want to make [y/n]?: ").strip().lower()
        )
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        print(f"'{answer}' is not a valid option. Please try again.")


def duplicate_run_delete(scenario_name, selected_baseline, selected_profile):
    # Prepare resources
    new_folder = f"resources/{scenario_name}"
    old_folder = f"resources/{selected_baseline}"

    year = 2025 if "2025" in scenario_name else 2030

    files_to_copy = [f"costs_{year}.csv", f"networks/base_s_39___{year}_brownfield.nc"]

    os.makedirs(os.path.join(new_folder, "networks"), exist_ok=True)

    for f in files_to_copy:
        old_path = os.path.join(old_folder, f)
        new_path = os.path.join(new_folder, f)
        if os.path.isfile(old_path):
            shutil.copy(old_path, new_path)
            print(f"Copied {old_path} to {new_path}")
        else:
            print(f"WARNING: File not found: {old_path}")

    # Run snakemake
    run_cmd = f"snakemake {selected_profile} solve_sector_networks --configfile run/config.go_temp.yaml --rerun-trigger mtime"
    os.system(run_cmd + " --touch")
    os.system(run_cmd)

    # Clean up
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
        print(f"Deleted folder: {new_folder}")
    else:
        print(f"Folder does not exist (already removed?): {new_folder}")

    temp_file = "run/config.go_temp.yaml"

    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Deleted file: {temp_file}")
    else:
        print(f"File does not exist: {temp_file}")


# ---------------------- Main Script ----------------------


def main():
    scenario_yaml = ["scenarios.go-1H.yaml", "scenarios.go-3H.yaml"]

    selected_scenario_yaml = select_scenario(scenario_yaml, name="time resolution")
    if not selected_scenario_yaml:
        print("\nOperation cancelled by user.")
        return

    print("\n=================================================================")
    # Load scenario configuration
    with open(f"config/{selected_scenario_yaml}") as file:
        config_s = yaml.safe_load(file)

    # Extract scenarios with '--'
    baseline_list = [key for key in config_s]

    scenario_pair = []
    while True:
        # User input
        selected_baseline = select_scenario(baseline_list, name="baseline")
        if not selected_baseline:
            print("\nOperation cancelled by user.")
            return

        scenario_pair += [selected_baseline]

        print("\nFinal selection:")
        print("Baseline:")
        for selected_baseline in scenario_pair:
            print(f"{selected_baseline}")

        selected_multiruns = select_multiruns()
        if not selected_multiruns:
            break

    print("\n=================================================================")
    selected_profile = select_profile()
    if not selected_profile:
        print("\nOperation cancelled by user.")
        return

    print(f"Profile: {selected_profile}")

    # Iterate runs
    for selected_baseline in scenario_pair:
        print("\n=================================================================")
        print("Currently running:")
        print(f"Baseline: {selected_baseline}")

        with open("config/config.go.yaml") as file:
            config = yaml.safe_load(file)

        deep_update(config, config_s[selected_baseline])

        config_update = {
            "run": {"name": selected_baseline, "scenarios": {"enable": False}}
        }

        deep_update(config, config_update)

        # Write temp config
        with open("run/config.go_temp.yaml", "w") as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        run_cmd = f"snakemake {selected_profile} solve_sector_networks --configfile run/config.go_temp.yaml"
        os.system(run_cmd)

        temp_file = "run/config.go_temp.yaml"

        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Deleted file: {temp_file}")
        else:
            print(f"File does not exist: {temp_file}")


if __name__ == "__main__":
    main()
