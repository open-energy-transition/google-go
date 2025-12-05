"""
Color mapping utilities based on colors.csv
"""
import pandas as pd
from pathlib import Path


class ColorMapper:
    """Handles color mappings for different carriers and categories"""

    def __init__(self, colors_csv_path="../results/colors.csv"):
        self.colors_df = pd.read_csv(colors_csv_path)
        self.color_maps = self._create_color_maps()

    def _create_color_maps(self):
        """Create color dictionaries for each result category"""
        color_maps = {}

        for category in self.colors_df['Results'].unique():
            category_data = self.colors_df[self.colors_df['Results'] == category]
            color_maps[category] = dict(zip(
                category_data['carrier'],
                category_data.iloc[:, 2]  # Color column
            ))

        return color_maps

    def get_color(self, carrier, category="(a) Energy mix"):
        """Get color for a specific carrier in a category"""
        if category in self.color_maps:
            return self.color_maps[category].get(carrier, '#CCCCCC')
        return '#CCCCCC'

    def get_color_map(self, category):
        """Get entire color map for a category"""
        return self.color_maps.get(category, {})

    def get_all_carriers(self, category):
        """Get all carriers for a category"""
        if category in self.color_maps:
            return list(self.color_maps[category].keys())
        return []


def format_scenario_name(scenario_name):
    """Convert raw scenario names to human-readable format"""
    if not scenario_name:
        return scenario_name

    # Handle baseline
    if scenario_name == 'baseline':
        return 'Baseline'

    # Handle energy-match scenarios
    if 'energy-match' in scenario_name:
        if 'energy-match-25' in scenario_name:
            return 'Energy Match 25%'
        elif 'energy-match-50' in scenario_name:
            return 'Energy Match 50%'

    # Handle hourly-match scenarios
    if 'hourly-match' in scenario_name:
        parts = scenario_name.split('-')
        if len(parts) >= 3:
            # Extract CI level and percentage
            ci_level = parts[2]  # e.g., '25', '50', 'noadd'

            if len(parts) >= 4:
                percentage = parts[3]  # e.g., '90', '95'
                if len(parts) >= 5:
                    # Has second percentage like hourly-match-noadd-90-99
                    second_pct = parts[4]
                    if ci_level == 'noadd':
                        return f'Hourly {percentage}%-{second_pct}% (No Add.)'
                    else:
                        return f'Hourly {percentage}%-{second_pct}% (CI {ci_level}%)'
                else:
                    # Single percentage
                    if ci_level == 'noadd':
                        return f'Hourly {percentage}% (No Add.)'
                    else:
                        return f'Hourly {percentage}% (CI {ci_level}%)'

    # If no pattern matches, return title case
    return scenario_name.replace('-', ' ').replace('_', ' ').title()


def format_main_scenario_name(main_scenario):
    """Convert main scenario names to human-readable format"""
    mapping = {
        'CI_25': 'CI 25%',
        'CI_50': 'CI 50%',
        'CI_noadd': 'No Additionality'
    }
    return mapping.get(main_scenario, main_scenario)
