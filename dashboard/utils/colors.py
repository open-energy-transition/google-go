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
