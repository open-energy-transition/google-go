"""
Data loading utilities for the dashboard
Handles loading and preprocessing of results from CI_25, CI_50, and CI_noadd
"""
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Loads and manages data from different scenario results"""

    def __init__(self, results_dir="../results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.scenarios = ['CI_25', 'CI_50', 'CI_noadd']

    def load_all_data(self):
        """Load data from all scenarios"""
        for scenario in self.scenarios:
            print(f"Loading {scenario}...")
            self.data[scenario] = self.load_scenario_data(scenario)

    def load_scenario_data(self, scenario):
        """Load data for a specific scenario"""
        scenario_dir = self.results_dir / scenario
        scenario_data = {}

        # Load main results CSV
        results_file = scenario_dir / "results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file, header=[0, 1, 2], index_col=[0, 1, 2])
            scenario_data['results'] = df
            scenario_data['years'] = self._extract_years(df)
            scenario_data['scenarios'] = self._extract_scenarios(df)
            scenario_data['metrics'] = self._extract_metrics(df)
            print(f"  Found {len(scenario_data['years'])} years, {len(scenario_data['scenarios'])} scenarios, {len(scenario_data['metrics'])} metrics")

        # Load country-specific results if available
        scenario_data['countries'] = {}
        for country_file in scenario_dir.glob("results-*.csv"):
            country = country_file.stem.replace('results-', '')
            df_country = pd.read_csv(country_file, header=[0, 1, 2], index_col=[0, 1, 2])
            scenario_data['countries'][country] = df_country

        return scenario_data

    def _extract_years(self, df):
        """Extract available years from the dataframe"""
        if df is not None and not df.empty:
            years = df.columns.get_level_values('year').unique()
            # Convert to int, filtering out any non-numeric values
            return sorted([int(y) for y in years if str(y).isdigit()])
        return []

    def _extract_scenarios(self, df):
        """Extract available scenarios from the dataframe"""
        if df is not None and not df.empty:
            return list(df.columns.get_level_values('scenario').unique())
        return []

    def _extract_metrics(self, df):
        """Extract available metrics (result types) from the dataframe"""
        if df is not None and not df.empty:
            # Get first level of index (the metric categories)
            return list(df.index.get_level_values(0).unique())
        return []

    def get_data(self, scenario, year=None, scenario_name=None, metric=None):
        """
        Get data for specific filters

        Parameters:
        -----------
        scenario : str
            One of 'CI_25', 'CI_50', 'CI_noadd'
        year : int, optional
            Specific year to filter
        scenario_name : str, optional
            Specific scenario name (e.g., 'baseline', 'energy-match-25')
        metric : str, optional
            Specific metric (e.g., '(a) Energy mix')
        """
        if scenario not in self.data:
            return None

        df = self.data[scenario].get('results')
        if df is None:
            return None

        # Apply filters using IndexSlice for better MultiIndex handling
        idx = pd.IndexSlice

        # Filter columns (year, scenario)
        if year is not None and scenario_name is not None:
            # Filter both year and scenario
            try:
                df = df.loc[:, idx[str(year), scenario_name, :]]
            except KeyError:
                return None
        elif year is not None:
            # Filter only year
            try:
                df = df.loc[:, idx[str(year), :, :]]
            except KeyError:
                return None
        elif scenario_name is not None:
            # Filter only scenario
            try:
                df = df.loc[:, idx[:, scenario_name, :]]
            except KeyError:
                return None

        # Filter rows (metric)
        if metric is not None:
            try:
                df = df.loc[idx[metric, :, :], :]
            except KeyError:
                return None

        return df

    def get_carriers_for_metric(self, scenario, metric):
        """Get list of carriers for a specific metric"""
        df = self.data[scenario].get('results')
        if df is None:
            return []

        try:
            # Get data for this metric (level 0 of index)
            metric_data = df.loc[pd.IndexSlice[metric, :, :], :]
            # Get unique carriers (level 2 of index)
            carriers = list(metric_data.index.get_level_values(2).unique())
            return carriers
        except Exception as e:
            print(f"Error getting carriers for {scenario} {metric}: {e}")
            return []

    def get_time_series_data(self, scenario, year, scenario_name, carrier):
        """
        Get time series data for a specific carrier
        This would need to be implemented based on how time series data is stored
        """
        # Placeholder - implement based on actual data structure
        return None

    def get_summary_stats(self, scenario):
        """Get summary statistics for a scenario"""
        if scenario not in self.data:
            return {}

        data = self.data[scenario]
        return {
            'years': data.get('years', []),
            'scenarios': data.get('scenarios', []),
            'metrics': data.get('metrics', []),
            'num_countries': len(data.get('countries', {}))
        }
