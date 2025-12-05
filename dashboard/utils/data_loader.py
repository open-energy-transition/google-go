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
        self.frontier_data = None

    def load_all_data(self):
        """Load data from all scenarios"""
        for scenario in self.scenarios:
            print(f"Loading {scenario}...")
            self.data[scenario] = self.load_scenario_data(scenario)

        # Load frontier data
        print("Loading frontier data...")
        self.frontier_data = self.load_frontier_data()

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

    def load_frontier_data(self):
        """Load the frontier results from results_frontier.csv"""
        frontier_file = self.results_dir / "results_frontier.csv"
        if not frontier_file.exists():
            print("  Warning: results_frontier.csv not found")
            return None

        try:
            # Read the CSV
            df = pd.read_csv(frontier_file)

            # The column headers (except first) are scenario names (but pandas adds .1, .2, etc. for duplicates)
            # Row 0: years
            # Row 1: countries
            # Row 2+: data points (frontier values)

            # Extract metadata from first 2 rows
            years_row = df.iloc[0, 1:]  # Skip first column 'scenario'
            countries_row = df.iloc[1, 1:]

            # Build column multi-index from scenario (column name), year, country
            columns_tuples = []
            for col_name in df.columns[1:]:  # Skip first column 'scenario'
                col_idx = df.columns.get_loc(col_name)
                year = df.iloc[0, col_idx]
                country = df.iloc[1, col_idx]
                # Remove pandas-added suffixes (.1, .2, etc.) from scenario name
                scenario = col_name.split('.')[0] if '.' in col_name else col_name
                columns_tuples.append((scenario, str(year), str(country)))

            # Create new dataframe with multi-index columns, starting from row 2
            frontier_df = pd.DataFrame(
                df.iloc[2:, 1:].values,  # Skip first 2 rows (metadata) and first column
                columns=pd.MultiIndex.from_tuples(columns_tuples, names=['scenario', 'year', 'country'])
            )

            # Convert to numeric
            frontier_df = frontier_df.apply(pd.to_numeric, errors='coerce')

            print(f"  Loaded frontier data: {frontier_df.shape[0]} points, {len(frontier_df.columns)} scenario-year-country combinations")
            return frontier_df

        except Exception as e:
            print(f"  Error loading frontier data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_frontier_data(self, main_scenario, year, country='EU'):
        """
        Get frontier data for a specific main scenario, year, and country

        Parameters:
        -----------
        main_scenario : str
            One of 'CI_25', 'CI_50', 'CI_noadd' (not currently used for filtering)
        year : int or str
            Year to filter
        country : str
            Country/region name (default: 'EU' for system level)

        Returns:
        --------
        dict: Dictionary mapping scenario names to frontier arrays
        """
        if self.frontier_data is None:
            return {}

        try:
            # Get all columns matching the year and country using idx slicing
            idx = pd.IndexSlice

            # Filter by year and country
            matching_data = self.frontier_data.loc[:, idx[:, str(year), str(country)]]

            # Extract data for each scenario
            result = {}
            for col in matching_data.columns:
                scenario, yr, ctry = col
                # Get the data column and remove NaN values
                data = matching_data[col].dropna().values
                if len(data) > 0:
                    result[scenario] = data

            return result

        except Exception as e:
            print(f"Error getting frontier data: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_frontier_countries(self, main_scenario, year):
        """Get list of available countries for a given main scenario and year"""
        if self.frontier_data is None:
            return []

        try:
            countries = set()
            for col in self.frontier_data.columns:
                scenario, yr, ctry = col
                if str(yr) == str(year):
                    countries.add(str(ctry))

            # Return sorted list with EU first
            countries_list = sorted(list(countries))
            if 'EU' in countries_list:
                countries_list.remove('EU')
                countries_list.insert(0, 'EU')

            return countries_list

        except Exception as e:
            print(f"Error getting frontier countries: {e}")
            return []
