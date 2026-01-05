"""
Data loading utilities for the dashboard
Handles loading and preprocessing of consolidated results
"""
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Loads and manages data from consolidated results"""

    def __init__(self, results_dir="../results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.frontier_data = None
        self.timeseries_cache = {}  # Cache for timeseries data
        self.timeseries_metadata_cache = None  # Cache metadata

    def load_all_data(self):
        """Load consolidated data"""
        print("Loading consolidated results...")
        self.data = self.load_scenario_data()

        # Load frontier data
        print("Loading frontier data...")
        self.frontier_data = self.load_frontier_data()

        # Optionally pre-load timeseries metadata
        print("Loading timeseries metadata...")
        self.get_timeseries_metadata(None)

    def load_scenario_data(self):
        """Load data from consolidated results.csv"""
        scenario_data = {}

        # Load main consolidated results CSV
        results_file = self.results_dir / "results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file, header=[0, 1, 2], index_col=[0, 1, 2])
            scenario_data['results'] = df
            scenario_data['years'] = self._extract_years(df)
            scenario_data['scenarios'] = self._extract_scenarios(df)
            scenario_data['metrics'] = self._extract_metrics(df)
            print(f"  Found {len(scenario_data['years'])} years, {len(scenario_data['scenarios'])} scenarios, {len(scenario_data['metrics'])} metrics")
        else:
            print(f"  Warning: {results_file} not found")

        # Load country-specific results if available
        scenario_data['countries'] = {}
        for country_file in self.results_dir.glob("results-*.csv"):
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

    def get_data(self, year=None, scenario_name=None, metric=None):
        """
        Get data for specific filters

        Parameters:
        -----------
        year : int, optional
            Specific year to filter
        scenario_name : str, optional
            Specific scenario name (e.g., 'baseline', 'energy-match-25')
        metric : str, optional
            Specific metric (e.g., '(a) Energy mix')
        """
        df = self.data.get('results')
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

    def get_carriers_for_metric(self, metric):
        """Get list of carriers for a specific metric"""
        df = self.data.get('results')
        if df is None:
            return []

        try:
            # Get data for this metric (level 0 of index)
            metric_data = df.loc[pd.IndexSlice[metric, :, :], :]
            # Get unique carriers (level 2 of index)
            carriers = list(metric_data.index.get_level_values(2).unique())
            return carriers
        except Exception as e:
            print(f"Error getting carriers for {metric}: {e}")
            return []

    def get_time_series_data(self, year, scenario_name, carrier):
        """
        Get time series data for a specific carrier
        This would need to be implemented based on how time series data is stored
        """
        # Placeholder - implement based on actual data structure
        return None

    def get_summary_stats(self):
        """Get summary statistics"""
        return {
            'years': self.data.get('years', []),
            'scenarios': self.data.get('scenarios', []),
            'metrics': self.data.get('metrics', []),
            'num_countries': len(self.data.get('countries', {}))
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

    def get_frontier_data(self, year, country='EU'):
        """
        Get frontier data for a specific year and country

        Parameters:
        -----------
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

    def get_frontier_countries(self, year):
        """Get list of available countries for a given year"""
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

    # ==================== Timeseries Data Methods ====================

    def get_timeseries_metadata(self, year=None):
        """
        Get available metadata for timeseries data without loading the full file
        Returns dict with available: scenarios, countries, types, carriers
        """
        # Return cached metadata if available
        if self.timeseries_metadata_cache is not None:
            return self.timeseries_metadata_cache

        # Try parquet first (much faster), fall back to CSV
        ts_parquet = self.results_dir / "results_time_series.parquet"
        ts_csv = self.results_dir / "results_time_series.csv"

        if ts_parquet.exists():
            ts_file = ts_parquet
            use_parquet = True
        elif ts_csv.exists():
            ts_file = ts_csv
            use_parquet = False
        else:
            return {'scenarios': [], 'countries': [], 'types': [], 'carriers': []}

        try:
            print(f"Extracting metadata from timeseries file ({'parquet' if use_parquet else 'CSV'})...")

            # Collect unique values from chunks to get ALL scenarios
            all_scenarios = set()
            all_years = set()
            all_countries = set()
            all_types = set()
            all_carriers = set()

            if use_parquet:
                # Parquet has metadata in MultiIndex - just read the index
                try:
                    df = pd.read_parquet(ts_file)
                    # Metadata is in the MultiIndex
                    all_scenarios.update(df.index.get_level_values('scenario').unique())
                    all_years.update(df.index.get_level_values('year').unique().astype(int))
                    all_countries.update(df.index.get_level_values('country').unique())
                    all_types.update(df.index.get_level_values('Results').unique())
                    all_carriers.update(df.index.get_level_values('carrier').unique())
                    print(f"  Loaded metadata from parquet MultiIndex (fast!)")
                except Exception as e:
                    print(f"  Error reading parquet (falling back to CSV): {e}")
                    # Fall back to CSV
                    if ts_csv.exists():
                        ts_file = ts_csv
                        use_parquet = False
                    else:
                        raise

            if not use_parquet:
                # CSV: Read in chunks
                chunk_count = 0
                for chunk in pd.read_csv(ts_file, chunksize=100000, usecols=['scenario', 'year', 'Results', 'country', 'carrier']):
                    chunk_count += 1
                    all_scenarios.update(chunk['scenario'].unique())
                    all_years.update(chunk['year'].unique())
                    all_countries.update(chunk['country'].unique())
                    all_types.update(chunk['Results'].unique())
                    all_carriers.update(chunk['carrier'].unique())

                    # Print progress every 20 chunks
                    if chunk_count % 20 == 0:
                        print(f"  Processed {chunk_count} chunks...")

            metadata = {
                'scenarios': sorted(list(all_scenarios)),
                'years': sorted(list(all_years)),
                'countries': sorted(list(all_countries)),
                'types': sorted(list(all_types)),
                'carriers': sorted(list(all_carriers))
            }

            print(f"  Found {len(metadata['scenarios'])} scenarios, {len(metadata['years'])} years, {len(metadata['countries'])} countries")

            # Cache the metadata
            self.timeseries_metadata_cache = metadata
            return metadata
        except Exception as e:
            print(f"Error getting timeseries metadata: {e}")
            return {'scenarios': [], 'years': [], 'countries': [], 'types': [], 'carriers': []}

    def load_timeseries_data(self, year, scenarios, ts_type, country, carriers=None, time_range='week1'):
        """
        Load timeseries data for specific parameters (with caching)

        Parameters:
        - year: year to load
        - scenarios: list of scenario names
        - ts_type: Results type (e.g., 'Electricity Balance')
        - country: country name
        - carriers: optional list of carriers to filter
        - time_range: time range to load ('full', '2013-01', 'week1', etc.)

        Returns: DataFrame with timeseries data
        """
        # Create cache key
        scenarios_tuple = tuple(sorted(scenarios))
        carriers_tuple = tuple(sorted(carriers)) if carriers else None
        cache_key = (year, scenarios_tuple, ts_type, country, carriers_tuple, time_range)

        # Check cache first
        if cache_key in self.timeseries_cache:
            print(f"Returning cached timeseries data for {country} {year}")
            return self.timeseries_cache[cache_key]

        # Try parquet first (much faster), fall back to CSV
        ts_parquet = self.results_dir / "results_time_series.parquet"
        ts_csv = self.results_dir / "results_time_series.csv"

        if ts_parquet.exists():
            ts_file = ts_parquet
            use_parquet = True
        elif ts_csv.exists():
            ts_file = ts_csv
            use_parquet = False
        else:
            return None

        try:
            print(f"Loading timeseries data for {country} {year}... ({'parquet' if use_parquet else 'CSV'})")

            if use_parquet:
                # Parquet: Data is in MultiIndex format
                try:
                    df_all = pd.read_parquet(ts_file)

                    # Filter by MultiIndex levels
                    idx = pd.IndexSlice
                    df = df_all.loc[idx[scenarios, str(year), ts_type, :, country, :, :], :]

                    # Further filter by carriers if specified
                    if carriers is not None and len(carriers) > 0:
                        df = df.loc[idx[:, :, :, :, :, :, carriers], :]

                    # Reset index to make it a regular DataFrame
                    df = df.reset_index()

                    print(f"  Loaded {len(df)} rows from parquet MultiIndex (fast!)")
                except Exception as e:
                    print(f"  Error reading parquet (falling back to CSV): {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to CSV
                    if ts_csv.exists():
                        ts_file = ts_csv
                        use_parquet = False
                    else:
                        raise

            if not use_parquet:
                # CSV: Read in chunks and filter
                chunks = []
                chunk_count = 0

                for chunk in pd.read_csv(ts_file, chunksize=100000):
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print(f"  Processing chunk {chunk_count}...")

                    # Filter by parameters
                    mask = (
                        (chunk['year'] == year) &
                        (chunk['scenario'].isin(scenarios)) &
                        (chunk['Results'] == ts_type) &
                        (chunk['country'] == country)
                    )

                    if carriers is not None and len(carriers) > 0:
                        mask = mask & (chunk['carrier'].isin(carriers))

                    filtered = chunk[mask]
                    if not filtered.empty:
                        chunks.append(filtered)

                if not chunks:
                    print(f"  No data found for {country} {year}")
                    return None

                # Combine all chunks
                df = pd.concat(chunks, ignore_index=True)

            if df.empty:
                return None

            # Get timestamp columns (all columns after 'carrier')
            metadata_cols = ['scenario', 'year', 'Results', 'y_label', 'country', 'type', 'carrier']
            timestamp_cols = [col for col in df.columns if col not in metadata_cols]

            # Filter by time range if specified
            if time_range != 'full':
                timestamp_cols = self._filter_time_range(timestamp_cols, time_range)

            # Reshape data for plotting
            # Keep metadata + timestamp columns
            df = df[metadata_cols + timestamp_cols]

            result = (df, timestamp_cols)

            # Cache the result (limit cache size to 50 entries)
            if len(self.timeseries_cache) > 50:
                # Remove oldest entry
                self.timeseries_cache.pop(next(iter(self.timeseries_cache)))

            self.timeseries_cache[cache_key] = result
            print(f"  Loaded {len(df)} rows, cached for future use")

            return result

        except Exception as e:
            print(f"Error loading timeseries data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _filter_time_range(self, timestamp_cols, time_range):
        """Filter timestamp columns based on time range (handles both strings and Timestamp objects)"""
        if time_range == 'full':
            return timestamp_cols
        elif time_range == 'week1':
            # First 7 days = 7 * 8 timestamps (3-hour intervals)
            return timestamp_cols[:56]
        elif time_range == 'week_summer':
            # July first week
            result = []
            for col in timestamp_cols:
                if isinstance(col, str):
                    if col.startswith('2013-07') and int(col.split('-')[2].split()[0]) <= 7:
                        result.append(col)
                else:  # Timestamp object
                    if col.month == 7 and col.day <= 7:
                        result.append(col)
            return result
        elif time_range == 'week_winter':
            # January third week (more representative of winter)
            result = []
            for col in timestamp_cols:
                if isinstance(col, str):
                    if col.startswith('2013-01') and 15 <= int(col.split('-')[2].split()[0]) <= 21:
                        result.append(col)
                else:  # Timestamp object
                    if col.month == 1 and 15 <= col.day <= 21:
                        result.append(col)
            return result
        elif time_range.startswith('2013-'):
            # Specific month
            month = int(time_range.split('-')[1])
            result = []
            for col in timestamp_cols:
                if isinstance(col, str):
                    if col.startswith(time_range):
                        result.append(col)
                else:  # Timestamp object
                    if col.month == month:
                        result.append(col)
            return result
        else:
            return timestamp_cols
