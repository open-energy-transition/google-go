"""
Comprehensive Timeseries Analysis: Hourly Patterns Across Scenarios
Analyzing actual hourly matching behavior, storage cycling, and procurement strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 100)
print("TIMESERIES DEEP-DIVE ANALYSIS")
print("Hourly Patterns, Storage Cycling, and Procurement Strategies")
print("=" * 100)

def load_timeseries_data():
    """Load parquet timeseries data"""
    print("\n" + "=" * 100)
    print("LOADING TIMESERIES DATA")
    print("=" * 100)

    try:
        df = pd.read_parquet('../results/results_timeseries.parquet')
        print(f"\nLoaded timeseries data: {df.shape}")
        print(f"Index levels: {df.index.names}")
        print(f"Scenarios: {len(df.index.get_level_values('scenario').unique())}")
        print(f"Countries: {len(df.index.get_level_values('country').unique())}")

        # Get timestamp columns
        timestamp_cols = [col for col in df.columns if isinstance(col, pd.Timestamp)]
        print(f"Hourly timestamps: {len(timestamp_cols)}")

        return df, timestamp_cols
    except Exception as e:
        print(f"Error loading timeseries: {e}")
        return None, None


def analyze_hourly_matching_patterns(df, timestamp_cols):
    """Analyze how well each scenario achieves hourly matching"""
    print("\n" + "=" * 100)
    print("ANALYSIS 1: HOURLY MATCHING ACHIEVEMENT")
    print("=" * 100)

    # Sample key scenarios
    scenarios = ['baseline', 'hourly-match-50-99', 'hourly-match-noadd-50-99']
    countries_sample = ['EU', 'Germany', 'Norway', 'Ireland']

    results = {}

    for scenario in scenarios:
        try:
            scenario_data = df.xs(scenario, level='scenario', drop_level=False)

            for country in countries_sample:
                try:
                    country_data = scenario_data.xs(country, level='country', drop_level=False)

                    # Get electricity balance (net supply vs demand)
                    elec_balance = country_data.xs('Electricity Balance', level='Results', drop_level=False)

                    if len(elec_balance) > 0:
                        # Extract hourly values
                        hourly_values = elec_balance[timestamp_cols].values.flatten()
                        hourly_values = hourly_values[~np.isnan(hourly_values)]

                        if len(hourly_values) > 0:
                            # Calculate matching metrics
                            surplus_hours = np.sum(hourly_values > 0)
                            deficit_hours = np.sum(hourly_values < 0)
                            total_hours = len(hourly_values)

                            # How much surplus/deficit
                            total_surplus = np.sum(hourly_values[hourly_values > 0])
                            total_deficit = np.abs(np.sum(hourly_values[hourly_values < 0]))

                            # Matching percentage
                            if total_surplus + total_deficit > 0:
                                match_pct = (total_surplus / (total_surplus + total_deficit)) * 100
                            else:
                                match_pct = 100

                            results[(scenario, country)] = {
                                'surplus_hours': surplus_hours,
                                'deficit_hours': deficit_hours,
                                'surplus_pct': (surplus_hours / total_hours) * 100,
                                'total_surplus_GWh': total_surplus,
                                'total_deficit_GWh': total_deficit,
                                'matching_pct': match_pct
                            }
                except:
                    continue
        except:
            continue

    # Print results
    print("\nHOURLY MATCHING PERFORMANCE:")
    print("-" * 100)
    print(f"{'Scenario':<30} {'Country':<15} {'Hours Surplus':<15} {'Hours Deficit':<15} {'Match %':<10}")
    print("-" * 100)

    for (scenario, country), data in sorted(results.items()):
        print(f"{scenario:<30} {country:<15} {data['surplus_hours']:<15.0f} "
              f"{data['deficit_hours']:<15.0f} {data['matching_pct']:<10.2f}")

    return results


def analyze_storage_cycling_patterns(df, timestamp_cols):
    """Analyze battery vs LDES cycling patterns"""
    print("\n" + "=" * 100)
    print("ANALYSIS 2: STORAGE CYCLING PATTERNS")
    print("=" * 100)

    scenarios = ['baseline', 'hourly-match-50-99', 'hourly-match-noadd-50-99']
    storage_types = ['battery', 'H2']  # Looking for these carriers

    results = {}

    for scenario in scenarios:
        try:
            scenario_data = df.xs(scenario, level='scenario', drop_level=False)

            # Get EU-level data
            try:
                eu_data = scenario_data.xs('EU', level='country', drop_level=False)

                # Look for storage dispatch
                for storage_type in storage_types:
                    try:
                        # Find rows with this carrier
                        storage_rows = eu_data[eu_data.index.get_level_values('carrier').str.contains(storage_type, case=False, na=False)]

                        if len(storage_rows) > 0:
                            # Get hourly dispatch
                            hourly_dispatch = storage_rows[timestamp_cols].values.flatten()
                            hourly_dispatch = hourly_dispatch[~np.isnan(hourly_dispatch)]

                            if len(hourly_dispatch) > 0:
                                # Calculate cycling metrics
                                charge_events = np.sum(np.diff(hourly_dispatch) > 0)
                                discharge_events = np.sum(np.diff(hourly_dispatch) < 0)

                                # Average cycle length
                                if charge_events > 0:
                                    avg_cycle_length = len(hourly_dispatch) / charge_events
                                else:
                                    avg_cycle_length = 0

                                # Utilization
                                utilization = (np.abs(hourly_dispatch).mean() / (np.abs(hourly_dispatch).max() + 1e-6)) * 100

                                results[(scenario, storage_type)] = {
                                    'charge_events': charge_events,
                                    'discharge_events': discharge_events,
                                    'avg_cycle_hours': avg_cycle_length,
                                    'utilization_pct': utilization,
                                    'max_discharge': np.max(hourly_dispatch),
                                    'max_charge': np.abs(np.min(hourly_dispatch))
                                }
                    except:
                        continue
            except:
                continue
        except:
            continue

    print("\nSTORAGE CYCLING CHARACTERISTICS:")
    print("-" * 100)
    print(f"{'Scenario':<30} {'Storage':<10} {'Charge Events':<15} {'Avg Cycle (h)':<15} {'Utilization %':<15}")
    print("-" * 100)

    for (scenario, storage), data in sorted(results.items()):
        print(f"{scenario:<30} {storage:<10} {data['charge_events']:<15.0f} "
              f"{data['avg_cycle_hours']:<15.1f} {data['utilization_pct']:<15.2f}")

    return results


def analyze_seasonal_patterns(df, timestamp_cols):
    """Analyze seasonal differences between scenarios"""
    print("\n" + "=" * 100)
    print("ANALYSIS 3: SEASONAL PATTERN DIFFERENCES")
    print("=" * 100)

    scenarios = ['baseline', 'hourly-match-50-99']

    seasonal_results = {}

    for scenario in scenarios:
        try:
            scenario_data = df.xs(scenario, level='scenario', drop_level=False)

            # Get EU-level electricity balance
            try:
                eu_data = scenario_data.xs('EU', level='country', drop_level=False)
                balance_data = eu_data.xs('Electricity Balance', level='Results', drop_level=False)

                if len(balance_data) > 0:
                    # Group by month
                    monthly_means = {}
                    for col in timestamp_cols:
                        month = col.month
                        value = balance_data[col].values
                        if len(value) > 0 and not np.isnan(value[0]):
                            if month not in monthly_means:
                                monthly_means[month] = []
                            monthly_means[month].append(value[0])

                    # Calculate stats per month
                    monthly_stats = {}
                    for month, values in monthly_means.items():
                        if len(values) > 0:
                            monthly_stats[month] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values)
                            }

                    seasonal_results[scenario] = monthly_stats
            except:
                continue
        except:
            continue

    # Print seasonal patterns
    print("\nSEASONAL PATTERNS (Monthly Electricity Balance):")
    print("-" * 100)

    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for scenario, monthly_data in seasonal_results.items():
        print(f"\n{scenario}:")
        print(f"{'Month':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 60)
        for month in sorted(monthly_data.keys()):
            stats = monthly_data[month]
            month_name = months_names[month-1]
            print(f"{month_name:<8} {stats['mean']:>11.2f} {stats['std']:>11.2f} "
                  f"{stats['min']:>11.2f} {stats['max']:>11.2f}")

    # Calculate winter vs summer difference
    for scenario, monthly_data in seasonal_results.items():
        winter_months = [1, 2, 12]
        summer_months = [6, 7, 8]

        winter_mean = np.mean([monthly_data[m]['mean'] for m in winter_months if m in monthly_data])
        summer_mean = np.mean([monthly_data[m]['mean'] for m in summer_months if m in monthly_data])

        seasonal_swing = winter_mean - summer_mean

        print(f"\n{scenario} Seasonal Swing: {seasonal_swing:.2f} GW (winter-summer)")

    return seasonal_results


def analyze_curtailment_patterns(df, timestamp_cols):
    """Analyze when and how much curtailment occurs"""
    print("\n" + "=" * 100)
    print("ANALYSIS 4: CURTAILMENT PATTERNS")
    print("=" * 100)

    scenarios = ['baseline', 'hourly-match-50-99']

    curtailment_results = {}

    for scenario in scenarios:
        try:
            scenario_data = df.xs(scenario, level='scenario', drop_level=False)

            # Look for solar/wind generation
            carriers = ['solar', 'onwind', 'offwind']

            for carrier in carriers:
                try:
                    # Find data for this carrier
                    carrier_data = scenario_data[scenario_data.index.get_level_values('carrier').str.contains(carrier, case=False, na=False)]

                    if len(carrier_data) > 0:
                        # Sum across all rows with this carrier
                        hourly_gen = []
                        for idx in carrier_data.index:
                            row_vals = carrier_data.loc[idx, timestamp_cols].values
                            if len(hourly_gen) == 0:
                                hourly_gen = row_vals
                            else:
                                hourly_gen = hourly_gen + row_vals

                        hourly_gen = np.array(hourly_gen)
                        hourly_gen = hourly_gen[~np.isnan(hourly_gen)]

                        if len(hourly_gen) > 0:
                            # Hours with significant generation (>10% of max)
                            threshold = np.max(hourly_gen) * 0.1
                            active_hours = np.sum(hourly_gen > threshold)

                            # Variability
                            cv = np.std(hourly_gen) / (np.mean(hourly_gen) + 1e-6)

                            curtailment_results[(scenario, carrier)] = {
                                'total_generation_TWh': np.sum(hourly_gen) / 1000,
                                'active_hours': active_hours,
                                'active_pct': (active_hours / len(hourly_gen)) * 100,
                                'coef_var': cv,
                                'max_output_GW': np.max(hourly_gen)
                            }
                except:
                    continue
        except:
            continue

    print("\nGENERATION PATTERNS BY CARRIER:")
    print("-" * 100)
    print(f"{'Scenario':<25} {'Carrier':<10} {'Total TWh':<12} {'Active Hours':<12} {'CV':<10}")
    print("-" * 100)

    for (scenario, carrier), data in sorted(curtailment_results.items()):
        print(f"{scenario:<25} {carrier:<10} {data['total_generation_TWh']:>11.2f} "
              f"{data['active_hours']:>11.0f} {data['coef_var']:>9.3f}")

    return curtailment_results


def calculate_optimal_procurement_strategy():
    """Calculate optimal procurement strategy based on all analyses"""
    print("\n" + "=" * 100)
    print("STRATEGIC ANALYSIS: OPTIMAL PROCUREMENT STRATEGY")
    print("=" * 100)

    # Load frontier data for cost comparison
    df_frontier = pd.read_csv('../results/results_frontier.csv')
    years = df_frontier.iloc[0, 1:].values
    countries = df_frontier.iloc[1, 1:].values
    df_numeric = df_frontier.iloc[2:, 1:].copy()
    df_numeric.columns = df_frontier.columns[1:]
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Parse scenario info
    scenario_info = []
    for idx, name in enumerate(df_numeric.columns):
        base_name = name.split('.')[0] if '.' in name else name

        if 'baseline' in base_name:
            scenario_type = 'baseline'
        elif 'hourly-match-50' in base_name:
            scenario_type = 'hourly-match-50'
        elif 'hourly-match-25' in base_name:
            scenario_type = 'hourly-match-25'
        elif 'noadd' in base_name:
            scenario_type = 'noadd'
        else:
            scenario_type = 'other'

        scenario_info.append({
            'scenario_name': name,
            'scenario_type': scenario_type,
            'year': years[idx],
            'country': countries[idx]
        })

    meta_df = pd.DataFrame(scenario_info)

    # Calculate cost per country for baseline vs hourly matching
    countries_unique = ['EU', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands',
                       'Norway', 'Sweden', 'Denmark', 'Ireland', 'Luxembourg']

    print("\nCOST-BENEFIT BY COUNTRY (2030):")
    print("-" * 100)
    print(f"{'Country':<15} {'Baseline':<12} {'HM-25%':<12} {'HM-50%':<12} {'Best Strategy':<20}")
    print("-" * 100)

    recommendations = {}

    for country in countries_unique:
        baseline_mask = (meta_df['scenario_type'] == 'baseline') & (meta_df['country'] == country) & (meta_df['year'] == '2030')
        hm25_mask = (meta_df['scenario_type'] == 'hourly-match-25') & (meta_df['country'] == country) & (meta_df['year'] == '2030')
        hm50_mask = (meta_df['scenario_type'] == 'hourly-match-50') & (meta_df['country'] == country) & (meta_df['year'] == '2030')

        baseline_scenarios = meta_df[baseline_mask]['scenario_name'].tolist()
        hm25_scenarios = meta_df[hm25_mask]['scenario_name'].tolist()
        hm50_scenarios = meta_df[hm50_mask]['scenario_name'].tolist()

        if len(baseline_scenarios) > 0:
            baseline_cost = df_numeric[baseline_scenarios].mean(axis=1).mean()

            hm25_cost = df_numeric[hm25_scenarios].mean(axis=1).mean() if len(hm25_scenarios) > 0 else baseline_cost
            hm50_cost = df_numeric[hm50_scenarios].mean(axis=1).mean() if len(hm50_scenarios) > 0 else baseline_cost

            # Determine best strategy
            costs = {'baseline': baseline_cost, 'HM-25%': hm25_cost, 'HM-50%': hm50_cost}
            best = min(costs, key=costs.get)

            # Calculate benefit
            benefit = ((costs[best] - baseline_cost) / baseline_cost) * 100

            recommendations[country] = {
                'best_strategy': best,
                'benefit_pct': benefit,
                'baseline': baseline_cost,
                'hm25': hm25_cost,
                'hm50': hm50_cost
            }

            print(f"{country:<15} {baseline_cost:>11.2f} {hm25_cost:>11.2f} {hm50_cost:>11.2f} "
                  f"{best:<20}")

    # Rank countries by GO market benefit
    print("\n\nCOUNTRIES RANKED BY GO MARKET BENEFIT:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Country':<15} {'Best Strategy':<15} {'Benefit %':<12} {'Interpretation'}")
    print("-" * 100)

    ranked = sorted(recommendations.items(), key=lambda x: abs(x[1]['benefit_pct']), reverse=True)

    for rank, (country, data) in enumerate(ranked[:10], 1):
        if data['benefit_pct'] < 0:
            interpretation = "High cost reduction potential"
        elif data['benefit_pct'] > 5:
            interpretation = "High cost increase"
        else:
            interpretation = "Moderate impact"

        print(f"{rank:<6} {country:<15} {data['best_strategy']:<15} {data['benefit_pct']:>11.2f} {interpretation}")

    return recommendations


def main():
    """Run comprehensive timeseries analysis"""

    # Load data
    df, timestamp_cols = load_timeseries_data()

    if df is not None and timestamp_cols is not None:
        # Run timeseries analyses
        matching_results = analyze_hourly_matching_patterns(df, timestamp_cols)
        storage_results = analyze_storage_cycling_patterns(df, timestamp_cols)
        seasonal_results = analyze_seasonal_patterns(df, timestamp_cols)
        curtailment_results = analyze_curtailment_patterns(df, timestamp_cols)

    # Strategic analysis (doesn't require timeseries)
    recommendations = calculate_optimal_procurement_strategy()

    print("\n" + "=" * 100)
    print("TIMESERIES ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
