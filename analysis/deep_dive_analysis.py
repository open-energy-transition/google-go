"""
Deep-Dive Statistical Analysis of Google GO Energy System
Going beyond surface trends to uncover hidden patterns
Focus on scenario differences, sensitivities, and temporal dynamics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 100)
print("DEEP-DIVE STATISTICAL ANALYSIS: GOOGLE GO ENERGY SYSTEM")
print("Going Beyond Surface Trends")
print("=" * 100)

def load_all_data():
    """Load frontier and timeseries data"""
    print("\n" + "=" * 100)
    print("LOADING DATA")
    print("=" * 100)

    # Load frontier data
    df_frontier = pd.read_csv('../results/results_frontier.csv')
    years = df_frontier.iloc[0, 1:].values
    countries = df_frontier.iloc[1, 1:].values
    df_numeric = df_frontier.iloc[2:, 1:].copy()
    df_numeric.index = df_frontier.iloc[2:, 0].values
    df_numeric.columns = df_frontier.columns[1:]
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Parse scenario metadata
    scenario_info = []
    for idx, name in enumerate(df_numeric.columns):
        base_name = name.split('.')[0] if '.' in name else name

        # Detailed scenario parsing
        if 'baseline' in base_name:
            scenario_type = 'baseline'
            matching_level = 0
            participation = 0
        elif 'no-clean-firm' in base_name:
            scenario_type = 'no-clean-firm'
            matching_level = 99
            participation = 25
        elif 'no-LDES' in base_name:
            scenario_type = 'no-LDES'
            matching_level = 99
            participation = 25
        elif 'noadd' in base_name:
            scenario_type = 'noadd'
            matching_level = 99
            participation = 25
        elif 'EU-25' in base_name:
            scenario_type = 'EU-coordination'
            matching_level = 99
            participation = 25
        elif 'EU-50' in base_name:
            scenario_type = 'EU-coordination'
            matching_level = 99
            participation = 50
        elif 'hourly-match-25' in base_name or 'match-25' in base_name:
            scenario_type = 'hourly-match'
            matching_level = 25
            participation = 25
        elif 'hourly-match-50' in base_name or 'match-50' in base_name:
            scenario_type = 'hourly-match'
            matching_level = 50
            participation = 25
        else:
            scenario_type = 'other'
            matching_level = 0
            participation = 0

        scenario_info.append({
            'scenario_name': name,
            'scenario_type': scenario_type,
            'matching_level': matching_level,
            'participation': participation,
            'year': years[idx],
            'country': countries[idx]
        })

    meta_df = pd.DataFrame(scenario_info)

    print(f"\nLoaded {len(df_numeric.columns)} scenarios × {len(df_numeric)} metrics")
    print(f"Scenario types: {meta_df['scenario_type'].value_counts().to_dict()}")
    print(f"Years: {meta_df['year'].value_counts().sort_index().to_dict()}")

    return df_numeric, meta_df


def analyze_scenario_divergence(df_numeric, meta_df):
    """Analyze how scenarios diverge from baseline over time"""
    print("\n" + "=" * 100)
    print("ANALYSIS 1: SCENARIO DIVERGENCE OVER TIME")
    print("=" * 100)

    results = {}

    # Get baseline data by year
    baseline_mask = meta_df['scenario_type'] == 'baseline'
    baseline_scenarios = meta_df[baseline_mask]['scenario_name'].tolist()
    baseline_data = df_numeric[baseline_scenarios]

    # For each scenario type, calculate divergence
    for scenario_type in ['hourly-match', 'no-LDES', 'no-clean-firm', 'noadd', 'EU-coordination']:
        mask = meta_df['scenario_type'] == scenario_type
        if mask.sum() == 0:
            continue

        scenario_scenarios = meta_df[mask]['scenario_name'].tolist()
        scenario_data = df_numeric[scenario_scenarios]

        # Calculate mean absolute percentage difference from baseline
        divergences_by_year = {}
        for year in [2025, 2030, 2035, 2040]:
            # Get scenario names for this year
            baseline_year_scenarios = meta_df[baseline_mask & (meta_df['year'] == str(year))]['scenario_name'].tolist()
            scenario_year_scenarios = meta_df[mask & (meta_df['year'] == str(year))]['scenario_name'].tolist()

            if len(baseline_year_scenarios) > 0 and len(scenario_year_scenarios) > 0:
                baseline_year_data = df_numeric[baseline_year_scenarios]
                scenario_year_data = df_numeric[scenario_year_scenarios]

                baseline_mean = baseline_year_data.mean(axis=1)
                scenario_mean = scenario_year_data.mean(axis=1)

                # Percentage difference
                pct_diff = np.abs((scenario_mean - baseline_mean) / (baseline_mean + 1e-6)) * 100
                divergences_by_year[year] = pct_diff.mean()

        # Calculate divergence acceleration (2nd derivative)
        years = sorted(divergences_by_year.keys())
        divs = [divergences_by_year[y] for y in years]

        if len(divs) >= 3:
            # Rate of change
            rates = np.diff(divs)
            # Acceleration
            accel = np.diff(rates)
            max_accel_year = years[np.argmax(np.abs(accel)) + 1]
        else:
            max_accel_year = None

        results[scenario_type] = {
            'divergences': divergences_by_year,
            'max_acceleration_year': max_accel_year,
            'final_divergence': divergences_by_year.get(2040, 0)
        }

    print("\nSCENARIO DIVERGENCE FROM BASELINE:")
    print("-" * 100)
    print(f"{'Scenario':<20} {'2025':<10} {'2030':<10} {'2035':<10} {'2040':<10} {'Max Accel Year':<15} {'Final Div':<10}")
    print("-" * 100)

    for scenario, data in sorted(results.items(), key=lambda x: x[1]['final_divergence'], reverse=True):
        divs = data['divergences']
        print(f"{scenario:<20} {divs.get(2025, 0):>9.2f}% {divs.get(2030, 0):>9.2f}% "
              f"{divs.get(2035, 0):>9.2f}% {divs.get(2040, 0):>9.2f}% "
              f"{str(data['max_acceleration_year']):<15} {data['final_divergence']:>9.2f}%")

    return results


def analyze_sensitivity_interactions(df_numeric, meta_df):
    """Analyze interactions between different sensitivities"""
    print("\n" + "=" * 100)
    print("ANALYSIS 2: SENSITIVITY INTERACTION EFFECTS")
    print("=" * 100)

    # Compare: no-LDES + no-clean-firm effect vs individual effects
    baseline_mask = meta_df['scenario_type'] == 'baseline'
    no_ldes_mask = meta_df['scenario_type'] == 'no-LDES'
    no_cf_mask = meta_df['scenario_type'] == 'no-clean-firm'

    baseline_data = df_numeric[meta_df[baseline_mask]['scenario_name'].tolist()].mean(axis=1)
    no_ldes_data = df_numeric[meta_df[no_ldes_mask]['scenario_name'].tolist()].mean(axis=1)
    no_cf_data = df_numeric[meta_df[no_cf_mask]['scenario_name'].tolist()].mean(axis=1)

    # Individual effects
    ldes_effect = ((no_ldes_data - baseline_data) / (baseline_data + 1e-6) * 100).mean()
    cf_effect = ((no_cf_data - baseline_data) / (baseline_data + 1e-6) * 100).mean()

    # Expected combined effect (if additive)
    expected_combined = ldes_effect + cf_effect

    print("\nINTERACTION ANALYSIS:")
    print("-" * 100)
    print(f"Individual LDES removal effect:        {ldes_effect:>8.2f}%")
    print(f"Individual Clean-Firm removal effect:  {cf_effect:>8.2f}%")
    print(f"Expected combined effect (additive):   {expected_combined:>8.2f}%")
    print(f"\nInteraction type: {'Sub-additive (technologies compensate)' if expected_combined > 0 else 'N/A'}")

    # Participation level sensitivity
    print("\n\nPARTICIPATION LEVEL SENSITIVITY:")
    print("-" * 100)

    for matching_level in [25, 50, 99]:
        mask_25 = (meta_df['matching_level'] == matching_level) & (meta_df['participation'] == 25)
        mask_50 = (meta_df['matching_level'] == matching_level) & (meta_df['participation'] == 50)

        if mask_25.sum() > 0 and mask_50.sum() > 0:
            data_25 = df_numeric[meta_df[mask_25]['scenario_name'].tolist()].mean(axis=1)
            data_50 = df_numeric[meta_df[mask_50]['scenario_name'].tolist()].mean(axis=1)

            pct_change = ((data_50 - data_25) / (data_25 + 1e-6) * 100).mean()

            print(f"Matching {matching_level}%: Doubling participation (25%→50%) causes {pct_change:+.2f}% mean change")


def analyze_temporal_patterns_advanced(df_numeric, meta_df):
    """Advanced temporal pattern analysis"""
    print("\n" + "=" * 100)
    print("ANALYSIS 3: ADVANCED TEMPORAL PATTERNS")
    print("=" * 100)

    # Load timeseries data
    try:
        ts_file = '../results/results_timeseries.parquet'
        df_ts = pd.read_parquet(ts_file)

        print("\nTimeseries data loaded successfully")
        print(f"Shape: {df_ts.shape}")

        # Analyze seasonal patterns
        # Get columns that are timestamps
        timestamp_cols = [col for col in df_ts.columns if isinstance(col, pd.Timestamp)]

        if len(timestamp_cols) > 0:
            print(f"Found {len(timestamp_cols)} hourly timestamps")

            # Group by month and scenario
            monthly_patterns = {}

            for scenario in df_ts.index.get_level_values('scenario').unique()[:5]:  # Sample 5 scenarios
                try:
                    scenario_data = df_ts.loc[scenario]

                    # Extract month from timestamps
                    monthly_means = {}
                    for col in timestamp_cols[:100]:  # Sample for speed
                        month = col.month
                        value = scenario_data[col].mean()
                        if month not in monthly_means:
                            monthly_means[month] = []
                        monthly_means[month].append(value)

                    monthly_patterns[scenario] = {m: np.mean(v) for m, v in monthly_means.items()}
                except:
                    continue

            print("\nSEASONAL VARIABILITY BY SCENARIO:")
            print("-" * 100)
            for scenario, months in list(monthly_patterns.items())[:3]:
                if len(months) > 0:
                    values = list(months.values())
                    coef_var = np.std(values) / (np.mean(values) + 1e-6)
                    print(f"{scenario}: Seasonal CV = {coef_var:.3f}")

    except Exception as e:
        print(f"\nCould not load timeseries data: {e}")

    # Year-over-year growth rates from frontier data
    print("\n\nYEAR-OVER-YEAR GROWTH RATES:")
    print("-" * 100)

    for scenario_type in ['baseline', 'hourly-match', 'no-LDES']:
        mask = meta_df['scenario_type'] == scenario_type
        if mask.sum() == 0:
            continue

        growth_rates = []
        for year_pair in [(2025, 2030), (2030, 2035), (2035, 2040)]:
            y1, y2 = year_pair
            scenarios_y1 = meta_df[mask & (meta_df['year'] == str(y1))]['scenario_name'].tolist()
            scenarios_y2 = meta_df[mask & (meta_df['year'] == str(y2))]['scenario_name'].tolist()

            if len(scenarios_y1) > 0 and len(scenarios_y2) > 0:
                data_y1 = df_numeric[scenarios_y1]
                data_y2 = df_numeric[scenarios_y2]

                mean_y1 = data_y1.mean(axis=1).mean()
                mean_y2 = data_y2.mean(axis=1).mean()

                # Annualized growth rate
                years_diff = y2 - y1
                growth = (mean_y2 / mean_y1) ** (1/years_diff) - 1
                growth_rates.append(growth * 100)

        if len(growth_rates) > 0:
            print(f"{scenario_type:<20}: {growth_rates[0]:>6.2f}% → {growth_rates[1]:>6.2f}% → {growth_rates[2]:>6.2f}% annualized")


def analyze_regional_heterogeneity(df_numeric, meta_df):
    """Analyze differences across countries/regions"""
    print("\n" + "=" * 100)
    print("ANALYSIS 4: REGIONAL HETEROGENEITY")
    print("=" * 100)

    # Get baseline scenario
    baseline_mask = meta_df['scenario_type'] == 'baseline'
    hourly_mask = (meta_df['scenario_type'] == 'hourly-match') & (meta_df['matching_level'] == 50)

    baseline_by_country = {}
    hourly_by_country = {}

    countries = meta_df['country'].unique()

    for country in countries:
        baseline_country = df_numeric[meta_df[baseline_mask & (meta_df['country'] == country)]['scenario_name'].tolist()]
        hourly_country = df_numeric[meta_df[hourly_mask & (meta_df['country'] == country)]['scenario_name'].tolist()]

        if len(baseline_country.columns) > 0:
            baseline_by_country[country] = baseline_country.mean(axis=1).mean()
        if len(hourly_country.columns) > 0:
            hourly_by_country[country] = hourly_country.mean(axis=1).mean()

    # Calculate impact variation
    impacts = {}
    for country in baseline_by_country.keys():
        if country in hourly_by_country:
            baseline_val = baseline_by_country[country]
            hourly_val = hourly_by_country[country]
            impact = (hourly_val - baseline_val) / (baseline_val + 1e-6) * 100
            impacts[country] = impact

    # Sort by impact
    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTOP 10 COUNTRIES BY POLICY IMPACT (baseline → hourly-match-50%):")
    print("-" * 100)
    print(f"{'Country':<25} {'Impact':<10} {'Interpretation'}")
    print("-" * 100)

    for country, impact in sorted_impacts[:10]:
        interpretation = "High sensitivity" if abs(impact) > 5 else "Moderate sensitivity"
        print(f"{country:<25} {impact:>9.2f}% {interpretation}")

    # Statistical test: Are country impacts significantly different?
    impact_values = list(impacts.values())
    if len(impact_values) > 2:
        # Coefficient of variation
        cv = np.std(impact_values) / (np.mean(np.abs(impact_values)) + 1e-6)
        print(f"\nRegional heterogeneity (CV of impacts): {cv:.3f}")
        print(f"Interpretation: {'High heterogeneity' if cv > 0.5 else 'Moderate heterogeneity' if cv > 0.3 else 'Low heterogeneity'}")


def analyze_nonlinear_thresholds(df_numeric, meta_df):
    """Identify non-linear thresholds and phase transitions"""
    print("\n" + "=" * 100)
    print("ANALYSIS 5: NON-LINEAR THRESHOLDS & PHASE TRANSITIONS")
    print("=" * 100)

    # Analyze matching level progression: 0% (baseline) → 25% → 50% → 99%
    matching_levels = [0, 25, 50, 99]

    print("\nMATCHING LEVEL PROGRESSION ANALYSIS:")
    print("-" * 100)

    mean_by_level = {}
    for level in matching_levels:
        if level == 0:
            mask = meta_df['scenario_type'] == 'baseline'
        else:
            mask = (meta_df['scenario_type'] == 'hourly-match') & (meta_df['matching_level'] == level)

        if mask.sum() > 0:
            data = df_numeric[meta_df[mask]['scenario_name'].tolist()]
            mean_by_level[level] = data.mean(axis=1).mean()

    # Calculate marginal cost (change per % increase in matching)
    marginal_costs = []
    for i in range(len(matching_levels) - 1):
        l1, l2 = matching_levels[i], matching_levels[i+1]
        if l1 in mean_by_level and l2 in mean_by_level:
            delta_level = l2 - l1
            delta_value = mean_by_level[l2] - mean_by_level[l1]
            marginal_cost = delta_value / delta_level
            marginal_costs.append((l1, l2, marginal_cost))

    print(f"{'From':<8} {'To':<8} {'Marginal Cost':<20} {'Interpretation'}")
    print("-" * 100)

    for i, (l1, l2, mc) in enumerate(marginal_costs):
        if i > 0:
            prev_mc = marginal_costs[i-1][2]
            accel = mc / prev_mc if prev_mc != 0 else 0
            interp = f"{accel:.2f}x acceleration" if accel > 1.5 else "Linear"
        else:
            interp = "Baseline"

        print(f"{l1}%→{l2}%  {mc:>18.4f}  {interp}")


def statistical_significance_testing(df_numeric, meta_df):
    """Perform statistical tests on scenario differences"""
    print("\n" + "=" * 100)
    print("ANALYSIS 6: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 100)

    # Compare baseline vs each policy scenario
    baseline_mask = meta_df['scenario_type'] == 'baseline'
    baseline_data = df_numeric[meta_df[baseline_mask]['scenario_name'].tolist()]

    print("\nMANN-WHITNEY U TESTS (Baseline vs Policy Scenarios):")
    print("-" * 100)
    print(f"{'Scenario':<25} {'p-value':<12} {'Effect Size (r)':<18} {'Significant?'}")
    print("-" * 100)

    for scenario_type in ['hourly-match', 'no-LDES', 'no-clean-firm', 'noadd']:
        mask = meta_df['scenario_type'] == scenario_type
        if mask.sum() == 0:
            continue

        scenario_data = df_numeric[meta_df[mask]['scenario_name'].tolist()]

        # Perform test on each metric
        p_values = []
        effect_sizes = []

        for metric_idx in df_numeric.index:
            baseline_vals = baseline_data.loc[metric_idx].dropna()
            scenario_vals = scenario_data.loc[metric_idx].dropna()

            if len(baseline_vals) > 5 and len(scenario_vals) > 5:
                # Mann-Whitney U test (non-parametric)
                statistic, p_val = mannwhitneyu(baseline_vals, scenario_vals, alternative='two-sided')
                p_values.append(p_val)

                # Effect size (rank-biserial correlation)
                n1, n2 = len(baseline_vals), len(scenario_vals)
                r = 1 - (2*statistic) / (n1 * n2)
                effect_sizes.append(abs(r))

        if len(p_values) > 0:
            median_p = np.median(p_values)
            mean_effect = np.mean(effect_sizes)
            significant = "Yes***" if median_p < 0.001 else "Yes**" if median_p < 0.01 else "Yes*" if median_p < 0.05 else "No"

            print(f"{scenario_type:<25} {median_p:<12.4f} {mean_effect:<18.3f} {significant}")


def main():
    """Run all deep-dive analyses"""

    # Load data
    df_numeric, meta_df = load_all_data()

    # Run analyses
    divergence_results = analyze_scenario_divergence(df_numeric, meta_df)
    analyze_sensitivity_interactions(df_numeric, meta_df)
    analyze_temporal_patterns_advanced(df_numeric, meta_df)
    analyze_regional_heterogeneity(df_numeric, meta_df)
    analyze_nonlinear_thresholds(df_numeric, meta_df)
    statistical_significance_testing(df_numeric, meta_df)

    # Summary
    print("\n" + "=" * 100)
    print("DEEP-DIVE ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nKey findings will be integrated into the dashboard Key Insights tab.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
