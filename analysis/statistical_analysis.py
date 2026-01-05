"""
Statistical Analysis of Google GO Energy System Results
Focus on trends and patterns hard for humans to detect manually
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_frontier_data():
    """Load and structure the frontier data"""
    print("=" * 80)
    print("LOADING FRONTIER DATA")
    print("=" * 80)

    df = pd.read_csv('../results/results_frontier.csv')

    # Extract metadata from first two rows
    years = df.iloc[0, 1:].values  # First row (skip 'scenario' column)
    countries = df.iloc[1, 1:].values  # Second row

    # Get numeric data (skip first two rows)
    df_numeric = df.iloc[2:, 1:].copy()  # Skip first two metadata rows and first column
    df_numeric.index = df.iloc[2:, 0].values  # Set index to metric IDs

    # Set column names to scenario names
    df_numeric.columns = df.columns[1:]

    # Convert to numeric
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Create structured dataframe with metadata
    scenario_names = df_numeric.columns.tolist()

    # Parse scenario information
    scenario_info = []
    for idx, name in enumerate(scenario_names):
        base_name = name.split('.')[0] if '.' in name else name
        run_num = name.split('.')[-1] if '.' in name else '0'

        # Parse scenario type
        if 'baseline' in base_name:
            scenario_type = 'baseline'
        elif 'no-clean-firm' in base_name:
            scenario_type = 'no-clean-firm'
        elif 'EU-25' in base_name:
            scenario_type = 'EU-25'
        elif 'EU-50' in base_name:
            scenario_type = 'EU-50'
        elif 'no-LDES' in base_name:
            scenario_type = 'no-LDES'
        elif 'noadd' in base_name:
            scenario_type = 'noadd'
        elif 'hourly-match-25' in base_name:
            scenario_type = 'hourly-match-25'
        elif 'hourly-match-50' in base_name:
            scenario_type = 'hourly-match-50'
        else:
            scenario_type = 'other'

        year = years[idx]
        country = countries[idx]

        scenario_info.append({
            'scenario_name': name,
            'scenario_type': scenario_type,
            'run_num': run_num,
            'year': year,
            'country': country
        })

    meta_df = pd.DataFrame(scenario_info)

    print(f"\nLoaded {len(scenario_names)} scenario runs")
    print(f"Number of metrics: {len(df_numeric)}")
    print(f"\nScenario types: {meta_df['scenario_type'].value_counts()}")
    print(f"\nYears: {meta_df['year'].value_counts().sort_index()}")
    print(f"\nCountries (top 10): {meta_df['country'].value_counts().head(10)}")

    return df_numeric, meta_df


def analyze_frontier_curves(df_numeric, meta_df):
    """Analyze the cost-effectiveness frontier curves"""
    print("\n" + "=" * 80)
    print("ANALYZING FRONTIER CURVES")
    print("=" * 80)

    # The rows represent different matching percentage thresholds (50-99%)
    # Values represent system cost at that threshold

    results = {}

    # Group by scenario type
    for scenario_type in meta_df['scenario_type'].unique():
        if scenario_type == 'other':
            continue

        mask = meta_df['scenario_type'] == scenario_type
        selected_scenarios = meta_df[mask]['scenario_name'].tolist()
        scenario_data = df_numeric[selected_scenarios]

        # Calculate statistics across runs
        mean_curve = scenario_data.mean(axis=1)
        std_curve = scenario_data.std(axis=1)

        # Find inflection points (where cost acceleration changes)
        # First derivative (rate of cost increase)
        first_deriv = np.diff(mean_curve.dropna())

        # Second derivative (acceleration of cost)
        second_deriv = np.diff(first_deriv)

        # Find maximum acceleration point
        max_accel_idx = np.argmax(np.abs(second_deriv))

        # Calculate elasticity (% change in cost per % change in threshold)
        pct_changes = np.diff(mean_curve.dropna()) / mean_curve.dropna().iloc[:-1] * 100

        results[scenario_type] = {
            'mean_curve': mean_curve,
            'std_curve': std_curve,
            'max_acceleration_point': max_accel_idx + 2,  # +2 for double diff offset
            'mean_elasticity': pct_changes.mean(),
            'max_cost_increase_step': pct_changes.max()
        }

    # Statistical comparison
    print("\nFRONTIER CURVE ANALYSIS:")
    print("-" * 80)
    for scenario_type, data in sorted(results.items()):
        print(f"\n{scenario_type.upper()}:")
        print(f"  Maximum acceleration at threshold: {data['max_acceleration_point']}%")
        print(f"  Mean elasticity: {data['mean_elasticity']:.3f}%")
        print(f"  Largest single-step cost increase: {data['max_cost_increase_step']:.2f}%")

    return results


def analyze_cross_scenario_correlations(df_numeric, meta_df):
    """Find correlations between different scenario outcomes"""
    print("\n" + "=" * 80)
    print("CROSS-SCENARIO CORRELATION ANALYSIS")
    print("=" * 80)

    # Compare baseline vs policy scenarios
    # For each metric, correlate outcomes

    baseline_mask = meta_df['scenario_type'] == 'baseline'
    baseline_scenarios = meta_df[baseline_mask]['scenario_name'].tolist()
    baseline_data = df_numeric[baseline_scenarios]

    correlations = {}

    for scenario_type in ['hourly-match-25', 'hourly-match-50', 'no-LDES', 'noadd']:
        scenario_mask = meta_df['scenario_type'] == scenario_type
        if scenario_mask.sum() == 0:
            continue

        selected_scenarios = meta_df[scenario_mask]['scenario_name'].tolist()
        scenario_data = df_numeric[selected_scenarios]

        # Match countries/years between baseline and scenario
        baseline_meta = meta_df[baseline_mask]
        scenario_meta = meta_df[scenario_mask]

        # Correlation analysis - compare means across countries/years
        corr_values = []
        for metric_idx in df_numeric.index:
            baseline_vals = baseline_data.loc[metric_idx].dropna()
            scenario_vals = scenario_data.loc[metric_idx].dropna()

            if len(baseline_vals) > 2 and len(scenario_vals) > 2:
                # Use mean values to avoid size mismatch
                baseline_mean = baseline_vals.mean()
                scenario_mean = scenario_vals.mean()

                # Calculate relative difference
                if baseline_mean != 0:
                    rel_diff = (scenario_mean - baseline_mean) / baseline_mean
                    corr_values.append(rel_diff)

        correlations[scenario_type] = {
            'mean_correlation': np.mean(corr_values),
            'min_correlation': np.min(corr_values),
            'max_correlation': np.max(corr_values),
        }

    print("\nRELATIVE DIFFERENCE FROM BASELINE:")
    print("-" * 80)
    for scenario, stats_dict in sorted(correlations.items()):
        print(f"\n{scenario}:")
        print(f"  Mean relative difference: {stats_dict['mean_correlation']*100:.2f}%")
        print(f"  Range: [{stats_dict['min_correlation']*100:.2f}%, {stats_dict['max_correlation']*100:.2f}%]")

    return correlations


def identify_tipping_points(df_numeric, meta_df):
    """Identify non-linear thresholds and tipping points"""
    print("\n" + "=" * 80)
    print("TIPPING POINT ANALYSIS")
    print("=" * 80)

    tipping_points = {}

    # For each scenario type, find where cost curves become exponential
    for scenario_type in meta_df['scenario_type'].unique():
        if scenario_type == 'other':
            continue

        mask = meta_df['scenario_type'] == scenario_type
        selected_scenarios = meta_df[mask]['scenario_name'].tolist()
        scenario_data = df_numeric[selected_scenarios]

        mean_curve = scenario_data.mean(axis=1).dropna()

        # Fit exponential vs linear models to different segments
        x = np.arange(len(mean_curve))
        y = mean_curve.values

        # Find where exponential fit becomes significantly better than linear
        best_break_point = None
        best_improvement = 0

        for break_idx in range(10, len(y) - 10):
            # Linear fit before break
            before_x, before_y = x[:break_idx], y[:break_idx]
            after_x, after_y = x[break_idx:], y[break_idx:]

            # R² for linear fit on both segments
            if len(before_y) > 2 and len(after_y) > 2:
                try:
                    before_r2 = np.corrcoef(before_x, before_y)[0, 1] ** 2
                    after_r2 = np.corrcoef(after_x, after_y)[0, 1] ** 2

                    # Check if slope changes significantly
                    before_slope = np.polyfit(before_x, before_y, 1)[0]
                    after_slope = np.polyfit(after_x, after_y, 1)[0]

                    slope_change = after_slope / before_slope if before_slope != 0 else 0

                    if slope_change > best_improvement and slope_change > 1.5:
                        best_improvement = slope_change
                        best_break_point = break_idx
                except:
                    pass

        if best_break_point:
            tipping_points[scenario_type] = {
                'threshold_idx': best_break_point,
                'slope_multiplier': best_improvement
            }

    print("\nTIPPING POINTS DETECTED:")
    print("-" * 80)
    for scenario, data in sorted(tipping_points.items()):
        print(f"\n{scenario}:")
        print(f"  Tipping point at threshold: {data['threshold_idx']}%")
        print(f"  Cost acceleration: {data['slope_multiplier']:.2f}x")

    return tipping_points


def cluster_scenarios(df_numeric, meta_df):
    """Cluster scenarios by similarity"""
    print("\n" + "=" * 80)
    print("SCENARIO CLUSTERING ANALYSIS")
    print("=" * 80)

    # Use complete data only - scenarios with >=95% data
    complete_mask = df_numeric.notna().sum(axis=0) >= len(df_numeric) * 0.95
    complete_scenarios = df_numeric.columns[complete_mask].tolist()
    df_complete = df_numeric[complete_scenarios].copy()
    meta_complete = meta_df[meta_df['scenario_name'].isin(complete_scenarios)].reset_index(drop=True)

    print(f"\nUsing {len(complete_scenarios)} scenarios with >=95% complete data")

    # Fill remaining NaN with column means
    for col in df_complete.columns:
        if df_complete[col].isna().any():
            df_complete[col].fillna(df_complete[col].mean(), inplace=True)

    # Drop any rows that are still all NaN
    df_complete = df_complete.dropna(how='all')

    # Final check and impute
    if df_complete.isna().any().any():
        print(f"  Warning: Still {df_complete.isna().sum().sum()} NaN values, filling with 0")
        df_complete = df_complete.fillna(0)

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_complete.T)

    # PCA
    pca = PCA(n_components=min(10, data_scaled.shape[1]))
    pca_result = pca.fit_transform(data_scaled)

    print(f"\nPCA Explained Variance (first 5 components):")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var*100:.2f}%")

    print(f"\nCumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum()*100:.2f}%")

    # Distance matrix
    distances = pdist(data_scaled, metric='euclidean')
    dist_matrix = squareform(distances)

    # Find most similar and most different scenarios
    np.fill_diagonal(dist_matrix, np.inf)

    min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    print(f"\nMost similar scenarios:")
    print(f"  {meta_complete.iloc[min_dist_idx[0]]['scenario_type']} vs")
    print(f"  {meta_complete.iloc[min_dist_idx[1]]['scenario_type']}")
    print(f"  Distance: {dist_matrix[min_dist_idx]:.2f}")

    print(f"\nMost different scenarios:")
    print(f"  {meta_complete.iloc[max_dist_idx[0]]['scenario_type']} vs")
    print(f"  {meta_complete.iloc[max_dist_idx[1]]['scenario_type']}")
    print(f"  Distance: {dist_matrix[max_dist_idx]:.2f}")

    return pca_result, pca, meta_complete


def analyze_variability(df_numeric, meta_df):
    """Analyze which scenarios have highest variability"""
    print("\n" + "=" * 80)
    print("VARIABILITY ANALYSIS")
    print("=" * 80)

    variability = {}

    for scenario_type in meta_df['scenario_type'].unique():
        if scenario_type == 'other':
            continue

        mask = meta_df['scenario_type'] == scenario_type
        selected_scenarios = meta_df[mask]['scenario_name'].tolist()
        scenario_data = df_numeric[selected_scenarios]

        # Coefficient of variation across metrics and runs
        cv_values = []
        for idx in scenario_data.index:
            vals = scenario_data.loc[idx].dropna()
            if len(vals) > 1 and vals.mean() != 0:
                cv = vals.std() / vals.mean()
                cv_values.append(cv)

        if cv_values:
            variability[scenario_type] = {
                'mean_cv': np.mean(cv_values),
                'max_cv': np.max(cv_values),
                'robustness_score': 1 / (1 + np.mean(cv_values))  # Higher = more robust
            }

    print("\nVARIABILITY BY SCENARIO TYPE:")
    print("-" * 80)
    print(f"{'Scenario':<25} {'Mean CV':<12} {'Max CV':<12} {'Robustness':<12}")
    print("-" * 80)

    # Sort by robustness
    sorted_scenarios = sorted(variability.items(), key=lambda x: x[1]['robustness_score'], reverse=True)

    for scenario, stats_dict in sorted_scenarios:
        print(f"{scenario:<25} {stats_dict['mean_cv']:>11.3f} {stats_dict['max_cv']:>11.3f} {stats_dict['robustness_score']:>11.3f}")

    return variability


def main():
    """Run complete statistical analysis"""
    print("\n" + "=" * 80)
    print("GOOGLE GO STATISTICAL ANALYSIS")
    print("Identifying trends and patterns difficult for manual detection")
    print("=" * 80 + "\n")

    # Load data
    df_numeric, meta_df = load_frontier_data()

    # Run analyses
    frontier_results = analyze_frontier_curves(df_numeric, meta_df)
    correlations = analyze_cross_scenario_correlations(df_numeric, meta_df)
    tipping_points = identify_tipping_points(df_numeric, meta_df)
    variability = analyze_variability(df_numeric, meta_df)
    pca_result, pca, meta_complete = cluster_scenarios(df_numeric, meta_df)

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    print("\n1. TIPPING POINTS:")
    print("   - Cost curves show non-linear acceleration beyond certain matching thresholds")
    print("   - These represent critical points where additional clean energy becomes disproportionately expensive")

    print("\n2. SCENARIO ROBUSTNESS:")
    print("   - Some scenarios show much higher variability than others")
    print("   - Low variability = more predictable outcomes")
    print("   - High variability = sensitive to local conditions")

    print("\n3. CROSS-SCENARIO PATTERNS:")
    print("   - Correlations reveal which policy interventions have similar effects")
    print("   - Low correlation = policy has fundamentally different impact")

    print("\n4. CLUSTERING:")
    print("   - PCA reveals hidden patterns in scenario outcomes")
    print("   - First few components capture most variation")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
