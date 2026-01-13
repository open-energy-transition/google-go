# Comparison Tab Fix

## Problem Identified

The Comparison tab was showing the same values for CI_25, CI_50, and CI_noadd because it was always using the **first sub-scenario** (which was "baseline" for all three folders).

## Root Cause

The original code structure had:
- **Main scenarios**: CI_25, CI_50, CI_noadd (these are the folder names)
- **Sub-scenarios**: Different policy scenarios within each folder:
  - CI_25: `baseline`, `energy-match-25`, `hourly-match-25-90`, etc.
  - CI_50: `baseline`, `energy-match-50`, `hourly-match-50-90`, etc.
  - CI_noadd: `baseline`, `hourly-match-noadd-10-99`, etc.

The comparison was always comparing `baseline` from all three folders, which are identical (same baseline assumptions).

## Solution Implemented

### 1. Added Sub-Scenario Selector

Added a new dropdown in the Comparison tab to let users select **which sub-scenario** to compare across the three main scenarios.

**Layout Change:**
```python
# New dropdown added
dcc.Dropdown(
    id='comp-subscenario-selector',
    options=[{'label': s, 'value': s} for s in sub_scenarios],
    value='baseline',  # Default to baseline
    clearable=False
)
```

### 2. Smart Sub-Scenario Detection

The code now identifies **common sub-scenarios** across all three main scenarios:
- If all three have "baseline", it shows in the list
- Shows only sub-scenarios that exist in all three folders (intersection)
- If no common scenarios, shows all available sub-scenarios

```python
# Find common sub-scenarios
common_sub_scenarios = set(scenarios_data['CI_25'].get('scenarios', []))
for scenario in ['CI_50', 'CI_noadd']:
    common_sub_scenarios &= set(scenarios_data[scenario].get('scenarios', []))
```

### 3. Updated All Comparison Callbacks

All comparison-related callbacks now use the selected sub-scenario:

**Before:**
```python
def update_comparison_plot(year, metric, scenarios, plot_type):
    # Used first sub-scenario (always baseline)
    sub_scenario = sub_scenarios[0]
```

**After:**
```python
def update_comparison_plot(year, subscenario, metric, scenarios):
    # Uses user-selected sub-scenario
    df = data_loader.get_data(scenario, year=year, scenario_name=subscenario, metric=metric)
```

### 4. Updated Plot Titles

Plots now show which sub-scenario is being compared:

```python
title=f"Comparison: {metric} ({year}) - {subscenario}"
```

## How to Use

### Comparing Baseline Across CI_25, CI_50, CI_noadd:
1. Go to **Comparison** tab
2. Select **Year** (e.g., 2030)
3. Select **Sub-Scenario**: `baseline`
4. Select **Metric** (e.g., "(a) Energy mix")
5. Check which main scenarios to compare (CI_25, CI_50, CI_noadd)

Result: You'll see the same baseline across all three (as expected).

### Comparing Policy Scenarios:

To compare how different clean energy targets (25% vs 50% vs no additionality) affect outcomes:

1. Select **Year** (e.g., 2030)
2. Select **Sub-Scenario**:
   - For energy matching comparison: Can't directly compare (different names)
   - For hourly matching: Can't directly compare (different names)
   - **Recommended**: Compare "baseline" to see the baseline, then switch to individual tabs to see each policy

### Understanding the Difference:

**Baseline should be the same** because:
- Same baseline assumptions
- Same input data
- No policy constraints applied
- Represents the "business as usual" scenario

**Policy scenarios differ** because:
- CI_25 applies 25% clean energy target
- CI_50 applies 50% clean energy target
- CI_noadd applies no additionality constraint
- Different constraints lead to different energy mixes, costs, etc.

## What You Should See Now

### Baseline Comparison (should be identical):
```
CI_25 baseline (2030): Total = 1000 TWh
CI_50 baseline (2030): Total = 1000 TWh  ← Same as CI_25
CI_noadd baseline (2030): Total = 1000 TWh  ← Same as CI_25
```

### Policy Comparison (should differ):

To see the differences, you should compare the same type of constraint across different clean energy targets. However, since the sub-scenario names are different:
- CI_25: `hourly-match-25-99`
- CI_50: `hourly-match-50-99`
- CI_noadd: `hourly-match-noadd-50-99`

You'll need to use the **individual tabs** (CI_25, CI_50, CI_noadd) and note down the values, then compare manually, OR select different sub-scenarios one at a time in the comparison tab.

## Future Enhancement Idea

A more advanced comparison would be:
1. Add ability to select **different sub-scenarios per main scenario**
2. Create a mapping of "equivalent" scenarios across folders
3. Add a "smart comparison" mode that automatically picks comparable scenarios

For example:
- Compare `hourly-match-25-99` from CI_25
- Against `hourly-match-50-99` from CI_50
- Against `hourly-match-noadd-50-99` from CI_noadd

This would show how the **same policy type** (hourly matching at 99%) performs under different clean energy targets.

## Verification

To verify the fix works:

1. Go to Comparison tab
2. Select "baseline" → should show identical or very similar values
3. Select other sub-scenarios → should show different values (if that sub-scenario exists in the selected main scenarios)
4. Try selecting CI_25 only, then CI_50 only → values should differ for non-baseline scenarios
