# How to Use the Cross-Scenario Comparison Tab

## Overview

The Cross-Scenario Comparison tab allows you to compare up to 4 different scenarios side-by-side. All scenarios are now consolidated in a single dataset, making comparisons straightforward.

## Available Scenarios

All scenarios from the consolidated dataset are available for comparison, including:

### Common Scenarios:
- **baseline** - Base case without policy constraints

### Energy Matching Scenarios:
- **energy-match-25** - 25% energy matching
- **energy-match-50** - 50% energy matching

### Hourly Matching Scenarios (25% additionality):
- **hourly-match-25-90** - 25% additionality, 90% hourly matching
- **hourly-match-25-95** - 25% additionality, 95% hourly matching
- **hourly-match-25-98** - 25% additionality, 98% hourly matching
- **hourly-match-25-99** - 25% additionality, 99% hourly matching

### Hourly Matching Scenarios (50% additionality):
- **hourly-match-50-90** - 50% additionality, 90% hourly matching
- **hourly-match-50-95** - 50% additionality, 95% hourly matching
- **hourly-match-50-98** - 50% additionality, 98% hourly matching
- **hourly-match-50-99** - 50% additionality, 99% hourly matching

### No Additionality Scenarios:
- **hourly-match-noadd-10-99** - No additionality, 10% target, 99% hourly matching
- **hourly-match-noadd-50-99** - No additionality, 50% target, 99% hourly matching
- **hourly-match-noadd-90-99** - No additionality, 90% target, 99% hourly matching

## Understanding the Results

### When Comparing "baseline"
**Expected:** The baseline scenario provides a reference point for all policy scenarios.

**Use this to:** Understand the baseline energy system and measure policy impacts.

### When Comparing Policy Scenarios

#### Example 1: Compare Energy Matching Approaches
**Select:** `energy-match-25` (only shows CI_25 data)
- Only CI_25 will appear in the chart
- Shows 25% energy matching results

**Select:** `energy-match-50` (only shows CI_50 data)
- Only CI_50 will appear in the chart
- Shows 50% energy matching results

**To compare them:**
1. Note down values from `energy-match-25`
2. Switch to `energy-match-50` and compare

#### Example 2: Compare Hourly Matching at 99%
**Select:** `hourly-match-25-99`
- Shows CI_25 with 25% additionality at 99% hourly matching

**Select:** `hourly-match-50-99`
- Shows CI_50 with 50% additionality at 99% hourly matching

**Select:** `hourly-match-noadd-50-99`
- Shows CI_noadd (no additionality requirement) with 50% target at 99% hourly matching

**To see all three side-by-side:**
Unfortunately, since they have different names, you'll need to check each individually. However, the plot will show **which scenarios have data** for the selected sub-scenario in the subtitle.

## Recommended Comparison Workflows

### Workflow 1: Compare Policy Impacts
```
1. Select: Year = 2030
2. Select: Scenario 1 = baseline
3. Select: Scenario 2 = hourly-match-25-90
4. Select: Metric = (a) Energy mix
5. Result: See how 90% hourly matching affects the energy mix
```

### Workflow 2: Compare Additionality Levels
```
1. Note values from:
   - CI_25, hourly-match-25-99 (25% additionality)
   - CI_50, hourly-match-50-99 (50% additionality)
   - CI_noadd, hourly-match-noadd-50-99 (no additionality)
2. Compare how energy mix changes with additionality requirements
```

### Workflow 3: Compare Hourly Matching Levels
Within a single main scenario tab (e.g., CI_50):
```
1. Go to CI_50 tab
2. Select different sub-scenarios:
   - hourly-match-50-90 (90% matching)
   - hourly-match-50-95 (95% matching)
   - hourly-match-50-98 (98% matching)
   - hourly-match-50-99 (99% matching)
3. Use "Year Comparison" or "Stacked Bar" plot type
4. Observe how stricter matching requirements affect the system
```

### Workflow 4: Energy Matching vs Hourly Matching
```
1. CI_25 tab: Compare energy-match-25 vs hourly-match-25-99
2. CI_50 tab: Compare energy-match-50 vs hourly-match-50-99
3. See difference between energy and hourly matching approaches
```

## Visual Indicators

### Plot Title Shows Active Scenarios
The comparison plot title includes:
```
Comparison: (a) Energy mix (2030) - hourly-match-25-99
Showing: CI_25
```

This tells you:
- What you're comparing: Energy mix in 2030
- Which sub-scenario: hourly-match-25-99
- Which main scenarios have this sub-scenario: Only CI_25

### When No Data Available
If you select a sub-scenario that doesn't exist in any checked main scenario:
```
Sub-scenario 'hourly-match-25-99' not found in any selected main scenarios
```

**Fix:** Either:
1. Select a sub-scenario that exists in at least one checked scenario, OR
2. Check different main scenarios

## Tips for Effective Comparison

### 1. **Start with Baseline**
Always verify baseline is consistent before comparing policy scenarios.

### 2. **Use Individual Tabs for Detailed Analysis**
The individual tabs (CI_25, CI_50, CI_noadd) have more plot options:
- Stacked Bar (All Years) - See evolution over time
- Year Comparison - Compare years side-by-side
- Better for deep-dive into a single scenario

### 3. **Compare Equivalent Policies Manually**
Since naming differs, create a comparison table manually:

| Metric | CI_25 | CI_50 | CI_noadd |
|--------|-------|-------|----------|
| Solar (TWh) @ 99% hourly | [from hourly-match-25-99] | [from hourly-match-50-99] | [from hourly-match-noadd-50-99] |

### 4. **Focus on Key Differences**
Look for:
- **Renewable vs Fossil**: How does additionality affect clean energy deployment?
- **Storage**: Do stricter requirements need more storage?
- **Costs**: What's the cost impact of different policies?
- **Emissions**: How do policies affect CO2 emissions?

## Understanding the Scenario Names

### Format: `[type]-[additionality]-[matching]`

Examples:
- `energy-match-25` = Energy matching with 25% additionality
- `hourly-match-50-99` = Hourly matching with 50% additionality at 99% matching level
- `hourly-match-noadd-50-99` = Hourly matching with NO additionality, 50% target, 99% matching

### Key Parameters:

**Additionality Level:**
- `25` = 25% of clean energy must be new/additional
- `50` = 50% of clean energy must be new/additional
- `noadd` = No additionality requirement (can use existing clean energy)

**Matching Level:**
- `90` = 90% hourly matching requirement
- `95` = 95% hourly matching requirement
- `98` = 98% hourly matching requirement
- `99` = 99% hourly matching requirement

## Future Enhancement

A more advanced version could:
1. Allow selecting **different sub-scenarios per main scenario**
2. Auto-detect "equivalent" scenarios (e.g., all 99% hourly matching scenarios)
3. Add a "Smart Compare" mode that maps comparable scenarios automatically
4. Add scenario-to-scenario difference heatmaps

For now, use the dropdown to explore available sub-scenarios and check which main scenarios have data for each selection!
