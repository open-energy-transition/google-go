# New Features Added to Google-Go Dashboard

## Summary
Two major feature sets have been added to the dashboard based on user requirements:

1. **Within-Scenario Comparison Tab**: Compare two sub-scenarios within the same main scenario
2. **Technology Trajectory Plot**: Line plot showing technology evolution over years
3. **Human-Readable Scenario Names**: All scenario names are now displayed in a user-friendly format

---

## 1. Within-Scenario Comparison Tab

### Purpose
Allows users to compare any two sub-scenarios within the same main scenario. For example, you can now compare:
- Hourly 90% vs Hourly 95% within CI 50%
- Energy Match 25% vs Hourly 90% within CI 25%
- Any two sub-scenarios side-by-side

### Location
New tab: **"Within-Scenario"** (rightmost tab in the dashboard)

### Features
- **Main Scenario Selector**: Choose CI 25%, CI 50%, or No Additional Constraints
- **Year Selector**: Select the year to compare
- **Metric Selector**: Choose which metric to analyze
- **Sub-Scenario 1 & 2 Selectors**: Pick any two different sub-scenarios to compare

### Visualizations
1. **Side-by-Side Comparison Plot**: Grouped bar chart showing both sub-scenarios
2. **Difference Analysis Plot**: Shows the difference between Sub-Scenario 2 and Sub-Scenario 1
   - Green bars: Values higher in Scenario 2
   - Red bars: Values higher in Scenario 1
3. **Summary Statistics**: Overall comparison with:
   - Total values for each sub-scenario
   - Absolute difference
   - Percentage change

### How to Use
1. Navigate to the "Within-Scenario" tab
2. Select a main scenario (e.g., CI_50)
3. Choose year, metric, and two different sub-scenarios
4. View the side-by-side comparison and difference analysis
5. Check summary statistics for quick insights

---

## 2. Technology Trajectory Plot

### Purpose
Visualizes how each technology evolves over time for a given scenario. This line plot shows the trajectory/trend of each carrier (technology) across all available years.

### Location
Available in all three individual scenario tabs:
- CI_25 tab
- CI_50 tab
- CI_noadd tab

### How to Access
1. Navigate to any individual scenario tab (CI_25, CI_50, or CI_noadd)
2. In the "Plot Type" dropdown, select **"Technology Trajectory"**
3. Choose your scenario, metric, and carriers

### Features
- **Line Plot**: Each technology is represented as a line over years
- **Color Coding**: Uses the same color mapping from colors.csv for consistency
- **Interactive Legend**: Click to show/hide specific technologies
- **Markers**: Each data point is marked for clarity
- **Hover Details**: Shows exact values for each year and technology

### Use Cases
- Track how a specific technology grows or declines over time
- Identify trends in technology deployment
- Compare growth rates between different carriers
- Visualize technology mix evolution

### Example
To see how solar capacity changes from 2025 to 2050 under CI_50 with Hourly 90%:
1. Go to CI_50 tab
2. Select "Hourly 90%" scenario
3. Choose "(c) Generator capacity" metric
4. Select plot type: "Technology Trajectory"
5. Filter to "solar" carrier (or keep all selected)
6. View the line plot showing solar's trajectory

---

## 3. Human-Readable Scenario Names

### Purpose
Makes the dashboard more user-friendly by displaying scenario names in a readable format instead of raw CSV naming.

### Transformations
| Raw Name | Display Name |
|----------|--------------|
| `baseline` | Baseline |
| `energy-match-25` | Energy Match 25% |
| `energy-match-50` | Energy Match 50% |
| `hourly-match-25-90` | Hourly 90% (CI 25%) |
| `hourly-match-50-95` | Hourly 95% (CI 50%) |
| `hourly-match-noadd-90-99` | Hourly 90%-99% (No Add.) |
| `CI_25` | CI 25% |
| `CI_50` | CI 50% |
| `CI_noadd` | No Additional Constraints |

### Where Applied
- All dropdown menus showing sub-scenarios
- Plot titles and subtitles
- Legend labels
- Summary statistics
- Comparison displays

### Benefits
- Easier to understand what each scenario represents
- More professional appearance
- Clearer communication of results
- Better for presentations and reports

---

## Technical Implementation

### Files Modified
1. **`/dashboard/utils/colors.py`**: Added formatting functions
   - `format_scenario_name()`: Converts sub-scenario names
   - `format_main_scenario_name()`: Converts main scenario names

2. **`/dashboard/layouts/within_scenario_layout.py`**: New layout file for within-scenario comparison

3. **`/dashboard/callbacks.py`**: Added new callbacks and functions
   - `create_within_comparison_plot()`
   - `create_within_difference_plot()`
   - `create_within_summary_stats()`
   - `create_trajectory_plot()`
   - Updated all plot titles to use formatted names

4. **`/dashboard/layouts/ci25_layout.py`**: Added trajectory plot option
5. **`/dashboard/layouts/ci50_layout.py`**: Added trajectory plot option
6. **`/dashboard/layouts/cinoadd_layout.py`**: Added trajectory plot option
7. **`/dashboard/layouts/comparison_layout.py`**: Updated to use formatted names

8. **`/dashboard/app.py`**: Registered new within-scenario tab

### Backward Compatibility
- All existing features continue to work
- Data loading unchanged
- Existing URLs and bookmarks still functional
- No breaking changes to data structure

---

## Usage Examples

### Example 1: Compare Hourly Matching Stringency within CI 50%
**Question**: How much more capacity is needed for Hourly 95% vs Hourly 90% in 2050?

**Steps**:
1. Go to "Within-Scenario" tab
2. Select: CI 50%, Year: 2050, Metric: "(c) Generator capacity"
3. Sub-Scenario 1: Hourly 90% (CI 50%)
4. Sub-Scenario 2: Hourly 95% (CI 50%)
5. View the difference plot to see which technologies need more capacity

### Example 2: Visualize Solar Growth Trajectory
**Question**: How does solar capacity grow over time under the most stringent scenario?

**Steps**:
1. Go to CI_50 tab
2. Select scenario: Hourly 95% (CI 50%)
3. Metric: "(c) Generator capacity"
4. Plot Type: "Technology Trajectory"
5. Carriers: Select only "solar"
6. View the line plot showing solar's growth from 2025 to 2050

### Example 3: Compare Baseline Across All Three Main Scenarios
**Question**: What's the difference in energy mix for baseline across CI levels?

**Steps**:
1. Go to "Comparison" tab
2. Sub-Scenario: Select "Baseline"
3. Year: 2050, Metric: "(a) Energy mix"
4. Check all three main scenarios: CI 25%, CI 50%, No Additional Constraints
5. View grouped bar chart comparing baseline across scenarios

---

## Future Enhancements (Suggested)

1. **Export Functionality**: Add buttons to export plots as PNG/SVG
2. **Data Tables**: Add downloadable CSV tables for each visualization
3. **Multi-Year Trajectory**: Allow selecting specific years to show on trajectory plots
4. **Carrier Grouping**: Add ability to group carriers by category in trajectory plots
5. **Scenario Comparison Matrix**: Heat map showing differences across all scenarios
6. **Animated Trajectories**: Animated transitions showing technology evolution

---

## Support

For questions or issues:
1. Check the main README.md for general dashboard usage
2. Review VERIFICATION_CHECKLIST.md to ensure proper setup
3. Contact the development team with specific questions

Dashboard updated: December 4, 2025
