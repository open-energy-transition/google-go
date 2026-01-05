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
1. Navigate to the "Cross-Scenario Comparison" tab
2. Choose year, metric, and up to 4 scenarios to compare
3. View the side-by-side comparison
4. Check summary statistics for quick insights

---

## 2. Technology Trajectory Plot

### Purpose
Visualizes how each technology evolves over time for a given scenario. This line plot shows the trajectory/trend of each carrier (technology) across all available years.

### Location
Available in:
- Single Scenario Analysis tab
- Cross-Scenario Comparison tab
- Timeseries Exploration tab

### How to Access
1. Navigate to the Single Scenario Analysis tab
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
To see how solar capacity changes from 2025 to 2040 under hourly-match-50-90:
1. Go to Single Scenario Analysis tab
2. Select "Hourly 90% (CI 50%)" scenario
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
| `baseline` | Baseline |
| `energy-match-25` | Energy Match 25% |
| `hourly-match-50-90` | Hourly 90% (CI 50%) |

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

### Example 1: Compare Hourly Matching Stringency
**Question**: How much more capacity is needed for Hourly 95% vs Hourly 90% in 2040?

**Steps**:
1. Go to "Cross-Scenario Comparison" tab
2. Select: Year: 2040, Metric: "(c) Capacity mix"
3. Scenario 1: hourly-match-50-90
4. Scenario 2: hourly-match-50-95
5. View the comparison to see which technologies need more capacity

### Example 2: Visualize Solar Growth Trajectory
**Question**: How does solar capacity grow over time under the most stringent scenario?

**Steps**:
1. Go to Single Scenario Analysis tab
2. Select scenario: hourly-match-50-95
3. Metric: "(c) Capacity mix"
4. Plot Type: "Year on Year Evolution"
5. View the evolution showing solar's growth from 2025 to 2040

### Example 3: Explore Timeseries Data
**Question**: How does electricity balance vary throughout the year?

**Steps**:
1. Go to "Timeseries Exploration" tab
2. Select Year: 2035, Scenario: baseline
3. Timeseries Type: "Electricity Balance"
4. Country: EU
5. View hourly data across different time ranges (week, month, full year)

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
