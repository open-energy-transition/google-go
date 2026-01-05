# Summary Statistics Feature

## Overview

The Google-Go Analysis Dashboard now includes **comprehensive summary statistics** displayed below each plot, providing key insights and metrics about the data being visualized.

## Types of Summary Statistics

The dashboard automatically displays different summary statistics based on the selected **plot type**.

### 1. Single-Year Statistics
**Displayed for**: Bar Chart, Pie Chart, Stacked Area plots

Shows detailed statistics for a single selected year and scenario.

#### Key Metrics (Top Row)
Four prominent cards displaying:
- **Total**: Sum of all carrier values
- **Mean**: Average value across carriers
- **Max**: Maximum carrier value
- **Carriers**: Number of carriers with data

#### Statistical Summary (Bottom Left)
Detailed statistics table:
- **Median**: Middle value of the distribution
- **Std Dev**: Standard deviation (measure of variability)
- **Min**: Minimum carrier value
- **Range**: Difference between max and min

#### Top Contributors (Bottom Right)
Ordered list of the **top 5 carriers** by value:
- Carrier name
- Absolute value
- Percentage of total

**Example Display:**
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Total     │    Mean     │    Max      │  Carriers   │
│   1245.67   │   62.28     │   450.23    │     20      │
└─────────────┴─────────────┴─────────────┴─────────────┘

Statistical Summary:          Top Contributors:
Median:    45.12              1. solar: 450.23 (36.1%)
Std Dev:   78.34              2. onwind: 320.45 (25.7%)
Min:       0.15               3. offwind-ac: 180.12 (14.5%)
Range:     450.08             4. CCGT: 120.34 (9.7%)
                              5. hydro: 95.67 (7.7%)
```

### 2. Multi-Year Statistics
**Displayed for**: Stacked Bar (All Years), Year Comparison plots

Shows comparative statistics across all available years for trend analysis.

#### Key Metrics (Top Row)
Four colored cards displaying:
- **2025 Total** (Blue): Starting year total
- **2040 Total** (Green): Ending year total
- **Total Growth** (Orange): Percentage change from first to last year (color-coded: green for positive, red for negative)
- **Unique Carriers** (Purple): Total number of unique carriers across all years

#### Year-by-Year Totals (Bottom Left)
Detailed table showing progression:
- **Year**: Each year (2025, 2030, 2035, 2040)
- **Total**: Sum of all carriers for that year
- **Active Carriers**: Number of carriers with positive values
- **Change**: Percentage change from previous year (color-coded)

#### Growth Metrics (Bottom Right)
Summary of trends:
- **Annual Avg Growth**: Average growth rate per year
- **Time Span**: Number of years covered
- **Total Change**: Absolute difference between first and last year

**Example Display:**
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 2025 Total  │ 2040 Total  │Total Growth │   Unique    │
│  1000.50    │  1450.75    │   +45.0%    │  Carriers   │
│             │             │             │     22      │
└─────────────┴─────────────┴─────────────┴─────────────┘

Year-by-Year Totals:          Growth Metrics:
Year | Total   | Active | Change    Annual Avg Growth: +3.0%/year
2025 | 1000.50 |   18   |    -      Time Span:        15 years
2030 | 1150.25 |   19   | +14.9%    Total Change:     +450.25
2035 | 1300.60 |   20   | +13.1%
2040 | 1450.75 |   21   | +11.5%
```

## Color Coding

Summary statistics use color coding for quick insights:

### Single-Year Stats
- **Gray backgrounds**: Neutral metric cards
- **Black text**: Standard values

### Multi-Year Stats
- **Blue card**: Starting year (2025)
- **Green card**: Ending year (2040)
- **Orange card**: Growth percentage
- **Purple card**: Unique carriers count
- **Green text**: Positive growth
- **Red text**: Negative growth/decline
- **Gray text**: No change or N/A

## Use Cases

### 1. Quick Overview
The top row of cards provides an immediate overview of:
- Scale of the metric (Total)
- Distribution characteristics (Mean, Max)
- Data richness (Number of carriers)

### 2. Distribution Analysis
Statistical summary helps understand:
- Central tendency (Median vs Mean)
- Variability (Std Dev, Range)
- Outliers (compare Min/Max to Mean)

### 3. Identifying Key Players
Top Contributors list shows:
- Which carriers dominate the metric
- Relative importance (percentages)
- Concentration vs diversity of the energy mix

### 4. Trend Analysis (Multi-Year)
Year-by-year table reveals:
- Growth trajectory
- Acceleration or deceleration
- Turning points
- Consistency of trends

## Technical Details

### Calculation Methods

**Single-Year Stats:**
- Total: `sum(all_values)`
- Mean: `mean(all_values)`
- Median: `median(all_values)`
- Std Dev: `std(all_values)`
- Range: `max - min`
- Top Contributors: Top 5 carriers by value (positive values only)

**Multi-Year Stats:**
- Total Growth: `((last_year - first_year) / first_year) * 100`
- Annual Avg Growth: `total_growth / years_span`
- Year-over-year Change: `((current_year - previous_year) / previous_year) * 100`
- Active Carriers: Count of carriers with `value > 0`

### Responsive Design

Summary statistics adapt to screen size:
- Desktop: Full width with multi-column layout
- Tablet: Stacked columns
- Mobile: Single column with scrolling

### Performance

- Statistics calculated on-demand when plot updates
- Lightweight calculations (< 100ms typically)
- Cached with plot data for efficiency
- No database queries (works from in-memory data)

## Examples by Metric

### Energy Mix (TWh)
- **Total**: Total generation across all carriers
- **Top Contributors**: Dominant generation sources
- **Growth**: Expansion or reduction of total generation

### Capacity Mix (GW)
- **Total**: Total installed capacity
- **Top Contributors**: Largest capacity sources
- **Growth**: Capacity expansion trends

### System Costs (€M)
- **Total**: Total system cost
- **Top Contributors**: Most expensive components
- **Growth**: Cost evolution over time

### CO2 Emissions (Mt)
- **Total**: Total emissions
- **Top Contributors**: Largest emission sources
- **Growth**: Emission trajectory (ideally negative)

## Integration with Plots

Summary statistics update automatically when you:
- Change year selection
- Change scenario selection
- Change metric selection
- Change plot type
- Filter carriers (stats reflect filtered data)

The statistics always match the data shown in the plot above them, ensuring consistency.

## Tips for Interpretation

1. **Compare Total and Mean**: If very different, indicates uneven distribution
2. **Check Std Dev**: High std dev means high variability between carriers
3. **Look at Top 5**: If top 5 are >80% of total, system is concentrated
4. **Monitor Growth Rates**: Consistent growth vs. fluctuating patterns
5. **Active Carriers**: Increasing count suggests diversification

## Future Enhancements

Potential additions:
- [ ] Export statistics as CSV
- [ ] Add confidence intervals
- [ ] Include year-over-year growth charts
- [ ] Add correlation metrics
- [ ] Custom statistic selection
- [ ] Historical comparison to previous runs
