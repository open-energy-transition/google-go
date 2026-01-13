# Dashboard Verification Checklist

## After Starting the Dashboard

### 1. Check Sub-Scenario Dropdown
Go to **Comparison** tab and check the **Sub-Scenario** dropdown.

**Expected:** You should see approximately 14 options:
```
- baseline
- energy-match-25
- energy-match-50
- hourly-match-25-90
- hourly-match-25-95
- hourly-match-25-98
- hourly-match-25-99
- hourly-match-50-90
- hourly-match-50-95
- hourly-match-50-98
- hourly-match-50-99
- hourly-match-noadd-10-99
- hourly-match-noadd-50-99
- hourly-match-noadd-90-99
```

**If you only see "baseline":** The fix didn't take effect. Try:
1. Stop the dashboard
2. Restart it: `cd dashboard && python app.py`
3. Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)

### 2. Check Main Scenarios Checkboxes
Look at the **Main Scenarios** checkboxes.

**Expected:** All three should be checked by default:
- ☑ CI_25
- ☑ CI_50
- ☑ CI_noadd

**If some are unchecked:** Click to check them.

### 3. Test Baseline Comparison
Settings:
- Year: 2025 (or any year)
- Sub-Scenario: baseline
- Metric: (a) Energy mix
- Main Scenarios: All three checked

**Expected Plot:**
- Should show THREE groups of bars (one for each scenario)
- Title should say: "Comparison: (a) Energy mix (2025) - baseline"
- Subtitle should say: "Showing: CI_25, CI_50, CI_noadd"
- All three should have nearly identical values (e.g., ~3390 TWh total)

**If you only see one scenario (e.g., CI_50):**
- Check that all three checkboxes are actually checked
- Check the plot subtitle - it tells you which scenarios have data
- Try unchecking and re-checking the boxes

### 4. Test Policy Scenario
Settings:
- Year: 2030
- Sub-Scenario: hourly-match-50-99
- Metric: (a) Energy mix
- Main Scenarios: All three checked

**Expected Plot:**
- Should show ONLY ONE bar group (CI_50)
- Title: "Comparison: (a) Energy mix (2030) - hourly-match-50-99"
- Subtitle: "Showing: CI_50"
- This is correct because only CI_50 has "hourly-match-50-99"

**Why?** CI_25 has "hourly-match-25-99" and CI_noadd has "hourly-match-noadd-50-99" - different names!

### 5. Test Individual Tabs
Go to **CI_25** tab:
- Year: 2030
- Scenario: hourly-match-25-99
- Metric: (a) Energy mix
- Plot Type: Bar Chart

**Expected:** Should show detailed breakdown for CI_25's hourly-match-25-99 scenario

Go to **CI_50** tab:
- Same settings but Scenario: hourly-match-50-99

**Expected:** Different values than CI_25 (because 50% vs 25% additionality)

## Common Issues

### Issue 1: Only "baseline" in Sub-Scenario Dropdown
**Cause:** Old code still running (before the fix)
**Solution:**
1. Stop dashboard (Ctrl+C)
2. Restart: `cd dashboard && python app.py`
3. Hard refresh browser

### Issue 2: Only One Scenario Shows in Comparison
**Possible causes:**
a) Only one checkbox is actually checked
   - **Solution:** Check all three boxes manually

b) The selected sub-scenario only exists in one main scenario
   - **Solution:** This is correct behavior! Check the subtitle to see which scenarios have this sub-scenario
   - Try selecting "baseline" to see all three

c) Data loading issue
   - **Solution:** Check terminal output for errors when dashboard starts

### Issue 3: Values Look Wrong
**Check:**
- Are you comparing the right year?
- Are you looking at the right metric?
- For baseline, all three SHOULD be identical
- For policy scenarios, they SHOULD differ

## Data Verification

Run this from terminal to verify data exists:
```bash
cd /home/user/google-go
python -c "
from dashboard.utils.data_loader import DataLoader
data_loader = DataLoader(results_dir='results')
data_loader.load_all_data()

print('\\nAvailable sub-scenarios:')
for scenario in ['CI_25', 'CI_50', 'CI_noadd']:
    stats = data_loader.get_summary_stats(scenario)
    print(f'\\n{scenario}:')
    for s in stats['scenarios']:
        print(f'  - {s}')
"
```

Expected output should list all 6-14 sub-scenarios per main scenario.

## What Should Be Different Between Scenarios

### Baseline (should be ~identical):
All three use the same baseline assumptions, so values should match.

### Policy Scenarios (should differ):

**CI_25 vs CI_50:**
- Different additionality requirements (25% vs 50%)
- More renewable capacity in CI_50
- Different storage deployment
- Different system costs

**CI_noadd vs Others:**
- No additionality requirement
- Can use existing clean energy
- Likely lower costs
- Different capacity expansion patterns

## Need Help?

If things still aren't working:
1. Share a screenshot of the Comparison tab
2. Check terminal for error messages
3. Verify: `ls results/` shows all three folders (CI_25, CI_50, CI_noadd)
4. Verify: Each folder has a results.csv file
