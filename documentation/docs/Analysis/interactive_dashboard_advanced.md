# Interactive Dashboard - Advanced Guide

This guide covers technical architecture, customization, performance optimization, and development topics for the Google-Go Interactive Dashboard.

---

## Technical Architecture

### Technology Stack

**Core Framework:**

- **Dash (Plotly)**: Web application framework for Python
- **Plotly**: Interactive visualization library
- **Dash Bootstrap Components**: UI component library

**Data Processing:**

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Parquet/CSV**: Data storage formats

**Deployment:**

- **Flask**: WSGI web server (built into Dash)
- **Gunicorn**: Production WSGI server (optional)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (User Interface)                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Tab 1    │ │ Tab 2    │ │ Tab 3    │ │ Tab 4    │ ...  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────┴────────────────────────────────────┐
│                   Dash Application (app.py)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Callback Management Layer                │  │
│  │  • User input handlers                                │  │
│  │  • Plot generation                                    │  │
│  │  • Data filtering                                     │  │
│  └────────┬─────────────────────────────────────┬────────┘  │
└───────────┼─────────────────────────────────────┼───────────┘
            │                                     │
┌───────────┴───────────┐           ┌────────────┴────────────┐
│   Layout Components   │           │   Utility Modules       │
│   (layouts/)          │           │   (utils/)              │
│                       │           │                         │
│ • single_scenario     │           │ • DataLoader            │
│ • cross_scenario      │           │   - CSV/Parquet I/O    │
│ • deadzone            │           │   - Caching            │
│ • timeseries          │           │   - Data filtering     │
│ • insights            │           │                         │
└───────────────────────┘           │ • ColorMapper           │
                                    │   - Carrier colors     │
                                    │   - Consistent themes  │
                                    └────────┬────────────────┘
                                             │
┌────────────────────────────────────────────┴────────────────┐
│                    Data Layer (results/)                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results.csv (Consolidated)                           │  │
│  │  • Multi-level headers (year, scenario, scope)        │  │
│  │  • Multi-level index (metric, y-label, carrier)       │  │
│  │  • ~145 rows × ~400 columns                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results_frontier.csv                                 │  │
│  │  • Frontier analysis data                             │  │
│  │  • Multiple scenarios, years, countries               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  results_time_series.parquet                          │  │
│  │  • Hourly data (8,760 hours/year)                     │  │
│  │  • ~millions of data points                           │  │
│  │  • Chunked loading with caching                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  colors.csv                                           │  │
│  │  • Carrier color mappings                             │  │
│  │  • Ensures visual consistency                         │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Startup**: `DataLoader` loads all consolidated results into memory
2. **User Selection**: User selects year, scenario, metric via dropdowns
3. **Callback Trigger**: Dash detects input changes and fires registered callbacks
4. **Data Filtering**: `DataLoader` filters data based on user selections
5. **Plot Generation**: Callback creates Plotly figure with filtered data
6. **Color Mapping**: `ColorMapper` applies consistent colors to carriers
7. **Rendering**: Plotly renders interactive visualization in browser

### File Structure

```
dashboard/
├── app.py                          # Main application entry point
├── callbacks.py                    # All callback functions (17 callbacks)
├── layouts/                        # Tab-specific layouts
│   ├── __init__.py
│   ├── single_scenario_layout.py   # Single scenario analysis
│   ├── cross_scenario_layout.py    # Multi-scenario comparison
│   ├── deadzone_layout.py          # Frontier analysis
│   ├── timeseries_layout.py        # Hourly timeseries
│   └── insights_layout.py          # Statistical insights (static)
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and caching
│   └── colors.py                   # Color mapping utilities
└── assets/                         # Static assets (CSS, images)
    └── custom.css                  # Custom styling
```

---

## Callback Architecture

The dashboard uses Dash's callback system for interactivity:

```python
@app.callback(
    Output('plot-id', 'figure'),      # What to update
    [Input('dropdown-id', 'value')]   # What triggers the update
)
def update_plot(selected_value):
    # Filter data based on input
    data = data_loader.get_data(selected_value)
    # Generate plot
    fig = create_plot(data)
    return fig
```

**17 callbacks** handle all dashboard interactivity:

- 5 for Single Scenario Analysis
- 1 for Cross-Scenario Comparison
- 3 for Dead Zone Analysis
- 5 for Timeseries Exploration
- 3 for dynamic dropdown population

### Callback Best Practices

1. **Keep callbacks focused**: Each callback should handle one specific UI update
2. **Use caching**: Data is cached in DataLoader to avoid redundant I/O
3. **Handle errors gracefully**: Return empty figures with error messages
4. **Validate inputs**: Check for None/invalid values before processing
5. **Minimize data transfer**: Only send necessary data to browser

---

## Multi-Index Data Handling

Pandas MultiIndex is used extensively for efficient data organization:

```python
# Column MultiIndex
columns = pd.MultiIndex.from_tuples([
    (2025, 'baseline', 'system'),
    (2025, 'energy-match-25', 'system'),
    # ...
], names=['year', 'scenario', 'scope'])

# Row MultiIndex
index = pd.MultiIndex.from_tuples([
    ('(a) Energy mix', 'Net generation (TWh)', 'solar'),
    ('(a) Energy mix', 'Net generation (TWh)', 'onwind'),
    # ...
], names=['Results', 'y_label', 'carrier'])

# Fast filtering with IndexSlice
idx = pd.IndexSlice
data = df.loc[idx['(a) Energy mix', :, :], idx[2025, 'baseline', :]]
```

### Why MultiIndex?

- **Fast filtering**: O(log n) lookups vs O(n) for single index
- **Natural hierarchy**: Reflects data structure (metric → label → carrier)
- **Memory efficient**: Shared index objects reduce memory overhead
- **Pandas optimized**: Built-in support for aggregations and operations

---

## Performance Optimization

### Memory Management

**Typical memory usage:**

- Dashboard base: ~100-200 MB
- Consolidated results: ~50-100 MB
- Frontier data: ~10-20 MB
- Timeseries cache: ~500 MB (50 queries)
- **Total**: ~1-2 GB for typical usage

**For large datasets:**

1. Use Parquet format (10-20x compression vs CSV)
2. Limit timeseries cache size
3. Use time range filtering instead of loading full year
4. Deploy with adequate RAM (4GB+ recommended)

### Loading Speed

**Startup time:**

- Consolidated results.csv: ~1-2 seconds
- Frontier data: ~0.5 seconds
- Timeseries metadata: ~5-10 seconds (parquet), ~30-60 seconds (CSV)

**Query response time:**

- Aggregated plots: ~0.1-0.5 seconds
- Timeseries plots (cached): ~0.2-0.5 seconds
- Timeseries plots (uncached): ~2-10 seconds (parquet), ~10-30 seconds (CSV)

### Optimization Recommendations

**1. Convert timeseries to Parquet**:

```python
import pandas as pd
df = pd.read_csv('results_time_series.csv')
df.to_parquet('results_time_series.parquet', compression='snappy')
```

**2. Increase cache size** for repeated queries (edit `data_loader.py`):

```python
if len(self.timeseries_cache) > 100:  # Increase to 100
    self.timeseries_cache.pop(next(iter(self.timeseries_cache)))
```

**3. Use shorter time ranges** for exploratory analysis

**4. Deploy with SSD** for faster I/O

---

## Advanced Customization

### Modifying Plot Types

To add new visualizations, edit `dashboard/callbacks.py`:

```python
def create_plot(data, metric, plot_type, color_mapper):
    if plot_type == 'my_custom_plot':
        fig = go.Figure()
        # Add your custom plot logic
        for carrier in data.index:
            fig.add_trace(go.Scatter(
                x=years,
                y=data.loc[carrier],
                name=carrier,
                marker=dict(color=color_mapper.get_color(metric, carrier))
            ))
        return fig
```

Then add to dropdown in `layouts/single_scenario_layout.py`:

```python
options=[
    {'label': 'Bar Chart', 'value': 'bar'},
    {'label': 'My Custom Plot', 'value': 'my_custom_plot'},
    # ...
]
```

### Adding New Tabs

To add a new analysis tab:

**1. Create layout** in `layouts/my_new_tab.py`:
```python
def create_my_tab_layout(data_loader):
    return dbc.Container([
        html.H3("My New Analysis"),
        dcc.Graph(id='my-plot'),
        # Add controls...
    ])
```

**2. Register in app.py**:
```python
from layouts import my_new_tab

# Add tab
dcc.Tab(label='My Analysis', value='my-tab')

# Add layout
html.Div(my_new_tab.create_my_tab_layout(data_loader),
         id='my-content', style={'display': 'none'})
```

**3. Add callback** in `callbacks.py`:
```python
@app.callback(
    Output('my-plot', 'figure'),
    [Input('my-selector', 'value')]
)
def update_my_plot(value):
    # Your plot logic
    return fig
```

4. **Update tab visibility** callback in `app.py`

### Adjusting Data Caching

Edit `dashboard/utils/data_loader.py`:

```python
# Change cache size (default: 50 entries)
if len(self.timeseries_cache) > 100:  # Increase to 100
    self.timeseries_cache.pop(next(iter(self.timeseries_cache)))
```

**Trade-off**: Larger cache → more memory usage, faster repeated queries

---

## API Reference

### DataLoader Class

**Location**: `dashboard/utils/data_loader.py`

**Methods:**

```python
# Load all data at startup
data_loader.load_all_data()

# Get summary statistics
stats = data_loader.get_summary_stats()
# Returns: {'years': [...], 'scenarios': [...], 'metrics': [...]}

# Get filtered data
data = data_loader.get_data(
    year=2035,              # int or None
    scenario_name='baseline',  # str or None
    metric='(a) Energy mix'     # str or None
)

# Get carriers for a metric
carriers = data_loader.get_carriers_for_metric('(a) Energy mix')

# Get frontier data
frontier = data_loader.get_frontier_data(
    year=2035,
    country='EU'
)

# Get frontier countries
countries = data_loader.get_frontier_countries(year=2035)

# Get timeseries metadata
metadata = data_loader.get_timeseries_metadata()

# Load timeseries data
data, timestamps = data_loader.load_timeseries_data(
    year=2035,
    scenarios=['baseline', 'energy-match-25'],
    ts_type='Electricity Balance',
    country='EU',
    carriers=['solar', 'onwind'],
    time_range='week_winter'
)
```

### ColorMapper Class

**Location**: `dashboard/utils/colors.py`

**Methods:**

```python
# Initialize with colors.csv
color_mapper = ColorMapper('../results/colors.csv')

# Get color for carrier in metric
color = color_mapper.get_color('(a) Energy mix', 'solar')

# Get all colors for a metric
colors = color_mapper.get_colors_for_metric('(a) Energy mix')

# Format scenario names for display
display_name = format_scenario_name('hourly-match-50-90')
# Returns: "Hourly 90% (CI 50%)"
```

---

## Development

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to all functions
- Comment complex logic

### Testing Changes

1. Test with full dataset
2. Verify all tabs load correctly
3. Check all dropdown combinations
4. Test edge cases (empty data, single year, etc.)
5. Verify timeseries loading with both Parquet and CSV

### Debugging Tips

1. **Check browser console** (F12) for JavaScript errors
2. **Review Python logs** in terminal for backend errors
3. **Use Dash debug mode** (default in dev):
   ```python
   app.run_server(debug=True)
   ```
4. **Print callback inputs** to verify data flow
5. **Test with minimal data** to isolate issues

---

## Contributing

### Submitting Updates

1. Document all changes in code comments
2. Update user guide if adding features
3. Test on clean Python environment
4. Ensure backward compatibility with existing data

### Code Organization

- **Layouts**: UI components only, no business logic
- **Callbacks**: Data processing and plot generation
- **Utils**: Reusable data loading and utility functions
- **Assets**: Static CSS/images only

---

## Further Resources

- **Dash Documentation**: https://dash.plotly.com/
- **Plotly Python**: https://plotly.com/python/
- **Pandas MultiIndex**: https://pandas.pydata.org/docs/user_guide/advanced.html
- **Parquet Format**: https://parquet.apache.org/
- **Dash Bootstrap Components**: https://dash-bootstrap-components.opensource.faculty.ai/

---

## Production Deployment

### Using Gunicorn

For production environments:

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050 --workers 4 --timeout 300 --log-level info
```

**Configuration options:**

- `--workers 4`: Number of worker processes (use 2-4 × CPU cores)
- `--timeout 300`: Request timeout in seconds (5 minutes for large queries)
- `-b 0.0.0.0:8050`: Bind address (0.0.0.0 = all interfaces)
- `--log-level info`: Logging verbosity (debug, info, warning, error)

### Using systemd (Linux)

Create service file `/etc/systemd/system/dashboard.service`:

```ini
[Unit]
Description=Google-Go Dashboard
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/google-go/dashboard
ExecStart=/usr/bin/gunicorn app:server -b 0.0.0.0:8050 --workers 4 --timeout 300
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable dashboard
sudo systemctl start dashboard
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ dashboard/
COPY results/ results/

WORKDIR /app/dashboard

EXPOSE 8050

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "4", "--timeout", "300"]
```

Build and run:
```bash
docker build -t dashboard .
docker run -p 8050:8050 dashboard
```

### Nginx Reverse Proxy

For production deployments behind Nginx:

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeout settings
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

---

## Security Considerations

### Data Access

- Dashboard serves data to all users - ensure data is not confidential
- No authentication by default - add if needed
- Consider IP whitelisting for internal deployments

### Adding Authentication

Use `dash-auth` for basic authentication:

```python
import dash_auth

# Add after creating app
VALID_USERNAME_PASSWORD_PAIRS = {
    'username': 'password'
}

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
```

For production, use proper authentication (OAuth, LDAP, etc.)

### HTTPS

Always use HTTPS in production:
- Terminate SSL at Nginx/load balancer
- Use Let's Encrypt for free certificates
- Redirect HTTP to HTTPS

---

## Monitoring and Logging

### Application Logging

Add logging to `app.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Performance Monitoring

Monitor key metrics:
- Memory usage: `ps aux | grep python`
- Request latency: Log callback execution time
- Error rates: Count exceptions in logs
- Active users: Track concurrent connections

### Health Checks

Add health check endpoint in `app.py`:

```python
from flask import jsonify

@app.server.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200
```

---

## Troubleshooting Advanced Issues

### High Memory Usage

**Symptoms**: Dashboard consumes >4GB RAM

**Solutions**:

1. Reduce timeseries cache size
2. Clear cache periodically: (see below)
3. Use time-based cache eviction
4. Deploy with more RAM

```python
# Add to data_loader.py
def clear_cache(self):
    self.timeseries_cache.clear()
```

### Slow Callback Execution

**Symptoms**: Plots take >5 seconds to render

**Solutions**:

1. Profile callbacks: (see below)
2. Optimize data filtering queries
3. Pre-compute expensive calculations
4. Use Parquet for timeseries

```python
import time
start = time.time()
# ... callback code ...
print(f"Callback took {time.time() - start:.2f}s")
```

### Concurrent User Issues

**Symptoms**: Dashboard slows with multiple users

**Solutions**:

1. Increase Gunicorn workers
2. Use Redis for shared caching
3. Deploy multiple instances with load balancer
4. Consider stateless architecture

---

## Support and Community

For technical questions or contributions:

1. Review this advanced guide thoroughly
2. Check the main user guide for usage questions
3. Search GitHub issues for similar problems
4. Contact the development team with specific technical questions

**Dashboard Status**: Production-ready, actively maintained
