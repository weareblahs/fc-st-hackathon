# EDA Plotting Refactor - Summary

## Overview

Refactored `eda.py` to save **one plot per feature** instead of crowded multi-subplot grids. This makes individual charts readable and easy to share.

## Changes Made

### 1. Configuration Flags

Added plotting mode controls at the top of the `eda()` function:

```python
EDA_PLOT_MODE = 'per_feature'  # or 'grid' for old behavior
DATASET_NAME = 'merged'         # Dataset identifier for output paths
```

### 2. Safe Filename Helper

Added `_safe_name(name)` function to sanitize column names for filesystem-safe filenames:

- Replaces non-alphanumeric characters (except `_`, `.`, `-`) with underscores
- Prevents path issues with special characters in column names

### 3. Refactored Plotting Functions

#### `plot_numeric_univariate()`

- **Per-feature mode**: Saves one histogram per numeric column
- **Output**: `deliverables/{dataset}/univariate/{dataset}_numeric__{column}.svg`
- **Grid mode**: Original behavior (all subplots in one file)
- Includes mean/std in title, 50 bins, sampling for large datasets

#### `plot_categorical_univariate()`

- **Per-feature mode**: Saves one bar chart per categorical column
- **Output**: `deliverables/{dataset}/univariate/{dataset}_categorical__{column}.svg`
- Shows top 20 categories for high-cardinality features
- Displays unique value count in title

#### `plot_numeric_vs_target()`

- **Per-feature mode**: Saves one scatter plot per feature vs target
- **Output**: `deliverables/{dataset}/target/{dataset}_numeric_vs_{target}__{feature}.svg`
- Includes Pearson correlation coefficient and p-value
- Adds regression trendline
- Accepts `dataset_name` parameter

#### `plot_categorical_vs_target()`

- **Per-feature mode**: Saves one box plot per feature vs target
- **Output**: `deliverables/{dataset}/target/{dataset}_categorical_vs_{target}__{feature}.svg`
- Performs Kruskal-Wallis test and displays p-value
- Limits to top 15 categories per feature
- Saves KW results to: `deliverables/{dataset}/summaries/categorical_vs_{target}_kruskal.csv`

### 4. Normalized Output Paths

All outputs now save under `deliverables/{dataset}/...`:

| Output Type           | Old Path                     | New Path                            |
| --------------------- | ---------------------------- | ----------------------------------- |
| Correlation heatmaps  | `deliverables/correlations/` | `deliverables/merged/correlations/` |
| Route efficiency dist | `deliverables/target/`       | `deliverables/merged/target/`       |
| Temporal plots        | `deliverables/target/`       | `deliverables/merged/target/`       |
| Summaries (CSV)       | `deliverables/summaries/`    | `deliverables/merged/summaries/`    |
| GPS map (HTML)        | `deliverables/maps/`         | `deliverables/merged/maps/`         |
| Univariate plots      | `deliverables/univariate/`   | `deliverables/merged/univariate/`   |
| Target mapping        | `deliverables/target/`       | `deliverables/merged/target/`       |

### 5. Performance Guardrails

All guardrails maintained:

- `SAMPLE_SIZE_PLOT = 100_000`: Max rows for plotting (prevents memory issues)
- `max_plots` parameter: Limits number of features plotted
- Top 20 categories: For categorical univariate
- Top 15 categories: For categorical vs target box plots
- Sampling applied consistently across all plot types

## Usage

### Default (per-feature mode)

```python
from eda import eda
eda('your_uuid_here')
```

Outputs individual SVG files for each feature.

### Grid mode (quick overview)

Edit `eda.py` and change:

```python
EDA_PLOT_MODE = 'grid'  # Switch to grid mode
```

### Change dataset identifier

```python
DATASET_NAME = 'your_dataset_name'  # Changes output folder
```

## Output Structure

```
deliverables/
└── merged/
    ├── univariate/
    │   ├── merged_numeric__speed_kmh.svg
    │   ├── merged_numeric__lat.svg
    │   ├── merged_categorical__vehicle_id.svg
    │   └── ...
    ├── target/
    │   ├── route_eff_distribution.svg
    │   ├── route_eff_temporal.svg
    │   ├── merged_numeric_vs_route_eff__speed_kmh.svg
    │   ├── merged_categorical_vs_route_eff__vehicle_id.svg
    │   └── ...
    ├── correlations/
    │   ├── pearson_correlation.svg
    │   └── spearman_correlation.svg
    ├── maps/
    │   └── gps_route_efficiency_map.html
    └── summaries/
        ├── route_eff_correlations.csv
        ├── route_eff_temporal.csv
        ├── categorical_vs_route_eff_kruskal.csv
        └── numeric_summary.csv
```

## Benefits

✅ **Readable**: Each chart is clear and sized appropriately  
✅ **Shareable**: Individual files easy to include in reports/presentations  
✅ **Organized**: Consistent folder structure per dataset  
✅ **Flexible**: Can switch between per-feature and grid modes  
✅ **Safe**: Handles special characters in column names  
✅ **Maintained**: All original sampling and performance limits preserved

## Backward Compatibility

- Set `EDA_PLOT_MODE = 'grid'` to restore original behavior
- All function signatures maintain backward compatibility with optional `mode` parameter
- Default behavior is now 'per_feature' for better readability
