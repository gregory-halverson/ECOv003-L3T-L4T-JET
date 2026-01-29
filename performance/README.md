# Performance Analysis Tools

This directory contains scripts and tools for analyzing the performance of the JET table processing pipeline.

## analyze_table_processing.py

Performance analysis script that runs an input CSV file through `process_JET_table`, displays all console log output in real time, and reports detailed performance metrics.

### Features

- **Real-time logging**: See all log output as it happens
- **Detailed timing**: Shows elapsed time for each processing stage (via pytictoc)
- **Performance metrics**: Calculates throughput (rows/second) and time per row
- **Memory tracking**: Reports memory usage of input and output DataFrames
- **Flexible input**: Accepts any CSV file with the required columns
- **Optional output saving**: Can save processed results to a new CSV file

### Usage

```bash
# Run with default input file (ECOv002-cal-val-JET-inputs.csv)
python analyze_table_processing.py

# Run with a specific CSV file
python analyze_table_processing.py path/to/your/data.csv

# Enable verbose (DEBUG level) logging
python analyze_table_processing.py -v

# Save output to a file
python analyze_table_processing.py --output results.csv

# Run in online mode (allows data downloads, slower)
python analyze_table_processing.py --online

# View help
python analyze_table_processing.py --help
```

### Example Output

```
================================================================================
  JET Table Processing Performance Analysis
================================================================================
Input file: /path/to/ECOv002-cal-val-JET-inputs.csv
Offline mode: True
Log level: INFO

================================================================================
  Loading Input Data
================================================================================
2026-01-28 10:30:45 - __main__ - INFO - Reading CSV file: /path/to/ECOv002-cal-val-JET-inputs.csv
2026-01-28 10:30:45 - __main__ - INFO - Loaded 1066 rows in 0.234 seconds
2026-01-28 10:30:45 - __main__ - INFO - Input columns: 128
2026-01-28 10:30:45 - __main__ - INFO - Memory usage: 12.45 MB


================================================================================
  Processing JET Table
================================================================================
2026-01-28 10:30:45 - process_JET_table - INFO - starting JET table processing
2026-01-28 10:30:45 - process_JET_table - INFO - started extracting geometry from JET input table
2026-01-28 10:30:45 - process_JET_table - INFO - completed extracting geometry from JET input table (0.023 seconds)
2026-01-28 10:30:45 - process_JET_table - INFO - started extracting time from JET input table
2026-01-28 10:30:45 - process_JET_table - INFO - completed extracting time from JET input table (0.012 seconds)
...
2026-01-28 10:30:52 - process_JET_table - INFO - JET table processing complete (7.234 seconds)


================================================================================
  Processing Complete
================================================================================
2026-01-28 10:30:52 - __main__ - INFO - Output has 1066 rows and 145 columns
2026-01-28 10:30:52 - __main__ - INFO - New columns added: 17
2026-01-28 10:30:52 - __main__ - INFO - Output memory usage: 15.67 MB


================================================================================
  PERFORMANCE SUMMARY
================================================================================
  Total execution time:    7.523 seconds
  JET processing time:     7.234 seconds
  Throughput:              147.36 rows/second
  Time per row:            6.78 ms
================================================================================
```

### Requirements

- pandas
- pytictoc
- All other ECOv003_L3T_L4T_JET dependencies

### Input CSV Format

The input CSV must contain the required columns for `process_JET_table`:
- ST_C (surface temperature in Celsius)
- emissivity
- NDVI
- albedo
- Ta_C (air temperature)
- Tmin_C (minimum air temperature)
- RH (relative humidity)
- SM (soil moisture)
- time_UTC
- geometry (or lat/lon)

See the ECOv002-cal-val-JET-inputs.csv for a complete example.
