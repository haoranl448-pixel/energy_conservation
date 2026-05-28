# Main Pipeline Scripts

This folder contains the scripts called by `scripts_new/main.py`, numbered in execution order.

1. `01_data_process.py` - clean raw data into processed section files
2. `02_train_residual_new.py` - train residual energy models
3. `03_build_class_lookup_tables.py` - build class lookup and standard time tables
4. `04_train_ATO_v8.py` - extract ATO class phase templates
5. `05_simulate_ATO_v8.py` - generate ATO speed curves
6. `06_ato_generated_results_energy.py` - calculate generated-curve energy menu
7. `07_full_line_validation_results.py` - calculate historical baseline
8. `08_ato_class_globall_v2.py` - run DP schedule optimization

Older unnumbered copies are kept for reference, but `main.py` uses the numbered files above.

DP station scope is controlled from `main.py` with `--line-scope`:

- omit it, or use `full`, to run the full forward line from `布政-张家潭` to `镇海大道-骆驼桥`.
- use a number, such as `--line-scope 5`, to run the first N forward sections only.
