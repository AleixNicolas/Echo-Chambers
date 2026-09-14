=============================================================================
 ECHO CHAMBER & POLARIZATION SIMULATION PIPELINE
=============================================================================

This repository evaluates empirical oTree network experiments and runs 
large-scale Monte Carlo stochastic simulations to analyze algorithmic 
interventions on echo chamber formation.

--- ARCHITECTURE ------------------------------------------------------------
Root Directory (Execution Scripts):
- config.py                  : Central hyper-parameters, paths, and mappings.
- initialize_experiment.py   : Generates static N/K topology baselines.
- run_experimental.py        : Master script for empirical data pipelines.
- run_simulations.py         : Master script for stochastic simulation sweeps.
- run_power_analysis.py      : Runs Mann-Whitney U tests on MC sweeps.

src/ Directory (Engines):
- admin_checks.py            : Validates demographics and tracks attrition.
- core_engine.py             : Pure mathematical simulation logic.
- data_wrangle.py            : Universal Data Adapters. Converts raw oTree 
                               and simulation dicts into standardized Pandas logs.
- networks.py                : Graph topologies (PPM) and network plotting.
- plotting_engine.py         : Unified plotting suite (takes standardized logs).
- reporting_engine.py        : Massive text summarization and export functions.
- stats_suite.py             : Matrix calculations and MWU extractors.

--- EXECUTION ORDER ---------------------------------------------------------
1. Configure settings in `config.py` (set `CURRENT_EMPIRICAL_RUN_ID`).
2. Run `python initialize_experiment.py` to lock network topologies.
3. Drop oTree CSVs into `data/empirical/<RUN_ID>/raw/`.
4. Run `python run_experimental.py` for real-world results.
5. Run `python run_simulations.py` for algorithmic tests.
6. Run `python run_power_analysis.py` for statistical confidence sweeps.
=============================================================================