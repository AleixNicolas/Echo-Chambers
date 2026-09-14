=============================================================================
 ECHO CHAMBER & POLARIZATION SIMULATION PIPELINE (v2.0)
=============================================================================

This repository evaluates empirical oTree network experiments and runs 
large-scale Monte Carlo stochastic simulations to analyze algorithmic 
interventions on echo chamber formation. 

With the v2.0 update, the pipeline fully supports Dual-Topic multiplexing, 
decoupling physical network topology (Base Topic) from information diffusion 
(Secondary Topic) to accurately measure cross-pressured sharing behaviors.

--- ARCHITECTURE ------------------------------------------------------------
Root Directory (Execution Scripts):
- config.py                  : Central hyper-parameters, Master Switches (Base/Side topics), and mappings.
- generate_baselines.py      : Generates static N/K topology baselines (PPM graphs) and starting states.
- run_experimental.py        : Master script for empirical data pipelines (reads oTree CSVs).
- run_simulations.py         : Master script for stochastic simulation sweeps and bot models.
- run_power_analysis.py      : Runs MWU statistical confidence sweeps on Monte Carlo data.

src/ Directory (Engines):
- admin_checks.py            : Validates demographics and tracks participant attrition.
- core_engine.py             : Pure mathematical simulation logic for algorithmic agents.
- data_wrangle.py            : Universal Data Adapter. Converts raw oTree and simulation dicts 
                               into a standardized Pandas Event Log, dynamically assigning 
                               participants to physical Left/Right structural chambers.
- networks.py                : Graph topologies (Planted Partition Model) and network plotting.
- plotting_engine.py         : Unified plotting suite. Processes standardized logs to generate 
                               visualizations cross-tabulated by physical structural chambers.
- reporting_engine.py        : Massive text summarization, dropout tracking, and export functions.
- stats_suite.py             : Matrix calculations and MWU extractors.

--- CONFIGURATION SWITCHES (config.py) --------------------------------------
Before running, verify the Master Switches in `config.py`:
- EMPIRICAL_BASE_TOPIC       : The topic determining physical network clustering (e.g., "imm").
- EMPIRICAL_SIDE_TOPIC       : The secondary topic diffusing through the network (e.g., "climate").
- INVERT_OPINIONS_FOR        : Topics requiring a 1-5 scale inversion (so '5' always equals Left).

--- EXECUTION ORDER ---------------------------------------------------------
1. Configure settings in `config.py` (set `CURRENT_EMPIRICAL_TRIAL_ID`).
2. Run `python generate_baselines.py` to lock network topologies.
3. Drop oTree CSVs and `network_map.json` into `data/empirical/<TRIAL_ID>/raw/`.
4. Run `python run_experimental.py` for real-world empirical results.
5. Run `python run_simulations.py` for algorithmic tests and agent simulations.
6. Run `python run_power_analysis.py` for statistical confidence sweeps.

--- PLOT INDEX --------------------------------------------------------------
- Plot 01 (Diet by Chamber)      : Proportion of Left/Center/Right news consumed, grouped 
                                   securely by the physical Left vs. Right network chambers.
- Plots 02 & 03 (Heatmaps)       : Matrix visualizations of feed composition and sharing rates, 
                                   cross-tabulated into distinct graphs for the Left and Right 
                                   physical chambers to expose cross-pressured behaviors.
- Plot 04 (Sourcing)             : Tracks where items originate (Initial Pool, Neighbor Shares, 
                                   or System Backlog) over the duration of the trial.
- Plot 05 & 06 (Distributions)   : Histograms displaying the spread of item exposure and active 
                                   sharing volume by participant leaning.
- Plot 07 (Correlations)         : Pearson correlation heatmaps safely anchored to Phase 1 
                                   (pre-treatment) baseline opinions to prevent endogeneity.
=============================================================================