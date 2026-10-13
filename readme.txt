ECHO CHAMBER & POLARIZATION SIMULATION PIPELINE (v3.0)
This repository evaluates empirical oTree network experiments and runs
large-scale Monte Carlo stochastic simulations to analyze algorithmic
interventions on echo chamber formation.
The pipeline supports Dual-Topic multiplexing, decoupling physical network
topology (Base Topic) from information diffusion (Secondary Topic) to
accurately measure cross-pressured sharing behaviors, while rigorously
stress-testing statistical power against real-world human behavioral noise.
--- ARCHITECTURE ------------------------------------------------------------
Root Directory (Execution Scripts):
config.py                  : Central hyper-parameters, Share Probability Matrices, and Master Noise Toggles.
pipeline_baselines.py      : Generates static N/K topology baselines (PPM graphs) and starting states.
pipeline_empirical.py      : Master script for empirical data pipelines (reads oTree CSVs).
pipeline_simulations.py    : Master script for stochastic simulation sweeps and visual reports.
pipeline_power.py          : Runs Monte Carlo simulations to calculate statistical power (Mann-Whitney U).
src/ Directory (Engines & Analysis):
simulation/core_engine.py  : Pure mathematical simulation logic. Executes trials and dynamically applies behavioral/demographic noise.
simulation/networks.py     : Handles network generation
empirical/otree_parser.py  : Handles raw oTree ingestion and validations.
empirical/prolific_admin.py: Handles participation
analysis/event_logger.py   : Universal Data Adapter. Converts raw oTree and simulation dicts into a standardized Event Log schema.
analysis/plotting.py       : Unified plotting suite. Processes standardized logs to generate cross-tabulated visualizations.
analysis/reporting.py      : Massive text summarization, dropout tracking, and H3 Pooled Regression exports.
analysis/metrics.py        : Mathematical definitions for polarization gaps and structural outcomes.
--- NOISE (config.py) ------------------------------------
Lurkers                   : config.SIM_LURKER_FRACTION randomly silences nodes per trial to stress-test network flow.
Aleatoric                 : config.USE_ALEATORIC_NOISE applies a blind 50/50 action probability (e.g., 10% error rate).
Composition               : config.USE_COMPOSITION_NOISE dynamically resamples secondary-topic demographics per trial using config.EMPIRICAL_COMPOSITION_MATRIX.
--- CONFIGURATION SWITCHES (config.py) --------------------------------------
Before running, verify the Master Switches in config.py:
EMPIRICAL_BASE_TOPIC       : The topic determining physical network clustering (e.g., "imm").
EMPIRICAL_SIDE_TOPIC       : The secondary topic diffusing through the network (e.g., "climate").
INVERT_OPINIONS_FOR        : Topics requiring a 1-5 scale inversion (so '5' always equals Left).
DETAILED_SIM_N             : List of specific N values to generate plots for during sweeps.
--- EXECUTION ORDER ---------------------------------------------------------
Configure settings in config.py.
Run python pipeline_baselines.py to lock network topologies.
Drop oTree CSVs and network_map.json into data/empirical/<TRIAL_ID>/raw/.
Run python pipeline_empirical.py to analyze real-world empirical results.
Run python pipeline_simulations.py for detailed algorithmic tests and plots.
Run python pipeline_power.py for statistical confidence sweeps.
--- PLOT INDEX --------------------------------------------------------------
Plots 01A & 01B (Diet/Amp)     : Chronological stacked bar charts breaking down the 5-category
ideological composition of the Incoming Feed (01A) vs. the
Outgoing Shares (01B), separated by physical network chamber.
Plots 02 & 03 (Heatmaps)       : Matrix visualizations of feed composition and sharing rates,
cross-tabulated into distinct graphs for the Left and Right
physical chambers.
Plot 04 (Sourcing)             : Tracks where items originate (Initial Pool, Neighbor Shares,
or System Backlog) over the duration of the trial.
Plot 05 & 06 (Distributions)   : Histograms displaying the spread of item exposure and active
sharing volume by participant leaning.
Plot 07 (Correlations)         : Pearson correlation heatmaps safely anchored to Phase 1
(pre-treatment) baseline opinions to prevent endogeneity.
=============================================================================