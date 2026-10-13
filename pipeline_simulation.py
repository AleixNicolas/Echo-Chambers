import json
import random
import numpy as np
from pathlib import Path

import config
from src.simulation import core_engine
from src.analysis import event_logger, plotting, reporting

logger = config.setup_logging("simulation_pipeline.log")

def set_global_seed(seed_value):
    """Enforces absolute reproducibility across all stochastic libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")

def run_simulation_batch():
    logger.info(f"=== STARTING STOCHASTIC SIMULATIONS: {config.SIMULATION_STUDY_NAME} ===")
    
    for priority_flag in config.LAST_ROUND_PRIORITY:
        priority_str = "TRUE" if priority_flag else "FALSE"
        
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            logger.info(f"=== Simulating Regime: [{regime_name}] | Priority: {priority_str} ===")
            
            for k in config.K_VALUES:
                sweep_dir = config.get_sim_sweep_dir(priority_flag, regime_name, k)
                sweep_dir.mkdir(parents=True, exist_ok=True)
                sweep_aggregator = {}
                
                for n in config.N_VALUES:
                    logger.info(f"-> Running N={n}, K={k} under {regime_name} (Priority={priority_str})...")
                    baseline_path = config.get_baseline_path(n, k)
                    if not baseline_path.exists(): 
                        logger.warning(f"Baseline missing for N={n}, K={k}. Skipping.")
                        continue
                        
                    with open(baseline_path, 'r') as f: 
                        baseline_data = json.load(f)
                    global_pool = baseline_data["global_item_pool"]
                    
                    # NOTE: Demographics/Composition Noise is now dynamically drawn per-trial 
                    # inside core_engine.run_batch, so we pass the raw baseline directly!
                    seg_results = core_engine.run_batch(baseline_data["segregated_baseline"], global_pool, regime_rules, config.NUM_TRIALS, 'segregated', priority_flag)
                    int_results = core_engine.run_batch(baseline_data["integrated_baseline"], global_pool, regime_rules, config.NUM_TRIALS, 'integrated', priority_flag)
                    
                    results_store = {'segregated': seg_results, 'integrated': int_results}
                    sweep_aggregator[n] = results_store
                    
                    if n in config.DETAILED_SIM_N:
                        logger.info("   Generating detailed visual suite & topic reports...")
                        n_dir = sweep_dir / f"N{n}"
                        n_plots = n_dir / "plots"
                        n_reports = n_dir / "reports"
                        n_plots.mkdir(parents=True, exist_ok=True)
                        n_reports.mkdir(parents=True, exist_ok=True)
                        
                        event_log_df = event_logger.build_simulation_event_log(results_store)
                        
                        if event_log_df.empty:
                            logger.warning(f"   [!] Event log empty for N={n}. Skipping plots.")
                            continue
                            
                        # 1. Generate Standard Plots
                        plotting.generate_suite(event_log_df, n_plots, prefix="", topics=config.SIM_TOPICS)
                        plotting.generate_share_distributions(event_log_df, n_plots, prefix="", topics=config.SIM_TOPICS)

                        # 2. Feed Slant Variance
                        for topic in config.SIM_TOPICS:
                            plotting.plot_feed_slant_variance(event_log_df, n_plots, topic, prefix="")

                        # 3. Comprehensive Summary Export
                        reporting.export_comprehensive_summary(
                            event_log_df, n_reports, is_empirical=False, topics=config.SIM_TOPICS
                        )
                        
                        # 4. Export H3 Analysis (Demographic Control & Regression)
                        reporting.export_h3_report(event_log_df, n_reports, config.SIM_TOPICS)
                        reporting.export_h3_pooled_regression(event_log_df, n_reports, config.SIM_TOPICS)
                        
                if sweep_aggregator:
                    agg_file = sweep_dir / 'sweep_aggregator.json'
                    with open(agg_file, 'w') as f: 
                        json.dump(sweep_aggregator, f)
                    reporting.export_sweep_summary(sweep_aggregator, sweep_dir, k, regime_name, priority_str)

    logger.info(f"=== BATCH SIMULATION COMPLETE. Saved in: {config.SIMULATION_RESULTS_DIR} ===")

if __name__ == "__main__":
    set_global_seed(config.MASTER_SEED)
    run_simulation_batch()