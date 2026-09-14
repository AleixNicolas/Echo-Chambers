import os
import json
import config
from src import core_engine, data_wrangle, plotting_engine, reporting_engine

def run_simulation_batch():
    print("=== STARTING STOCHASTIC SIMULATION BATCH (DUAL TOPICS) ===")
    print(f"Simulating Topics: {config.SIM_TOPICS}")
    
    for priority_flag in config.LAST_ROUND_PRIORITY:
        priority_str = "TRUE" if priority_flag else "FALSE"
        base_sim_dir = os.path.join(config.SIMULATION_RESULTS_DIR, f"LastRoundPriority_{priority_str}")
        
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            print(f"\n=== Simulating Regime: [{regime_name}] | Priority: {priority_str} ===")
            regime_dir = os.path.join(base_sim_dir, f"Regime_{regime_name}")
            
            for k in config.K_VALUES:
                sweep_dir = os.path.join(regime_dir, f"Sweep_K{k}")
                os.makedirs(sweep_dir, exist_ok=True)
                sweep_aggregator = {}
                
                for n in config.N_VALUES:
                    print(f"\n-> Running N={n}, K={k} under {regime_name} (Priority={priority_str})...")
                    baseline_path = config.get_baseline_path(n, k)
                    if not os.path.exists(baseline_path): continue
                        
                    with open(baseline_path, 'r') as f: baseline_data = json.load(f)
                    global_pool = baseline_data["global_item_pool"]
                    plots_dir = os.path.join(sweep_dir, f"N{n}")
                    os.makedirs(plots_dir, exist_ok=True)

                    print(f"   Running Segregated Treatment ({config.NUM_TRIALS} trials)...")
                    seg_results = core_engine.run_batch(baseline_data["segregated_baseline"], global_pool, regime_rules, config.NUM_TRIALS, priority_flag)
                    
                    print(f"   Running Integrated Treatment ({config.NUM_TRIALS} trials)...")
                    int_results = core_engine.run_batch(baseline_data["integrated_baseline"], global_pool, regime_rules, config.NUM_TRIALS, priority_flag)
                    
                    print("   Formatting Data and Generating Multi-Topic Plots...")
                    results_store = {'segregated': seg_res, 'integrated': int_results}
                    sweep_aggregator[n] = results_store
                    
                    event_log_df = data_wrangle.build_simulation_event_log(results_store)
                    
                    plotting_engine.generate_suite(event_log_df, plots_dir, prefix=f"Sim_N{n}_K{k}", topics=config.SIM_TOPICS)
                    plotting_engine.generate_share_distributions(event_log_df, plots_dir, prefix=f"Sim_N{n}_K{k}", topics=config.SIM_TOPICS)
                
                if sweep_aggregator:
                    reporting_engine.export_sweep_summary(sweep_aggregator, sweep_dir, k, regime_name, priority_str)

    print("\n=== BATCH SIMULATION COMPLETE ===")

if __name__ == "__main__":
    run_simulation_batch()