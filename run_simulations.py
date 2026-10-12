import json
import math
import random
import numpy as np
import config
from src import core_engine, data_wrangle, plotting_engine, reporting_engine

def assign_dual_topics(network_baseline):
    nodes = network_baseline.get('nodes', [])
    node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
    base_groups = {cat: [] for cat in config.USER_CATS}
    
    for node in node_list:
        base_cat = node.get('user_cat', node.get('opinion', 'Center'))
        if base_cat in base_groups:
            base_groups[base_cat].append(node)
            
    for base_cat, group_nodes in base_groups.items():
        if not group_nodes: continue
        
        if 'Left' in base_cat: split_dict = config.SIDE_TOPIC_SPLIT_LEFT_BASE
        elif 'Right' in base_cat: split_dict = config.SIDE_TOPIC_SPLIT_RIGHT_BASE
        else: split_dict = {cat: 0.2 for cat in config.USER_CATS}
            
        n_group = len(group_nodes)
        
        if config.SIDE_TOPIC_ASSIGNMENT_METHOD == 'largest_remainder':
            counts, remainders = {}, {}
            for side_cat, prob in split_dict.items():
                exact = n_group * prob
                counts[side_cat] = int(math.floor(exact))
                remainders[side_cat] = exact - counts[side_cat]
            
            shortfall = n_group - sum(counts.values())
            sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
            for i in range(shortfall):
                counts[sorted_remainders[i][0]] += 1
                
            assignment_pool = []
            for side_cat, count in counts.items():
                assignment_pool.extend([side_cat] * count)
            random.shuffle(assignment_pool)
            
            for i, node in enumerate(group_nodes):
                node['opinions'] = {config.SIM_TOPICS[0]: base_cat, config.SIM_TOPICS[1]: assignment_pool[i]}
                node['user_cat'] = base_cat 
        else:
            cats, probs = list(split_dict.keys()), list(split_dict.values())
            for node in group_nodes:
                node['opinions'] = {config.SIM_TOPICS[0]: base_cat, config.SIM_TOPICS[1]: np.random.choice(cats, p=probs)}
                node['user_cat'] = base_cat
                
    return network_baseline

def run_simulation_batch():
    print(f"=== STARTING STOCHASTIC SIMULATIONS: {config.SIMULATION_STUDY_NAME} ===")
    
    for priority_flag in config.LAST_ROUND_PRIORITY:
        priority_str = "TRUE" if priority_flag else "FALSE"
        
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            print(f"\n=== Simulating Regime: [{regime_name}] | Priority: {priority_str} ===")
            
            for k in config.K_VALUES:
                sweep_dir = config.get_sim_sweep_dir(priority_flag, regime_name, k)
                sweep_dir.mkdir(parents=True, exist_ok=True)
                sweep_aggregator = {}
                
                for n in config.N_VALUES:
                    print(f"\n-> Running N={n}, K={k} under {regime_name} (Priority={priority_str})...")
                    baseline_path = config.get_baseline_path(n, k)
                    if not baseline_path.exists(): continue
                        
                    with open(baseline_path, 'r') as f: 
                        baseline_data = json.load(f)
                    global_pool = baseline_data["global_item_pool"]
                    
                    seg_network = assign_dual_topics(baseline_data["segregated_baseline"])
                    int_network = assign_dual_topics(baseline_data["integrated_baseline"])
                    
                    seg_results = core_engine.run_batch(seg_network, global_pool, regime_rules, config.NUM_TRIALS, 'segregated', priority_flag)
                    int_results = core_engine.run_batch(int_network, global_pool, regime_rules, config.NUM_TRIALS, 'integrated', priority_flag)
                    
                    results_store = {'segregated': seg_results, 'integrated': int_results}
                    sweep_aggregator[n] = results_store
                    
                    if n in config.DETAILED_SIM_N:
                        print("   Generating detailed visual suite & topic reports...")
                        n_dir = sweep_dir / f"N{n}"
                        n_plots = n_dir / "plots"
                        n_reports = n_dir / "reports"
                        n_plots.mkdir(parents=True, exist_ok=True)
                        n_reports.mkdir(parents=True, exist_ok=True)
                        
                        event_log_df = data_wrangle.build_simulation_event_log(results_store)
                        
                        plotting_engine.generate_suite(event_log_df, n_plots, prefix="", topics=config.SIM_TOPICS)
                        plotting_engine.generate_share_distributions(event_log_df, n_plots, prefix="", topics=config.SIM_TOPICS)

                        for topic in config.SIM_TOPICS:
                            plotting_engine.plot_feed_slant_variance(event_log_df, n_plots, topic, prefix="")

                        reporting_engine.export_comprehensive_summary(
                            event_log_df, n_reports, is_empirical=False, topics=config.SIM_TOPICS
                        )
                        
                if sweep_aggregator:
                    agg_file = sweep_dir / 'sweep_aggregator.json'
                    with open(agg_file, 'w') as f: 
                        json.dump(sweep_aggregator, f)
                    reporting_engine.export_sweep_summary(sweep_aggregator, sweep_dir, k, regime_name, priority_str)

    print(f"\n=== BATCH SIMULATION COMPLETE. Saved in: {config.SIMULATION_RESULTS_DIR} ===")

if __name__ == "__main__":
    run_simulation_batch()