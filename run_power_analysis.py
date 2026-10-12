import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu, brunnermunzel
import warnings

import config
from src import core_engine, stats_suite

warnings.simplefilter('ignore')

def plot_monte_carlo_heatmap(data_dict, n_vals, t_vals, title, out_path):
    matrix = np.zeros((len(n_vals), len(t_vals)))
    for i, n in enumerate(n_vals):
        for j, t in enumerate(t_vals):
            matrix[i, j] = data_dict.get((n, t), np.nan)
            
    df = pd.DataFrame(matrix, index=n_vals, columns=t_vals)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = mcolors.ListedColormap(['#de2d26', '#fcae91', '#9ecae1', '#08519c'])
    bounds = [0.0, 50.0, 80.0, 95.0, 100.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    sns.heatmap(df, cmap=cmap, norm=norm, annot=True, fmt=".1f", linewidths=.5, 
                cbar_kws={'label': 'Statistical Power (% Success)'}, ax=ax, 
                annot_kws={"size": 9, "weight": "bold"})
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Trials (T)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Nominal Network Size (N) per Condition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def generate_cost_efficiency_summary(master_power, out_dir, t_sweep_values, topic_name):
    out_path = out_dir / "Cost_Efficiency_Summary.txt"
    with open(out_path, 'w') as f:
        f.write(f"=== MONTE CARLO COST EFFICIENCY SUMMARY: {topic_name.upper()} ===\n")
        f.write(f"Pool Size: {config.MC_POOL_SIZE} | Iterations: {config.MC_ITERATIONS}\n")
        f.write(f"Analysis Window Size: Final {config.ANALYSIS_WINDOW_SIZE} Round(s)\n")
        for metric in master_power.keys():
            f.write(f"\n{'#'*80}\n# METRIC: {metric.upper()}\n{'#'*80}\n")
            for chamber in master_power[metric].keys():
                f.write(f"\nTARGET STRUCTURAL GROUP: {chamber.upper()}\n" + "="*50 + "\n")
                
                for method in ['MWU', 'Brunner-Munzel']:
                    f.write(f"\n  --- Test Method: {method} ---\n")
                    f.write(f"  {'Trials (T)':<10} | {'Nodes (N)':<10} | {'Statistical Power (%)'}\n")
                    f.write(f"  {'-'*10}-+-{'-'*10}-+-{'-'*25}\n")
                    
                    pow_data = master_power[metric][chamber][method]
                    for n in config.N_VALUES:
                        for t in t_sweep_values:
                            power_val = pow_data.get((n, t), 0.0)
                            f.write(f"  {t:<10} | {n:<10} | {power_val:>24.1f}%\n")

def analyze_condition(priority_flag, regime_name, regime_rules, k):
    priority_str = "TRUE" if priority_flag else "FALSE"
    print(f"\n=======================================================")
    print(f"--- DUAL POWER ANALYSIS: Priority={priority_str} | Regime={regime_name} | K={k} ---")
    print(f"--- Evaluation Window: Final {config.ANALYSIS_WINDOW_SIZE} round(s) ---")
    
    base_out_dir = config.get_power_sweep_dir(priority_flag, regime_name, k)
    
    metrics = ['Target_GT_2', 'Opposite_EQ_0', 'Congenial_Count', 'Contrary_Count', 'Feed_Slant', 'Feed_Variance']
    methods = ['MWU', 'Brunner-Munzel']
    chambers = ['Left Chamber', 'Right Chamber']
    t_sweep_values = list(range(2, 11))

    master_power = {
        topic: {m: {chamber: {method: {} for method in methods} for chamber in chambers} for m in metrics}
        for topic in config.SIM_TOPICS
    }

    for n in config.N_VALUES:
        print(f"\n-> Simulating Base Pool for N={n} ({config.MC_POOL_SIZE} dual-topic trials)...")
        baseline_path = config.get_baseline_path(n, k) 
        if not baseline_path.exists(): 
            print(f"   [Skipping] Baseline missing: {baseline_path}")
            continue
            
        with open(baseline_path, 'r') as f: 
            base = json.load(f)

        int_res = core_engine.run_batch(base['integrated_baseline'], base['global_item_pool'], regime_rules, config.MC_POOL_SIZE, 'integrated', priority_flag)
        seg_res = core_engine.run_batch(base['segregated_baseline'], base['global_item_pool'], regime_rules, config.MC_POOL_SIZE, 'segregated', priority_flag)

        for topic in config.SIM_TOPICS:
            for metric in metrics:
                for chamber in chambers:
                    int_means, seg_means = [], []
                    
                    for t_idx in range(config.MC_POOL_SIZE):
                        i_trial = int_res['trials'][t_idx]
                        s_trial = seg_res['trials'][t_idx]
                        
                        i_out = stats_suite.get_structural_outcomes(
                            i_trial['seen_events'], n, chamber, metric, 
                            topic_filter=topic, window_size=config.ANALYSIS_WINDOW_SIZE
                        )
                        s_out = stats_suite.get_structural_outcomes(
                            s_trial['seen_events'], n, chamber, metric, 
                            topic_filter=topic, window_size=config.ANALYSIS_WINDOW_SIZE
                        )
                        
                        int_means.append(np.nanmean(list(i_out.values())) if i_out else np.nan)
                        seg_means.append(np.nanmean(list(s_out.values())) if s_out else np.nan)

                    for t in t_sweep_values:
                        mc_pvals_mwu, mc_pvals_bm = [], []
                        for _ in range(config.MC_ITERATIONS):
                            draws = np.random.choice(config.MC_POOL_SIZE, size=t, replace=False)
                            i_macro = [int_means[d] for d in draws if not np.isnan(int_means[d])]
                            s_macro = [seg_means[d] for d in draws if not np.isnan(seg_means[d])]
                            
                            if len(i_macro) > 1 and len(s_macro) > 1 and len(set(i_macro).union(set(s_macro))) > 1:
                                try:
                                    _, p_m = mannwhitneyu(i_macro, s_macro, alternative='two-sided')
                                except: p_m = np.nan
                                mc_pvals_mwu.append(p_m)
                                
                                try:
                                    if np.var(i_macro) == 0 and np.var(s_macro) == 0:
                                        # FIX: Prevents false positive significance on exactly identical trial outcomes
                                        p_b = np.nan 
                                    else:
                                        _, p_b = brunnermunzel(i_macro, s_macro, alternative='two-sided')
                                except: p_b = np.nan
                                mc_pvals_bm.append(p_b)

                        p_arr_mwu = np.array([x for x in mc_pvals_mwu if not np.isnan(x)])
                        p_arr_bm = np.array([x for x in mc_pvals_bm if not np.isnan(x)])
                        
                        master_power[topic][metric][chamber]['MWU'][(n, t)] = np.mean(p_arr_mwu < 0.05) * 100.0 if len(p_arr_mwu) > 0 else 0.0
                        master_power[topic][metric][chamber]['Brunner-Munzel'][(n, t)] = np.mean(p_arr_bm < 0.05) * 100.0 if len(p_arr_bm) > 0 else 0.0

    for topic in config.SIM_TOPICS:
        topic_dir = base_out_dir / topic
        heatmaps_dir = topic_dir / "heatmaps"
        topic_dir.mkdir(parents=True, exist_ok=True)
        heatmaps_dir.mkdir(parents=True, exist_ok=True)
        
        generate_cost_efficiency_summary(master_power[topic], topic_dir, t_sweep_values, topic)
        
        for metric in metrics:
            metric_dir = heatmaps_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            
            for chamber in chambers:
                safe_chamber = chamber.replace(' ', '_')
                for method in methods:
                    pow_data = master_power[topic][metric][chamber][method]
                    method_str = method.replace(' ', '').replace('-', '')
                    plot_monte_carlo_heatmap(
                        pow_data, config.N_VALUES, t_sweep_values, 
                        f"Power (%) - {method}\n{metric} - {chamber} ({topic.title()}, K={k})", 
                        metric_dir / f"Heatmap_{safe_chamber}_{method_str}.pdf"
                    )

def execute_power_sweep():
    print("=== EXECUTING MONTE CARLO POWER SWEEPS ===")
    print(f"Base Pool Size: {config.MC_POOL_SIZE} | Iterations: {config.MC_ITERATIONS}")
    print(f"Window: Final {config.ANALYSIS_WINDOW_SIZE} Round(s)\n")
    
    for priority_flag in config.LAST_ROUND_PRIORITY:
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            for k in config.K_VALUES:
                analyze_condition(priority_flag, regime_name, regime_rules, k)
                
    print(f"\n=== POWER SWEEPS COMPLETE. Saved in: {config.POWER_RESULTS_DIR} ===")

if __name__ == "__main__":
    execute_power_sweep()