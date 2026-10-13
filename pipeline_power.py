import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu

import config
from src.simulation import core_engine
from src.analysis import metrics

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
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Trials (T)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Nominal Network Size (N) per Condition", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def generate_cost_efficiency_summary(master_power, out_dir, t_sweep_values, topic_name):
    out_path = out_dir / "Cost_Efficiency_Summary.txt"
    with open(out_path, 'w') as f:
        f.write(f"=== MONTE CARLO COST EFFICIENCY SUMMARY: {topic_name.upper()} ===\n")
        f.write(f"Pool Size: {config.MC_POOL_SIZE} | Iterations: {config.MC_ITERATIONS}\n")
        f.write(f"Analysis Window Size: Final {config.ANALYSIS_WINDOW_SIZE} Round(s)\n")
        
        # Log the noise settings applied to this run
        f.write(f"Noise - Composition  : Active (Multinomial Resampling per Trial)\n")
        lurker_pct = getattr(config, 'SIM_LURKER_FRACTION', 0.0) * 100
        f.write(f"Noise - Lurkers      : {lurker_pct:.1f}%\n")
        if getattr(config, 'USE_ALEATORIC_NOISE', False):
            err_pct = getattr(config, 'ALEATORIC_ERROR_RATE', 0.0) * 100
            f.write(f"Noise - Trembling    : Active (Error Rate: {err_pct:.1f}%)\n")
        else:
            f.write(f"Noise - Trembling    : Inactive\n")
        
        for mc_pair in master_power.keys():
            metric, chamber = mc_pair
            f.write(f"\n{'#'*80}\n# METRIC: {metric.upper()} | SCOPE: {chamber.upper()}\n{'#'*80}\n")
            f.write("  --- Test Method: Mann-Whitney U ---\n")
            f.write(f"  {'Trials (T)':<10} | {'Nodes (N)':<10} | {'Statistical Power (%)'}\n")
            f.write(f"  {'-'*10}-+-{'-'*10}-+-{'-'*25}\n")
            
            pow_data = master_power[mc_pair]
            for n in config.N_VALUES:
                for t in t_sweep_values:
                    power_val = pow_data.get((n, t), 0.0)
                    f.write(f"  {t:<10} | {n:<10} | {power_val:>24.1f}%\n")

def _run_monte_carlo_tests(int_means, seg_means, t_sweep_values):
    power_results = {}
    for t in t_sweep_values:
        mc_pvals_mwu = []
        for _ in range(config.MC_ITERATIONS):
            draws = np.random.choice(config.MC_POOL_SIZE, size=t, replace=False)
            i_macro = [int_means[d] for d in draws if not np.isnan(int_means[d])]
            s_macro = [seg_means[d] for d in draws if not np.isnan(seg_means[d])]
            
            if len(i_macro) > 1 and len(s_macro) > 1 and len(set(i_macro).union(set(s_macro))) > 1:
                try:
                    _, p_m = mannwhitneyu(i_macro, s_macro, alternative='two-sided')
                    mc_pvals_mwu.append(p_m)
                except (ValueError, TypeError):
                    pass
                    
        p_arr_mwu = np.array(mc_pvals_mwu)
        power_results[t] = np.mean(p_arr_mwu < 0.05) * 100.0 if len(p_arr_mwu) > 0 else 0.0
    return power_results

def analyze_condition(priority_flag, regime_name, regime_rules, k):
    priority_str = "TRUE" if priority_flag else "FALSE"
    print(f"\n=======================================================")
    print(f"--- RUNNING POWER ANALYSIS (WITH FULL NOISE SUITE) ---")
    print(f"Priority={priority_str} | Regime={regime_name} | K={k}")
    print(f"=======================================================")
    
    base_out_dir = config.get_power_sweep_dir(priority_flag, regime_name, k)
    t_sweep_values = list(range(2, 11))

    metric_configs = [
        ('Feed_Slant', 'Left Chamber'),
        ('Feed_Slant', 'Right Chamber'),
        ('Feed_Variance', 'Left Chamber'),
        ('Feed_Variance', 'Right Chamber'),
        ('Polarization_Gap', 'Network-Wide')
    ]

    master_power = {topic: {mc_pair: {} for mc_pair in metric_configs} for topic in config.SIM_TOPICS}

    for n in config.N_VALUES:
        print(f"\n-> Simulating Resampled Trials for N={n} ({config.MC_POOL_SIZE} trials per condition)...")
        baseline_path = config.get_baseline_path(n, k)
        if not baseline_path.exists():
            print(f"   [!] Baseline map not found at {baseline_path}, skipping N={n}")
            continue
            
        with open(baseline_path, 'r') as f:
            base = json.load(f)

        int_res = core_engine.run_batch(
            base['integrated_baseline'], base['global_item_pool'], 
            regime_rules, config.MC_POOL_SIZE, 'integrated', priority_flag
        )
        seg_res = core_engine.run_batch(
            base['segregated_baseline'], base['global_item_pool'], 
            regime_rules, config.MC_POOL_SIZE, 'segregated', priority_flag
        )

        for topic in config.SIM_TOPICS:
            for mc_pair in metric_configs:
                metric, chamber = mc_pair
                int_means, seg_means = [], []
                
                for t_idx in range(config.MC_POOL_SIZE):
                    if metric == 'Polarization_Gap':
                        i_gap = metrics.get_polarization_gap(
                            int_res['trials'][t_idx]['seen_events'], n, topic, 
                            config.ANALYSIS_WINDOW_SIZE, int_res['trials'][t_idx]['node_opinions']
                        )
                        s_gap = metrics.get_polarization_gap(
                            seg_res['trials'][t_idx]['seen_events'], n, topic, 
                            config.ANALYSIS_WINDOW_SIZE, seg_res['trials'][t_idx]['node_opinions']
                        )
                        int_means.append(i_gap)
                        seg_means.append(s_gap)
                    else:
                        i_out = metrics.get_structural_outcomes(
                            int_res['trials'][t_idx]['seen_events'], n, chamber, metric, 
                            topic, config.ANALYSIS_WINDOW_SIZE
                        )
                        s_out = metrics.get_structural_outcomes(
                            seg_res['trials'][t_idx]['seen_events'], n, chamber, metric, 
                            topic, config.ANALYSIS_WINDOW_SIZE
                        )
                        int_means.append(np.nanmean(list(i_out.values())) if i_out else np.nan)
                        seg_means.append(np.nanmean(list(s_out.values())) if s_out else np.nan)

                power_res = _run_monte_carlo_tests(int_means, seg_means, t_sweep_values)
                for t, power_val in power_res.items():
                    master_power[topic][mc_pair][(n, t)] = power_val

    for topic in config.SIM_TOPICS:
        topic_dir = base_out_dir / topic
        heatmaps_dir = topic_dir / "heatmaps"
        topic_dir.mkdir(parents=True, exist_ok=True)
        heatmaps_dir.mkdir(parents=True, exist_ok=True)
        
        generate_cost_efficiency_summary(master_power[topic], topic_dir, t_sweep_values, topic)
        
        for mc_pair in metric_configs:
            metric, chamber = mc_pair
            metric_dir = heatmaps_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            
            safe_chamber = chamber.replace(' ', '_').replace('-', '_')
            
            # Formatted the title cleanly without line breaks breaking the code
            plot_title = f"Power (%) - MWU\n{metric} - {chamber} ({topic.title()}, K={k})\n[Full Noise Suite Active]"
            
            plot_monte_carlo_heatmap(
                master_power[topic][mc_pair], config.N_VALUES, t_sweep_values, 
                plot_title, 
                metric_dir / f"Heatmap_{safe_chamber}_MWU.pdf"
            )

def execute_power_sweep():
    print("=== EXECUTING MONTE CARLO POWER SWEEPS WITH NOISE ===")
    for priority_flag in config.LAST_ROUND_PRIORITY:
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            for k in config.K_VALUES:
                analyze_condition(priority_flag, regime_name, regime_rules, k)

if __name__ == "__main__":
    execute_power_sweep()