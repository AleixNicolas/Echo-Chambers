import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from collections import defaultdict
import warnings

import config
from src import core_engine, stats_suite

warnings.simplefilter('ignore')

MC_POOL_SIZE = 300
MC_ITERATIONS = 5000
STANDARD_TRIALS = 10
ANALYSIS_WINDOW_SIZE = 1

def run_standard_mwu(int_means, seg_means):
    """Runs a single MWU test comparing independent trial-level averages."""
    if len(set(int_means).union(set(seg_means))) > 1 and len(int_means) > 0 and len(seg_means) > 0:
        _, p = mannwhitneyu(int_means, seg_means, alternative='two-sided')
        return {'Macro (Network Level)': {'p_mean': p, 'p_std': 0.0, 'n_val': len(int_means), 'n_unit': 'Independent Trials/Trt'}}
    return {'Macro (Network Level)': {'p_mean': np.nan, 'p_std': 0.0, 'n_val': len(int_means), 'n_unit': 'Independent Trials/Trt'}}

def plot_power_curves(n_values, plot_data, title, out_path):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    c_idx = 0
    for method, metrics in plot_data.items():
        p_means = [m['p_mean'] for m in metrics]
        p_stds = [m['p_std'] for m in metrics]
        n_vals_eff = [m['n_val'] for m in metrics]
        n_units = [m['n_unit'] for m in metrics]
        
        valid_idx = [i for i, p in enumerate(p_means) if not np.isnan(p)]
        if not valid_idx: continue
        
        valid_n = [n_values[i] for i in valid_idx]
        valid_p = np.array([p_means[i] for i in valid_idx])
        valid_s = np.array([p_stds[i] for i in valid_idx])
        
        color = colors[c_idx % 10]
        c_idx += 1
        
        unique_units = list(set([n_units[i] for i in valid_idx]))
        display_unit = unique_units[0] if unique_units else ""
        label = f"{method} (n = {int(np.mean([n_vals_eff[i] for i in valid_idx]))} {display_unit.split()[0]})"
        
        ax1.plot(valid_n, valid_p, color=color, marker='o', linewidth=2, label=label)
        ax1.fill_between(valid_n, np.clip(valid_p - valid_s, 1e-20, 1.0), np.clip(valid_p + valid_s, 1e-20, 1.0), color=color, alpha=0.15)
        
    ax1.axhline(y=0.05, color='r', linestyle='-', linewidth=2, label='Significance (0.05)')
    ax1.set_yscale('log')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Nominal Network Size (N)', fontsize=12)
    ax1.set_ylabel('Mean p-value (Log Scale)', fontsize=12)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_monte_carlo_heatmap(data_dict, n_vals, t_vals, title, out_path):
    matrix = np.zeros((len(n_vals), len(t_vals)))
    for i, n in enumerate(n_vals):
        for j, t in enumerate(t_vals):
            matrix[i, j] = data_dict.get((n, t), np.nan)
            
    df = pd.DataFrame(matrix, index=n_vals, columns=t_vals)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = mcolors.ListedColormap(['#de2d26', '#fcae91', '#9ecae1', '#08519c'])
    bounds = [0.0, 50.0, 80.0, 95.0, 100.0]
    cbar_label = 'Statistical Power (% Success)'
    fmt_str = ".1f"
        
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sns.heatmap(df, cmap=cmap, norm=norm, annot=True, fmt=fmt_str, linewidths=.5, cbar_kws={'label': cbar_label}, ax=ax, annot_kws={"size": 9, "weight": "bold"})
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Trials (T)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Nominal Network Size (N) per Condition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def generate_text_summary(master_results, out_dir, topic_name):
    out_path = os.path.join(out_dir, f"Standard_Power_Summary_{topic_name}.txt")
    with open(out_path, 'w') as f:
        f.write(f"=== STATISTICAL POWER ANALYSIS SUMMARY: {topic_name.upper()} ===\n")
        f.write(f"Note: Standard trials output based on first {STANDARD_TRIALS} trials of the Monte Carlo pool.\n")
        for metric in master_results.keys():
            f.write(f"\n{'#'*80}\n# METRIC: {metric.upper()}\n{'#'*80}\n")
            for cat, n_data in master_results[metric].items():
                f.write(f"\nTARGET CATEGORY: {cat.upper()}\n" + "="*50 + "\n")
                for n in sorted(n_data.keys()):
                    stats = n_data[n]
                    if not stats: continue
                    f.write(f"\n  Nominal Network Size (N) = {n}\n")
                    f.write(f"  {'Method':<35} | {'p-value (Mean)':<15} | {'Std Dev':<10} | {'Effective Sample Size'}\n")
                    f.write(f"  {'-'*35}-+-{'-'*15}-+-{'-'*10}-+-{'-'*25}\n")
                    for method, m in stats.items():
                        p_val = m['p_mean']
                        p_str = f"{p_val:.4e}" if not np.isnan(p_val) else "N/A"
                        sig = " *" if not np.isnan(p_val) and p_val < 0.05 else ""
                        std_str = f"{m['p_std']:.4e}" if m['p_std'] > 0.0 else "0.0000"
                        f.write(f"  {method:<35} | {p_str+sig:<15} | {std_str:<10} | {m['n_val']} {m['n_unit']}\n")

def generate_cost_efficiency_summary(master_power, out_dir, t_sweep_values, topic_name):
    out_path = os.path.join(out_dir, f"Cost_Efficiency_Summary_{topic_name}.txt")
    with open(out_path, 'w') as f:
        f.write(f"=== MONTE CARLO COST EFFICIENCY SUMMARY: {topic_name.upper()} ===\n")
        f.write(f"Pool Size: {MC_POOL_SIZE} | Iterations: {MC_ITERATIONS}\n")
        for metric in master_power.keys():
            f.write(f"\n{'#'*80}\n# METRIC: {metric.upper()}\n{'#'*80}\n")
            for cat in master_power[metric].keys():
                f.write(f"\nTARGET CATEGORY: {cat.upper()}\n" + "="*50 + "\n")
                f.write(f"  {'Trials (T)':<10} | {'Nodes (N)':<10} | {'Statistical Power (%)'}\n")
                f.write(f"  {'-'*10}-+-{'-'*10}-+-{'-'*25}\n")
                
                model = 'Macro (Network Level)'
                pow_data = master_power[metric][cat][model]
                
                for n in config.N_VALUES:
                    for t in t_sweep_values:
                        power_val = pow_data.get((n, t), 0.0)
                        f.write(f"  {t:<10} | {n:<10} | {power_val:>24.1f}%\n")

def analyze_condition(priority_flag, regime_name, regime_rules, k):
    priority_str = "TRUE" if priority_flag else "FALSE"
    print(f"\n=======================================================")
    print(f"--- DUAL POWER ANALYSIS: Priority={priority_str} | Regime={regime_name} | K={k} ---")
    
    base_out_dir = os.path.join(
        config.RESULTS_DIR, 
        f"Power_Analysis_Priority_{priority_str}", 
        f"Regime_{regime_name}", 
        f"Sweep_K{k}"
    )
    
    metrics = ['Target_GT_2', 'Opposite_EQ_0', 'Congenial_Count', 'Contrary_Count']
    user_cats = ['Left Users', 'Right Users']
    t_sweep_values = list(range(2, 11))
    model = 'Macro (Network Level)'

    # Pre-structure storage per topic
    master_results = {
        topic: {m: {cat: {n: {} for n in config.N_VALUES} for cat in user_cats} for m in metrics}
        for topic in config.SIM_TOPICS
    }
    master_power = {
        topic: {m: {cat: {model: {}} for cat in user_cats} for m in metrics}
        for topic in config.SIM_TOPICS
    }

    for n in config.N_VALUES:
        print(f"\n-> Generating Base Pool for N={n} (Simulating {MC_POOL_SIZE} dual-topic trials)...")
        baseline_path = config.get_baseline_path(n, k) 
        if not os.path.exists(baseline_path): 
            print(f"   [Skipping] Baseline file missing: {baseline_path}")
            continue
            
        with open(baseline_path, 'r') as f: base = json.load(f)

        int_res = core_engine.run_batch(base['integrated_baseline'], base['global_item_pool'], regime_rules, MC_POOL_SIZE, priority_flag)
        seg_res = core_engine.run_batch(base['segregated_baseline'], base['global_item_pool'], regime_rules, MC_POOL_SIZE, priority_flag)

        # Analyze each topic independently
        for topic in config.SIM_TOPICS:
            for metric in metrics:
                for cat in user_cats:
                    int_means, seg_means = [], []
                    
                    for t_idx in range(MC_POOL_SIZE):
                        i_trial = int_res['trials'][t_idx]
                        s_trial = seg_res['trials'][t_idx]
                        
                        i_ops = {node: info[topic] for node, info in i_trial['node_opinions'].items()}
                        s_ops = {node: info[topic] for node, info in s_trial['node_opinions'].items()}
                        
                        i_out = stats_suite.extract_node_outcomes(
                            i_trial['seen_events'], i_ops, cat, metric, ANALYSIS_WINDOW_SIZE, topic_filter=topic
                        )
                        s_out = stats_suite.extract_node_outcomes(
                            s_trial['seen_events'], s_ops, cat, metric, ANALYSIS_WINDOW_SIZE, topic_filter=topic
                        )
                        
                        int_means.append(np.mean(list(i_out.values())) if i_out else 0.0)
                        seg_means.append(np.mean(list(s_out.values())) if s_out else 0.0)

                    master_results[topic][metric][cat][n] = run_standard_mwu(int_means[:STANDARD_TRIALS], seg_means[:STANDARD_TRIALS])

                    for t in t_sweep_values:
                        mc_pvals = []
                        for _ in range(MC_ITERATIONS):
                            draws = np.random.choice(MC_POOL_SIZE, size=t, replace=False)
                            i_macro = [int_means[d] for d in draws]
                            s_macro = [seg_means[d] for d in draws]
                            
                            if len(set(i_macro).union(set(s_macro))) > 1:
                                _, p = mannwhitneyu(i_macro, s_macro, alternative='two-sided')
                                mc_pvals.append(p)

                        p_array = np.array(mc_pvals)
                        master_power[topic][metric][cat][model][(n, t)] = np.mean(p_array < 0.05) * 100.0 if len(p_array) > 0 else 0.0

    # Output separate reports and plots for each topic
    for topic in config.SIM_TOPICS:
        print(f"-> Generating Output Files for {topic.upper()}...")
        topic_folder = "Base_Topic" if topic == 'base_topic' else "Side_Topic"
        out_dir = os.path.join(base_out_dir, topic_folder)
        std_dir = os.path.join(out_dir, "1_Standard_Power_Curves")
        cost_dir = os.path.join(out_dir, "2_Cost_Efficiency_Sweeps")
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(cost_dir, exist_ok=True)
        
        generate_text_summary(master_results[topic], std_dir, topic)
        generate_cost_efficiency_summary(master_power[topic], cost_dir, t_sweep_values, topic)
        
        for metric in metrics:
            metric_std_dir = os.path.join(std_dir, metric)
            metric_cost_dir = os.path.join(cost_dir, metric)
            os.makedirs(metric_std_dir, exist_ok=True)
            os.makedirs(metric_cost_dir, exist_ok=True)
            
            for cat in user_cats:
                safe_cat = cat.replace(' ', '_')
                
                plot_data = defaultdict(list)
                for n in config.N_VALUES:
                    for method, m_data in master_results[topic][metric][cat].get(n, {}).items():
                        plot_data[method].append(m_data)
                        
                if plot_data: 
                    plot_power_curves(
                        config.N_VALUES, 
                        plot_data, 
                        f"{metric}: {cat} ({topic.title()}, K={k}, {regime_name})", 
                        os.path.join(metric_std_dir, f"PC_Aggregated_{safe_cat}.pdf")
                    )

                pow_data = master_power[topic][metric][cat][model]
                plot_monte_carlo_heatmap(
                    pow_data, 
                    config.N_VALUES, 
                    t_sweep_values, 
                    f"Statistical Power (%)\n{metric} - {cat} ({topic.title()}, K={k})", 
                    os.path.join(metric_cost_dir, f"Heatmap_Power_{safe_cat}.pdf")
                )

def execute_power_sweep():
    print(f"=== EXECUTING MONTE CARLO POWER SWEEP ===")
    print(f"Base Pool Size: {MC_POOL_SIZE} Trials | Monte Carlo Draws: {MC_ITERATIONS} Iterations\n")
    
    for priority_flag in config.LAST_ROUND_PRIORITY:
        for regime_name, regime_rules in config.FILTER_REGIMES.items():
            for k in config.K_VALUES:
                analyze_condition(priority_flag, regime_name, regime_rules, k)
                
    print(f"\n=== COMPLETE. All sweeps across Priority, Regimes, and Topics saved. ===")

if __name__ == "__main__":
    execute_power_sweep()