import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import config
from src.analysis import metrics

def plot_feed_slant_variance(event_log_df, output_dir, topic, prefix=""):
    df = event_log_df[(event_log_df['topic'] == topic) & (event_log_df['action'] == 'seen')].copy()
    if df.empty: return

    mapping = {'Left': -2, 'Center-Left': -1, 'Center': 0, 'Center-Right': 1, 'Right': 2}
    df['slant'] = df['item_cat'].map(mapping)
    df = df.dropna(subset=['slant'])
    
    def determine_group(row):
        if row['treatment'] == 'segregated':
            return 'Segregated (Left Chamber)' if row['chamber'] == 'Left' else 'Segregated (Right Chamber)'
        else:
            return 'Integrated (Left-Leaning)' if row['user_cat'] in ['Left', 'Center-Left'] else 'Integrated (Right-Leaning)'
            
    df['group'] = df.apply(determine_group, axis=1)
    
    # Aggregate to User-level Mean Feed Slant per round
    user_slant_df = df.groupby(['trial_id', 'user_id', 'round', 'group'])['slant'].mean().reset_index()

    expected_groups = ['Segregated (Left Chamber)', 'Segregated (Right Chamber)', 
                       'Integrated (Left-Leaning)', 'Integrated (Right-Leaning)']
    present_groups = [g for g in expected_groups if g in user_slant_df['group'].values]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = {
        'Segregated (Left Chamber)': '#2980b9',
        'Segregated (Right Chamber)': '#c0392b',
        'Integrated (Left-Leaning)': '#7fb3d5',
        'Integrated (Right-Leaning)': '#d98880'
    }
    
    # 1. Background Strip Plot (Shows the raw participant distribution)
    sns.stripplot(data=user_slant_df, x='round', y='slant', hue='group', hue_order=present_groups,
                  palette=palette, dodge=True, alpha=0.35, size=5, jitter=True, ax=ax, legend=False)
    
    # 2. Foreground Point Plot (Shows Mean and Standard Deviation)
    sns.pointplot(data=user_slant_df, x='round', y='slant', hue='group', hue_order=present_groups,
                  palette=palette, dodge=0.8, errorbar='sd', markers="D", capsize=0.1, 
                  join=False, linestyle='none', ax=ax)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=2)
    
    plt.xlabel('Simulation Round', fontsize=12)
    plt.ylabel('Mean Feed Slant per Participant (-2 to +2)', fontsize=12)
    
    rounds = sorted(user_slant_df['round'].unique())
    plt.xticks(np.arange(len(rounds)), [f"Round {int(r)}" for r in rounds], fontsize=11)
    plt.ylim(-2.5, 2.5)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.95, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    
    safe_prefix = f"{prefix}_" if prefix else ""
    plt.savefig(output_dir / f"Plot_00_{safe_prefix}Feed_Slant_Variance_{topic}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def generate_suite(event_log_df, output_dir, prefix="", topics=['climate']):
    if event_log_df.empty or 'treatment' not in event_log_df.columns: return
    output_dir = Path(output_dir)
    is_sim = "Sim" in prefix or prefix == ""
    
    for topic in topics:
        topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        topic_out_dir = output_dir / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        
        plot_chronological_feed_amp(topic_df, topic_out_dir, prefix)
        
        for treatment in topic_df['treatment'].unique():
            t_df = topic_df[topic_df['treatment'] == treatment]
            safe_prefix = f"{treatment}" if not prefix else f"{prefix}_{treatment}"
            
            plot_temporal_heatmaps(t_df, topic_out_dir, safe_prefix)
            plot_feed_sources(t_df, topic_out_dir, safe_prefix)
            plot_exposure_distributions(t_df, topic_out_dir, safe_prefix, is_sim)

def plot_chronological_feed_amp(df, output_dir, prefix=""):
    if df.empty: return
    df = df.copy()
    
    if 'chamber' not in df.columns:
        df['chamber'] = 'Unknown'

    def determine_group(row):
        if row['treatment'] == 'segregated':
            return 'S-L' if row['chamber'] == 'Left' else 'S-R'
        else:
            return 'I-L' if row['user_cat'] in ['Left', 'Center-Left'] else 'I-R'

    df['group'] = df.apply(determine_group, axis=1)
    cats = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
    colors = {'Left': '#2980b9', 'Center-Left': '#7fb3d5', 'Center': '#bdc3c7', 'Center-Right': '#d98880', 'Right': '#c0392b'}
    rounds = sorted(df['round'].unique())
    x_indices = np.arange(len(rounds))
    
    width = 0.17 
    
    expected_groups = ['S-L', 'S-R', 'I-L', 'I-R']
    present_groups = [g for g in expected_groups if g in df['group'].values]
    
    num_groups = len(present_groups)
    if num_groups == 4: offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    elif num_groups == 2: offsets = [-0.5 * width, 0.5 * width]
    elif num_groups == 1: offsets = [0]
    else: offsets = np.linspace(-width*(num_groups-1)/2, width*(num_groups-1)/2, num_groups)
    group_offsets = dict(zip(present_groups, offsets))
    
    for action, plot_name in [('seen', '01A_Incoming_Diet'), ('shared', '01B_Outgoing_Amp')]:
        act_df = df[df['action'] == action]
        if act_df.empty: continue
        
        data = {g: {c: [] for c in cats} for g in present_groups}
        for g in present_groups:
            for r in rounds:
                sub = act_df[(act_df['round'] == r) & (act_df['group'] == g)]
                tot = len(sub) if len(sub) > 0 else 1
                for c in cats: data[g][c].append(len(sub[sub['item_cat'] == c]) / tot)
        
        fig, ax = plt.subplots(figsize=(16, 7))
        bottoms = {g: np.zeros(len(rounds)) for g in present_groups}
        
        for c in cats:
            for g in present_groups:
                # Hatching strictly applied to Right-leaning groups to allow ideological comparison
                hatch_pattern = '////' if g.endswith('R') else ''
                
                ax.bar(x_indices + group_offsets[g], data[g][c], width, bottom=bottoms[g], 
                       color=colors[c], edgecolor='black', linewidth=1.0, hatch=hatch_pattern)
                bottoms[g] += np.array(data[g][c])
            
        ax.set_ylabel("Proportion of Volume", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_indices)
        
        ax.tick_params(axis='x', pad=35)
        ax.set_xticklabels([f"Round {int(r)}" for r in rounds], fontsize=12, fontweight='bold')
        
        legend_elements = [mpatches.Patch(color=colors[c], label=c) for c in cats]
        legend_elements.extend([
            mpatches.Patch(facecolor='white', edgecolor='black', label='Left Chamber/Users'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Right Chamber/Users')
        ])
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(legend_elements), fontsize=11)
        
        for i in x_indices:
            for g in present_groups:
                ax.text(i + group_offsets[g], -0.03, g, ha='center', va='top', fontsize=11, rotation=0, fontweight='bold')

        plt.subplots_adjust(bottom=0.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        filename = f"Plot_{plot_name}_Combined_{prefix}.pdf" if prefix else f"Plot_{plot_name}_Combined.pdf"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def generate_share_distributions(event_log_df, output_dir, prefix="", topics=['climate']):
    if event_log_df.empty or 'treatment' not in event_log_df.columns: return
    output_dir = Path(output_dir)
    is_sim = "Sim" in prefix or prefix == ""
    
    for topic in topics:
        topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        topic_out_dir = output_dir / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        
        for treatment in topic_df['treatment'].unique():
            t_df = topic_df[topic_df['treatment'] == treatment]
            safe_prefix = f"{treatment}" if not prefix else f"{prefix}_{treatment}"
            plot_share_distributions(t_df, topic_out_dir, safe_prefix, is_sim)

def plot_share_distributions(df, output_dir, prefix, is_sim=False):
    if df.empty: return
    users_df = df[['trial_id', 'user_id', 'user_cat']].drop_duplicates()
    shares_df = df[df['action'] == 'shared']
    share_counts = shares_df.groupby(['trial_id', 'user_id']).size().reset_index(name='total_shares')
    
    merged = pd.merge(users_df, share_counts, on=['trial_id', 'user_id'], how='left')
    merged['total_shares'] = merged['total_shares'].fillna(0).astype(int)
    
    def map_lean(cat):
        if cat in ['Left', 'Center-Left']: return 'Combined Left (L/CL)'
        if cat in ['Right', 'Center-Right']: return 'Combined Right (R/CR)'
        return 'Pure Center (C)'
        
    merged['Lean Group'] = merged['user_cat'].apply(map_lean)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {'Combined Left (L/CL)': '#2980b9', 'Pure Center (C)': '#bdc3c7', 'Combined Right (R/CR)': '#c0392b'}
    
    sns.countplot(data=merged, x='total_shares', hue='Lean Group', palette=palette, edgecolor='black', ax=ax)
    
    ax.set_xlabel("Total Items Shared")
    ax.set_ylabel("Participants")
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_06_Share_Dist_{prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_exposure_distributions(df, output_dir, prefix, is_sim=False):
    seen_df = df[df['action'] == 'seen'].copy()
    if seen_df.empty: return
    if not is_sim: seen_df = seen_df[seen_df['round'] == seen_df['round'].max()]
        
    base = seen_df[['trial_id', 'round', 'user_id', 'user_cat']].drop_duplicates()
    item_counts = seen_df.groupby(['trial_id', 'round', 'user_id', 'item_cat']).size().unstack(fill_value=0).reset_index()
    for ic in config.ITEM_CATS:
        if ic not in item_counts.columns: item_counts[ic] = 0
            
    merged = pd.merge(base, item_counts, on=['trial_id', 'round', 'user_id'], how='left').fillna(0)
    
    def map_lean(cat):
        if cat in ['Left', 'Center-Left']: return 'Left (L/CL)'
        if cat in ['Right', 'Center-Right']: return 'Right (R/CR)'
        return 'Center (C)'
        
    merged['Lean Group'] = merged['user_cat'].apply(map_lean)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    palette = {'Left (L/CL)': '#2980b9', 'Center (C)': '#bdc3c7', 'Right (R/CR)': '#c0392b'}
    
    for idx, ic in enumerate(config.ITEM_CATS):
        sns.countplot(data=merged, x=ic, hue='Lean Group', palette=palette, edgecolor='black', ax=axes[idx])
        axes[idx].set_xlabel("Number of Items Seen")
        axes[idx].set_ylabel("Participants")
        axes[idx].grid(True, axis='y', alpha=0.3)
        if idx > 0: axes[idx].get_legend().remove()

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_05_Exposure_Dist_{prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_feed_sources(df, output_dir, prefix):
    multi_metrics = metrics.calculate_sourcing_metrics(df)
    if not multi_metrics or 'Global' not in multi_metrics: return
    
    mets = multi_metrics['Global']
    rounds = mets['rounds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=rounds, y=mets['rep_avg'], marker='o', lw=2, color='#e74c3c', label='Average Repeats', ax=ax)
    sns.lineplot(x=rounds, y=mets['pool_avg'], marker='s', lw=2, color='#3498db', label='Average Pool', ax=ax)
    sns.lineplot(x=rounds, y=mets['back_avg'], marker='^', lw=2, color='#27ae60', label='Average Backlog', ax=ax)
    sns.lineplot(x=rounds, y=mets['hid_avg'], marker='d', lw=2, color='#8e44ad', label='Average Hidden', ax=ax)
    sns.lineplot(x=rounds, y=mets['cross_avg'], marker='v', lw=2, color='#f39c12', label='Average Cross-Chamber', ax=ax)
    
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel("Accumulated Items")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_04_Feed_Sources_{prefix}.pdf", dpi=300)
    plt.close(fig)
    
    report_lines = [f"=== FEED SOURCES REPORT: {prefix} ==="]
    for scope in ['Global', 'Left', 'Right']:
        if scope not in multi_metrics: continue
        scope_met = multi_metrics[scope]
        report_lines.append(f"\n--- {scope.upper()} CHAMBER / SCOPE ---")
        
        def fmt(avg, std, mx): return f"Avg: {avg:>5.2f} (±{std:>5.2f}) | Max: {mx:>5.2f}" if not np.isnan(avg) else "N/A"
        
        for i, r in enumerate(rounds):
            report_lines.append(f"\n  Round {r}:")
            report_lines.append(f"    Repeats       : {fmt(scope_met['rep_avg'][i], scope_met['rep_std'][i], scope_met['rep_max'][i])}")
            report_lines.append(f"    Pool          : {fmt(scope_met['pool_avg'][i], scope_met['pool_std'][i], scope_met['pool_max'][i])}")
            report_lines.append(f"    Backlog       : {fmt(scope_met['back_avg'][i], scope_met['back_std'][i], scope_met['back_max'][i])}")
            report_lines.append(f"    Hidden        : {fmt(scope_met['hid_avg'][i], scope_met['hid_std'][i], scope_met['hid_max'][i])}")
            report_lines.append(f"    Cross-Chamber : {fmt(scope_met['cross_avg'][i], scope_met['cross_std'][i], scope_met['cross_max'][i])}")
            
    with open(output_dir / f"Report_04_Feed_Sources_{prefix}.txt", 'w') as f: 
        f.write("\n".join(report_lines))

def plot_temporal_heatmaps(df, output_dir, prefix):
    cmap_base = plt.cm.Greens; cmap_base.set_bad(color='#d3d3d3')
    indiv_dir = output_dir / "individual_rounds"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    
    for name, mode in [("02_InFeed_Comp", 'row'), ("03_Share_Rate", 'element'), ("03B_Norm_Share_Volume", 'norm_vol')]:
        fig_main, axes_main = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        for idx, chamber in enumerate(['Left', 'Right']):
            ax_main = axes_main[idx]
            if 'chamber' not in df.columns: continue
            chamber_df = df[df['chamber'] == chamber]
            
            if not chamber_df.empty:
                mat_seen = metrics.build_matrix(chamber_df[chamber_df['action'] == 'seen'], None)
                mat_share = metrics.build_matrix(chamber_df[chamber_df['action'] == 'shared'], None)
                mat_trg = mat_share if 'Share' in name else mat_seen
                
                if mode == 'norm_vol':
                    row_sums = mat_seen.sum(axis=1, keepdims=True)
                    v_rate = np.divide(mat_trg, row_sums, out=np.zeros_like(mat_trg, dtype=float), where=row_sums!=0)
                else:
                    v_rate = metrics.calc_rate(mat_trg, mat_seen, mode)
                    
                im = ax_main.imshow(v_rate, origin='upper', cmap=cmap_base, vmin=0, vmax=1.0)
                ax_main.set_title(f"{chamber} Chamber", fontweight='bold')
                ax_main.set_xticks(range(5)); ax_main.set_yticks(range(5))
                ax_main.set_xticklabels([l.replace(' ', '\n') for l in config.LEANING_ORDER])
                if idx == 0: ax_main.set_yticklabels([config.POS_LABELS[p] for p in [5, 4, 3, 2, 1]])
                if idx == 0: ax_main.set_ylabel("User Baseline Opinion", fontsize=12)
            
            if not chamber_df.empty:
                for r_target in [None] + list(range(1, config.ROUNDS + 1)):
                    mat_s = metrics.build_matrix(chamber_df[chamber_df['action'] == 'seen'], r_target)
                    mat_sh = metrics.build_matrix(chamber_df[chamber_df['action'] == 'shared'], r_target)
                    mat_t = mat_sh if 'Share' in name else mat_s
                    if mode == 'norm_vol':
                        r_sums = mat_s.sum(axis=1, keepdims=True)
                        v_r = np.divide(mat_t, r_sums, out=np.zeros_like(mat_t, dtype=float), where=r_sums!=0)
                    else:
                        v_r = metrics.calc_rate(mat_t, mat_s, mode)
                        
                    fig_indiv, ax_indiv = plt.subplots(figsize=(6, 6))
                    im_indiv = ax_indiv.imshow(v_r, origin='upper', cmap=cmap_base, vmin=0, vmax=1.0)
                    ax_indiv.set_xticks(range(5)); ax_indiv.set_yticks(range(5))
                    ax_indiv.set_xticklabels([l.replace(' ', '\n') for l in config.LEANING_ORDER])
                    ax_indiv.set_yticklabels([config.POS_LABELS[p] for p in [5, 4, 3, 2, 1]])
                    ax_indiv.set_title(f"{chamber} Chamber | {'All Rounds' if not r_target else f'Round {r_target}'}")
                    fig_indiv.colorbar(im_indiv, ax=ax_indiv, fraction=0.046, pad=0.04, label="Rate")
                    
                    r_str = f"Round_{r_target}" if r_target else "AllRounds"
                    fig_indiv.savefig(indiv_dir / f"Plot_{name}_{prefix}_{chamber}Chamber_{r_str}.pdf", dpi=300, bbox_inches='tight')
                    plt.close(fig_indiv)

        if 'im' in locals(): 
            fig_main.colorbar(im, ax=axes_main.tolist(), fraction=0.015, pad=0.04, label="Rate")
        plt.savefig(output_dir / f"Plot_{name}_{prefix}_SideBySide.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig_main)

def generate_correlation_heatmaps(phase2_csv, output_dir, topics=['climate'], empirical=False, phase1_csv=None):
    phase2_csv = Path(phase2_csv)
    output_dir = Path(output_dir)
    
    if phase1_csv is None: phase1_csv = phase2_csv.parent / 'all_apps_wide_1.csv'
    else: phase1_csv = Path(phase1_csv)
    
    df2 = pd.read_csv(phase2_csv, low_memory=False)
    
    if phase1_csv.exists():
        df1 = pd.read_csv(phase1_csv, low_memory=False)
        if 'participant.label' in df1.columns and 'participant.label' in df2.columns: 
            df = pd.merge(df2, df1, on='participant.label', how='left', suffixes=('', '_p1'))
        elif 'participant.code' in df1.columns and 'participant.code' in df2.columns: 
            df = pd.merge(df2, df1, on='participant.code', how='left', suffixes=('', '_p1'))
        else: df = df2
    else: df = df2
        
    if 'participant._is_bot' in df.columns: 
        df = df[df['participant._is_bot'] == 0]
        
    is_dual = len(topics) > 1
    
    for topic in topics:
        topic_out_dir = output_dir / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        
        op_cols = []
        if is_dual: targets = [f'{topic}_opinion_1', f'{topic}_opinion_2']
        else: targets = [f'{topic}_opinion_1', f'{topic}_opinion_2', f'{topic}_opinion_3', f'{topic}_opinion_4', 
                         'opinion_1', 'opinion_2', 'opinion_3', 'opinion_4', 
                         'baseline_opinion_1', 'baseline_opinion_2', 'baseline_opinion_3', 'baseline_opinion_4']
            
        for t in targets:
            match = next((c for c in df.columns if t in c and ('phase_1' in c or 'participant' in c)), None)
            if match and match not in op_cols: op_cols.append(match)
                
        op_cols = sorted(list(set(op_cols)))
        if not op_cols: continue
        
        data_rows = []
        leanings_5 = ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']
        map_5_to_3 = {'Left': 'Left', 'Lean Left': 'Left', 'Center': 'Center', 'Lean Right': 'Right', 'Right': 'Right'}
        
        for _, row in df.iterrows():
            seen_5, shared_5 = {k: 0 for k in leanings_5}, {k: 0 for k in leanings_5}
            op_vals = {col: pd.to_numeric(row.get(col), errors='coerce') for col in op_cols}
            
            for r in range(1, config.ROUNDS + 1):
                r_cols = [c for c in df.columns if f'phase_2.{r}.player.' in c]
                f_col = next((c for c in r_cols if 'incoming_feed' in c and (topic in c if is_dual else True)), None)
                s_col = next((c for c in r_cols if 'outgoing_shares' in c and (topic in c if is_dual else True)), None)
                if not f_col or not s_col: continue
                
                feed_raw, shares_raw = row.get(f_col, '[]'), row.get(s_col, '[]')
                feed = ast.literal_eval(feed_raw) if isinstance(feed_raw, str) and feed_raw.strip() != '' else []
                shares = ast.literal_eval(shares_raw) if isinstance(shares_raw, str) and shares_raw.strip() != '' else []
                
                for item in feed:
                    if isinstance(item, dict) and 'id' in item:
                        parts = str(item['id']).split('-')
                        lean = {'L': 'Left', 'LL': 'Lean Left', 'C': 'Center', 'LR': 'Lean Right', 'R': 'Right'}.get(parts[1], 'Center') if len(parts) >= 2 else 'Center'
                        seen_5[lean] += 1
                for iid in shares:
                    parts = str(iid).split('-')
                    lean = {'L': 'Left', 'LL': 'Lean Left', 'C': 'Center', 'LR': 'Lean Right', 'R': 'Right'}.get(parts[1], 'Center') if len(parts) >= 2 else 'Center'
                    shared_5[lean] += 1
                    
            row_data = op_vals.copy()
            seen_3, shared_3 = {'Left': 0, 'Center': 0, 'Right': 0}, {'Left': 0, 'Center': 0, 'Right': 0}
            
            for lean in leanings_5:
                mapped = map_5_to_3[lean]
                seen_3[mapped] += seen_5[lean]
                shared_3[mapped] += shared_5[lean]
                row_data[f'ShareRate_5_{lean}'] = shared_5[lean] / seen_5[lean] if seen_5[lean] > 0 else np.nan
            for lean in ['Left', 'Center', 'Right']:
                row_data[f'ShareRate_3_{lean}'] = shared_3[lean] / seen_3[lean] if seen_3[lean] > 0 else np.nan
            data_rows.append(row_data)

        plot_df = pd.DataFrame(data_rows).dropna(subset=op_cols, how='any')
        if plot_df.empty: continue
            
        col_labels = []
        for c in op_cols:
            if 'opinion_1' in c: col_labels.append('Op 1')
            elif 'opinion_2' in c: col_labels.append('Op 2')
            elif 'opinion_3' in c: col_labels.append('Op 3')
            elif 'opinion_4' in c: col_labels.append('Op 4')
            else: col_labels.append(c.split('.')[-1])
        
        def _plot_hm(leanings, pref, title, filename):
            corr_matrix = np.zeros((len(leanings), len(op_cols)))
            for r_idx, lean in enumerate(leanings):
                for c_idx, col in enumerate(op_cols):
                    tdf = plot_df.dropna(subset=[col, f'ShareRate_{pref}_{lean}'])
                    if not tdf.empty: corr_matrix[r_idx, c_idx] = tdf[col].corr(tdf[f'ShareRate_{pref}_{lean}'])
                    else: corr_matrix[r_idx, c_idx] = np.nan
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pd.DataFrame(corr_matrix, index=[f"{l} Items" for l in leanings], columns=col_labels), annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, ax=ax)
            plt.savefig(topic_out_dir / filename, dpi=300)
            plt.close(fig)

        _plot_hm(['Left', 'Center', 'Right'], '3', '3-Bucket Correlation', 'Plot_07A_Corr.pdf')
        if empirical: 
            _plot_hm(leanings_5, '5', '5-Bucket Correlation', 'Plot_07C_Corr.pdf')

def plot_awareness_survey(phase2_csv_path, event_log_df, output_dir, topics):
    """
    Plots the specific post-experiment awareness survey questions 
    against the actual measured feed slant of the participant.
    """
    df2 = pd.read_csv(phase2_csv_path, low_memory=False)
    output_dir = Path(output_dir)
    
    for topic in topics:
        topic_out_dir = output_dir / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Exact variable names as defined in oTree models.py
        target_vars = [f"{topic}_perceived_slant", f"{topic}_perceived_agreement"]
        
        for var_name in target_vars:
            # Find the exact column in the exported CSV (it will usually be in the final round's column list)
            s_col = next((c for c in df2.columns if c.endswith(f"player.{var_name}")), None)
            
            if not s_col: 
                continue
                
            df2['uid'] = df2.get('participant.code', df2.index).astype(str).str.upper()
            survey_df = df2[['uid', s_col]].dropna().drop_duplicates(subset=['uid'])
            survey_df[s_col] = pd.to_numeric(survey_df[s_col], errors='coerce')
            survey_df = survey_df.dropna()
            
            if survey_df.empty: 
                continue
            
            # Calculate actual Mean Feed Slant per user
            topic_events = event_log_df[(event_log_df['topic'] == topic) & (event_log_df['action'] == 'seen')].copy()
            mapping = {'Left': -2, 'Center-Left': -1, 'Center': 0, 'Center-Right': 1, 'Right': 2}
            topic_events['slant'] = topic_events['item_cat'].map(mapping)
            
            user_actuals = topic_events.groupby('user_id')['slant'].mean().reset_index()
            user_actuals.columns = ['uid', 'Actual_Feed_Slant']
            
            merged = pd.merge(survey_df, user_actuals, on='uid', how='inner')
            if merged.empty: 
                continue
            
            # Determine clean title labels based on the variable
            is_slant_plot = "slant" in var_name
            clean_title = "Perceived Slant" if is_slant_plot else "Perceived Agreement"
            y_label_anchors = "1 = Against, 3 = Balanced, 5 = In Favour" if is_slant_plot else "1 = None, 5 = All"
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(data=merged, x='Actual_Feed_Slant', y=s_col, scatter_kws={'alpha':0.6, 's':60, 'color':'#34495e'}, line_kws={'color':'#e74c3c'}, ax=ax)
            
            ax.set_xlabel(f"Actual Measured Feed Slant\n(-2 = Extreme Left, +2 = Extreme Right)", fontweight='bold', fontsize=11)
            ax.set_ylabel(f"{clean_title} (Survey)\n[{y_label_anchors}]", fontweight='bold', fontsize=11)
            ax.set_title(f"Awareness Check: Actual Slant vs. {clean_title} ({topic.title()})", pad=15, fontsize=13)
            
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_ylim(0.5, 5.5)
            ax.set_xlim(-2.2, 2.2)
            ax.grid(True, alpha=0.3)
            
            corr = merged['Actual_Feed_Slant'].corr(merged[s_col])
            ax.annotate(f"Pearson r = {corr:.2f}", xy=(0.05, 0.90), xycoords='axes fraction', 
                        fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
            
            plt.savefig(topic_out_dir / f"Plot_08_Awareness_{clean_title.replace(' ', '_')}.pdf", dpi=300, bbox_inches='tight')
            plt.close(fig)