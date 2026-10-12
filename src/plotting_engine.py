import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import config
from src import stats_suite

def plot_feed_slant_variance(event_log_df, output_dir, topic, prefix=""):
    """
    Plots the mean feed slant per round as grouped bar charts with variance (std) error bars.
    Plots 4 columns per round: Segregated Left, Segregated Right, Integrated Left-Leaning, Integrated Right-Leaning.
    """
    df = event_log_df[(event_log_df['topic'] == topic) & (event_log_df['action'] == 'seen')].copy()
    if df.empty:
        return

    mapping = {'Left': -2, 'Center-Left': -1, 'Center': 0, 'Center-Right': 1, 'Right': 2}
    df['slant'] = df['item_cat'].map(mapping)
    df = df.dropna(subset=['slant'])
    
    seg_left_df = df[(df['treatment'] == 'segregated') & (df['chamber'] == 'Left')]
    seg_left_grp = seg_left_df.groupby('round')['slant'].agg(['mean', 'std'])
    
    seg_right_df = df[(df['treatment'] == 'segregated') & (df['chamber'] == 'Right')]
    seg_right_grp = seg_right_df.groupby('round')['slant'].agg(['mean', 'std'])
    
    int_left_df = df[(df['treatment'] == 'integrated') & (df['user_cat'].isin(['Left', 'Center-Left']))]
    int_left_grp = int_left_df.groupby('round')['slant'].agg(['mean', 'std'])
    
    int_right_df = df[(df['treatment'] == 'integrated') & (df['user_cat'].isin(['Right', 'Center-Right']))]
    int_right_grp = int_right_df.groupby('round')['slant'].agg(['mean', 'std'])

    rounds = sorted(df['round'].unique())
    seg_left_grp = seg_left_grp.reindex(rounds).fillna(0)
    seg_right_grp = seg_right_grp.reindex(rounds).fillna(0)
    int_left_grp = int_left_grp.reindex(rounds).fillna(0)
    int_right_grp = int_right_grp.reindex(rounds).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(rounds))
    width = 0.20
    
    c_seg_l = '#2980b9'
    c_seg_r = '#c0392b'
    c_int_l = '#7fb3d5'
    c_int_r = '#d98880'

    ax.bar(x - 1.5*width, seg_left_grp['mean'], width, yerr=seg_left_grp['std'], 
           capsize=4, color=c_seg_l, edgecolor='black', zorder=3, label='Segregated (Left Chamber)')
           
    ax.bar(x - 0.5*width, seg_right_grp['mean'], width, yerr=seg_right_grp['std'], 
           capsize=4, color=c_seg_r, edgecolor='black', zorder=3, label='Segregated (Right Chamber)')
           
    ax.bar(x + 0.5*width, int_left_grp['mean'], width, yerr=int_left_grp['std'], 
           capsize=4, color=c_int_l, edgecolor='black', zorder=3, label='Integrated (Left-Leaning Users)')
           
    ax.bar(x + 1.5*width, int_right_grp['mean'], width, yerr=int_right_grp['std'], 
           capsize=4, color=c_int_r, edgecolor='black', zorder=3, label='Integrated (Right-Leaning Users)')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=2)
    
    #plt.title(f'Evolution of Feed Slant and Ideological Variance ({topic.upper()})', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Simulation Round', fontsize=12)
    plt.ylabel('Mean Feed Slant (-2 to +2)', fontsize=12)
    
    plt.xticks(x, [f"Round {int(r)}" for r in rounds], fontsize=11)
    plt.ylim(-2.5, 2.5)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.95, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    
    safe_prefix = f"{prefix}_" if prefix else ""
    out_name = output_dir / f"Plot_00_{safe_prefix}Feed_Slant_Variance_{topic}.pdf"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


def generate_suite(event_log_df, output_dir, prefix="", topics=['climate']):
    if event_log_df.empty or 'treatment' not in event_log_df.columns: return
    output_dir = Path(output_dir)

    is_sim = "Sim" in prefix or prefix == ""
    
    for topic in topics:
        topic_df = event_log_df[event_log_df['topic'] == topic] if 'topic' in event_log_df.columns else event_log_df
        topic_out_dir = output_dir / topic
        topic_out_dir.mkdir(parents=True, exist_ok=True)
        
        # PLOT 01 GENERATION: Passed the full topic_df (combined treatments) directly here
        plot_chronological_feed_amp(topic_df, topic_out_dir, prefix)
        
        for treatment in topic_df['treatment'].unique():
            t_df = topic_df[topic_df['treatment'] == treatment]
            safe_prefix = f"{treatment}" if not prefix else f"{prefix}_{treatment}"
            
            plot_temporal_heatmaps(t_df, topic_out_dir, safe_prefix)
            plot_feed_sources(t_df, topic_out_dir, safe_prefix)
            plot_exposure_distributions(t_df, topic_out_dir, safe_prefix, is_sim)


def plot_chronological_feed_amp(df, output_dir, prefix=""):
    if df.empty: return
    
    if 'chamber' not in df.columns:
        df = df.copy()
        df['chamber'] = 'Unknown'
        
    cats = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
    colors = {'Left': '#2980b9', 'Center-Left': '#7fb3d5', 'Center': '#bdc3c7', 'Center-Right': '#d98880', 'Right': '#c0392b'}
    rounds = sorted(df['round'].unique())
    x_indices = np.arange(len(rounds))
    width = 0.20
    
    for action, plot_name, title in [('seen', '01A_Incoming_Diet', 'Incoming Feed (Diet)'), ('shared', '01B_Outgoing_Amp', 'Outgoing Shares (Amplification)')]:
        act_df = df[df['action'] == action]
        if act_df.empty: continue
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        data_sl = {c: [] for c in cats}
        data_sr = {c: [] for c in cats}
        data_il = {c: [] for c in cats}
        data_ir = {c: [] for c in cats}
        
        for r in rounds:
            r_df = act_df[act_df['round'] == r]
            
            sl_sub = r_df[(r_df['treatment'] == 'segregated') & (r_df['chamber'] == 'Left')]
            sr_sub = r_df[(r_df['treatment'] == 'segregated') & (r_df['chamber'] == 'Right')]
            il_sub = r_df[(r_df['treatment'] == 'integrated') & (r_df['user_cat'].isin(['Left', 'Center-Left']))]
            ir_sub = r_df[(r_df['treatment'] == 'integrated') & (r_df['user_cat'].isin(['Right', 'Center-Right']))]
            
            def calc_props(sub):
                tot = len(sub) if len(sub) > 0 else 1
                return {c: len(sub[sub['item_cat'] == c]) / tot for c in cats}
                
            p_sl = calc_props(sl_sub)
            p_sr = calc_props(sr_sub)
            p_il = calc_props(il_sub)
            p_ir = calc_props(ir_sub)
            
            for c in cats:
                data_sl[c].append(p_sl[c])
                data_sr[c].append(p_sr[c])
                data_il[c].append(p_il[c])
                data_ir[c].append(p_ir[c])
                
        bottom_sl, bottom_sr, bottom_il, bottom_ir = np.zeros(len(rounds)), np.zeros(len(rounds)), np.zeros(len(rounds)), np.zeros(len(rounds))
        
        for c in cats:
            # S-L
            ax.bar(x_indices - 1.5*width, data_sl[c], width, bottom=bottom_sl, color=colors[c], edgecolor='black', linewidth=1.2)
            bottom_sl += np.array(data_sl[c])
            # S-R
            ax.bar(x_indices - 0.5*width, data_sr[c], width, bottom=bottom_sr, color=colors[c], edgecolor='black', linewidth=1.2)
            bottom_sr += np.array(data_sr[c])
            # I-L
            ax.bar(x_indices + 0.5*width, data_il[c], width, bottom=bottom_il, color=colors[c], edgecolor='white', linewidth=0.5, alpha=0.9)
            bottom_il += np.array(data_il[c])
            # I-R
            ax.bar(x_indices + 1.5*width, data_ir[c], width, bottom=bottom_ir, color=colors[c], edgecolor='white', linewidth=0.5, alpha=0.9)
            bottom_ir += np.array(data_ir[c])
            
        ax.set_ylabel("Proportion of Volume")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"Round {int(r)}" for r in rounds], fontsize=11, fontweight='bold')
        
        legend_elements = [mpatches.Patch(color=colors[c], label=c) for c in cats]
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.2, label='Segregated (S)'))
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='lightgray', linewidth=1, label='Integrated (I)'))
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=10)
        
        for i in x_indices:
            ax.text(i - 1.5*width, -0.02, 'S-L', ha='center', va='top', fontsize=9, rotation=90)
            ax.text(i - 0.5*width, -0.02, 'S-R', ha='center', va='top', fontsize=9, rotation=90)
            ax.text(i + 0.5*width, -0.02, 'I-L', ha='center', va='top', fontsize=9, rotation=90)
            ax.text(i + 1.5*width, -0.02, 'I-R', ha='center', va='top', fontsize=9, rotation=90)

        plt.subplots_adjust(bottom=0.15)
        
        title_prefix = f"({prefix})" if prefix else ""
        #plt.title(f"{title} by Configuration\n{title_prefix}", pad=30, fontweight='bold', fontsize=14)
        
        filename = f"Plot_{plot_name}_Combined_{prefix}.pdf" if prefix else f"Plot_{plot_name}_Combined.pdf"
        plt.savefig(output_dir / filename, dpi=300)
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
    
    left_mask = merged['user_cat'].isin(['Left', 'Center-Left'])
    center_mask = merged['user_cat'] == 'Center'
    right_mask = merged['user_cat'].isin(['Right', 'Center-Right'])
    has_center = center_mask.sum() > 0
    
    def get_avg_dist(data):
        if data.empty: return pd.Series(dtype=float)
        counts = data.groupby(['trial_id', 'total_shares']).size().unstack(fill_value=0)
        return counts.mean(axis=0)

    l_dist = get_avg_dist(merged[left_mask])
    c_dist = get_avg_dist(merged[center_mask]) if has_center else pd.Series(dtype=float)
    r_dist = get_avg_dist(merged[right_mask])
    
    max_shares = int(max([d.index.max() if not d.empty else 0 for d in [l_dist, c_dist, r_dist]]))
    l_dist = l_dist.reindex(range(max_shares + 1), fill_value=0)
    c_dist = c_dist.reindex(range(max_shares + 1), fill_value=0) if has_center else c_dist
    r_dist = r_dist.reindex(range(max_shares + 1), fill_value=0)
    
    x_ticks = np.arange(max_shares + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if has_center:
        width = 0.25
        ax.bar(x_ticks - width, l_dist, width, color='#2980b9', edgecolor='black', zorder=3, label='Combined Left (L/CL)')
        ax.bar(x_ticks, c_dist, width, color='#bdc3c7', edgecolor='black', zorder=3, label='Pure Center (C)')
        ax.bar(x_ticks + width, r_dist, width, color='#c0392b', edgecolor='black', zorder=3, label='Combined Right (R/CR)')
    else:
        width = 0.35
        ax.bar(x_ticks - width/2, l_dist, width, color='#2980b9', edgecolor='black', zorder=3, label='Combined Left (L/CL)')
        ax.bar(x_ticks + width/2, r_dist, width, color='#c0392b', edgecolor='black', zorder=3, label='Combined Right (R/CR)')
    
    ax.set_xlabel("Total Items Shared")
    ax.set_ylabel("Avg Participants" if is_sim else "Number of Participants")
    ax.set_xticks(x_ticks)
    if not is_sim:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.legend(title="Participant Leaning", title_fontsize=11)
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    left_mask = merged['user_cat'].isin(['Left', 'Center-Left'])
    center_mask = merged['user_cat'] == 'Center'
    right_mask = merged['user_cat'].isin(['Right', 'Center-Right'])
    has_center = center_mask.sum() > 0
    x_ticks = np.arange(5)
    
    def get_avg_dist(data, ic):
        if data.empty: return np.zeros(5)
        counts = data[ic].astype(int)
        grouped = data.assign(cnt=counts).groupby(['trial_id', 'round'])['cnt'].value_counts().unstack(fill_value=0)
        for c in range(5):
            if c not in grouped.columns: grouped[c] = 0
        return grouped[[0, 1, 2, 3, 4]].mean(axis=0).values

    for idx, ic in enumerate(config.ITEM_CATS):
        ax = axes[idx]
        l_dist = get_avg_dist(merged[left_mask], ic)
        c_dist = get_avg_dist(merged[center_mask], ic) if has_center else None
        r_dist = get_avg_dist(merged[right_mask], ic)
        
        if has_center:
            width = 0.25
            ax.bar(x_ticks - width, l_dist, width, color='#2980b9', edgecolor='black', zorder=3, label='Left (L/CL)')
            ax.bar(x_ticks, c_dist, width, color='#bdc3c7', edgecolor='black', zorder=3, label='Center (C)')
            ax.bar(x_ticks + width, r_dist, width, color='#c0392b', edgecolor='black', zorder=3, label='Right (R/CR)')
        else:
            width = 0.35
            ax.bar(x_ticks - width/2, l_dist, width, color='#2980b9', edgecolor='black', zorder=3, label='Left (L/CL)')
            ax.bar(x_ticks + width/2, r_dist, width, color='#c0392b', edgecolor='black', zorder=3, label='Right (R/CR)')
        
        ax.set_ylabel("Avg Participants" if is_sim else "Participants")
        ax.set_xlabel("Number of Items Seen")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.grid(True, axis='y', alpha=0.3, zorder=0)

    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=14, title="User Demographics", title_fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_05_Exposure_Dist_{prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feed_sources(df, output_dir, prefix):
    multi_metrics = stats_suite.calculate_sourcing_metrics(df)
    if not multi_metrics or 'Global' not in multi_metrics: return
    
    metrics = multi_metrics['Global']
    rounds = metrics['rounds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, metrics['rep_avg'], marker='o', lw=2, color='#e74c3c', label='Average Repeats')
    ax.plot(rounds, metrics['pool_avg'], marker='s', lw=2, color='#3498db', label='Average Pool')
    ax.plot(rounds, metrics['back_avg'], marker='^', lw=2, color='#27ae60', label='Average Backlog')
    ax.plot(rounds, metrics['hid_avg'], marker='d', lw=2, color='#8e44ad', label='Average Hidden')
    ax.plot(rounds, metrics['cross_avg'], marker='v', lw=2, color='#f39c12', label='Average Cross-Chamber')
    
    ax.plot(rounds, metrics['rep_max'], marker='x', lw=2, ls='--', color='#e74c3c')
    ax.plot(rounds, metrics['pool_max'], marker='x', lw=2, ls='--', color='#3498db')
    ax.plot(rounds, metrics['back_max'], marker='x', lw=2, ls='--', color='#27ae60')
    ax.plot(rounds, metrics['hid_max'], marker='x', lw=2, ls='--', color='#8e44ad')
    ax.plot(rounds, metrics['cross_max'], marker='x', lw=2, ls='--', color='#f39c12')
    
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
                mat_seen = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'seen'], None)
                mat_share = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'shared'], None)
                mat_trg = mat_share if 'Share' in name else mat_seen
                
                if mode == 'norm_vol':
                    row_sums = mat_seen.sum(axis=1, keepdims=True)
                    v_rate = np.divide(mat_trg, row_sums, out=np.zeros_like(mat_trg, dtype=float), where=row_sums!=0)
                else:
                    v_rate = stats_suite.calc_rate(mat_trg, mat_seen, mode)
                    
                im = ax_main.imshow(v_rate, origin='upper', cmap=cmap_base, vmin=0, vmax=1.0)
                ax_main.set_title(f"{chamber} Chamber", fontweight='bold')
                ax_main.set_xticks(range(5)); ax_main.set_yticks(range(5))
                ax_main.set_xticklabels([l.replace(' ', '\n') for l in config.LEANING_ORDER])
                if idx == 0: ax_main.set_yticklabels([config.POS_LABELS[p] for p in [5, 4, 3, 2, 1]])
                if idx == 0: ax_main.set_ylabel("User Baseline Opinion", fontsize=12)
            
            if not chamber_df.empty:
                for r_target in [None] + list(range(1, config.ROUNDS + 1)):
                    mat_s = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'seen'], r_target)
                    mat_sh = stats_suite.build_matrix(chamber_df[chamber_df['action'] == 'shared'], r_target)
                    mat_t = mat_sh if 'Share' in name else mat_s
                    if mode == 'norm_vol':
                        r_sums = mat_s.sum(axis=1, keepdims=True)
                        v_r = np.divide(mat_t, r_sums, out=np.zeros_like(mat_t, dtype=float), where=r_sums!=0)
                    else:
                        v_r = stats_suite.calc_rate(mat_t, mat_s, mode)
                        
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
        #plt.suptitle(f"{name.replace('_', ' ')} (Aggregate)\n{prefix}", fontsize=14, y=1.05)
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